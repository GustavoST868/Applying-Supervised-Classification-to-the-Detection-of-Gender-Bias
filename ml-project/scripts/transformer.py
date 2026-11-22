# fine-tuned transformer: deberta-v3-base with the
# hugging face transformers library
#
# usage: python scripts/transformer.py transformer --run-dir ... --config ...
#
# for each scenario (full and masked text):
# 1. exhaustive search (optuna GridSampler) over the grid of learning
#    rate and batch size of config.yaml
#    each configuration is trained on the official edos train split
#    for up to transformer.epochs epochs and evaluated after each epoch
#    on the official validation split, and the value of the configuration
#    is the best auprc among the epochs
# 2. the best configuration is trained again with each seed of
#    transformer.seeds, the model of the seed with the highest auprc
#    on validation stays, with the best epoch, and the variation among
#    the seeds is reported
# 3. the threshold that maximizes the macro f1 on the validation
#    predictions is saved with the model, which is copied to
#    models/<run>/transformer_<scenario>
#
# the base model is downloaded at the revision pinned in transformer.revision
# the partial state (optuna study and seeds already trained) stays
# in results/runs/<run>/transformer/<stage digest>,
# so an interrupted run is resumed, and a change in the
# code, in the configuration or in the data starts from scratch

import json
import shutil

import numpy as np
import optuna
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pipeline import StageFailed, measure, run_stage_cli
from common import (
    SCENARIOS,
    TEXT_COLUMN,
    auprc,
    drop_stale_studies,
    load_partition,
    macro_f1_threshold,
    optuna_storage,
    study_name,
)
from judges import unload_all_models

# batch size in the evaluation and in the prediction
EVAL_BATCH = 64

# fields that identify the installed model
INSTALLED_KEYS = ("stage_hash", "seed")

# posts already tokenized, with the label when there is one
class Encoded(torch.utils.data.Dataset):

    def __init__(self, tokenizer, texts, labels=None, max_length=512):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        if labels is None:
            self.labels = None
        else:
            self.labels = []
            for value in labels:
                self.labels.append(int(value))

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {}
        for key, values in self.encodings.items():
            item[key] = values[index]
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item

# probability of the sexist class from the logits
def positive_probability(logits):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    return torch.softmax(logits, dim=-1)[:, 1].numpy()

# metric the trainer calculates after each epoch
def compute_metrics(prediction):
    scores = positive_probability(prediction.predictions)
    return {"auprc": auprc(prediction.label_ids, scores)}

# training arguments of the trainer
# with save=True, keeps one checkpoint per epoch and goes back
# to the best epoch at the end of the training
def training_arguments(context, output_dir, learning_rate, batch_size, seed, save):
    settings = context.config["transformer"]
    if save:
        save_strategy = "epoch"
    else:
        save_strategy = "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH,
        num_train_epochs=settings["epochs"],
        warmup_ratio=settings["warmup"],
        weight_decay=settings["weight_decay"],
        adam_epsilon=settings["adam_epsilon"],
        adam_beta2=settings["adam_beta2"],
        max_grad_norm=settings["max_grad_norm"],
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=save,
        metric_for_best_model="auprc",
        greater_is_better=True,
        logging_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=seed,
        data_seed=seed,
        report_to="none",
        disable_tqdm=True,
    )

# working folder of this version of the stage,
# inside results/runs/<run>
# I used a dummy "x" file only so that context.path
# creates the folder, and kept its folder
def work_dir(context, *parts):
    return context.path("transformer", context.short_hash, *parts, "x").parent

# deletes the partial state left by other versions of the stage
def drop_stale_work(context):
    root = context.run_dir / "transformer"
    if not root.exists():
        return

    for item in root.iterdir():
        if item.name != context.short_hash:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

# epoch with the highest auprc on validation
def best_epoch(history):
    best = history[0]
    for item in history:
        if item["auprc"] > best["auprc"]:
            best = item
    return best

# trains the model and returns the trainer and the auprc of each epoch
def train(context, tokenizer, train_set, valid_set, output_dir, learning_rate, batch_size, seed,
          save, scenario):
    settings = context.config["transformer"]

    # the trainer calls this function to create a new
    # model, with the training seed already set
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            settings["model"],
            revision=settings["revision"],
            num_labels=2
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_arguments(context, output_dir, learning_rate, batch_size, seed, save),
        train_dataset=train_set,
        eval_dataset=valid_set,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    with measure(
        context,
        method=f"transformer_{scenario}",
        scenario=scenario,
        phase="training",
        n_documents=len(train_set)
    ):
        trainer.train()

    # auprc of each epoch, taken from the trainer log
    history = []
    for item in trainer.state.log_history:
        if "eval_auprc" in item:
            history.append({"epoch": int(round(item["epoch"])), "auprc": item["eval_auprc"]})

    return trainer, history

# search grid of config.yaml
def grid(context):
    settings = context.config["transformer"]["grid"]
    return {"learning_rate": settings["learning_rate"], "batch_size": settings["batch_size"]}

# exhaustive search over the grid, with the first seed
# the value of each configuration is the best auprc among the epochs
def search(context, scenario, tokenizer, train_set, valid_set):
    space = grid(context)
    size = len(space["learning_rate"]) * len(space["batch_size"])
    study = optuna.create_study(
        study_name=study_name(context, "transformer", scenario),
        storage=optuna_storage(context),
        direction="maximize",
        sampler=optuna.samplers.GridSampler(space, seed=context.seed),
        load_if_exists=True,
    )
    seed = context.config["transformer"]["seeds"][0]
    output = context.path("cache", "transformer", scenario, "search")

    def objective(trial):
        learning_rate = trial.suggest_categorical("learning_rate", space["learning_rate"])
        batch_size = trial.suggest_categorical("batch_size", space["batch_size"])
        _, history = train(
            context,
            tokenizer,
            train_set,
            valid_set,
            output,
            learning_rate,
            batch_size,
            seed,
            save=False,
            scenario=scenario
        )

        values = []
        for item in history:
            values.append(item["auprc"])
        trial.set_user_attr("auprc_per_epoch", values)

        best = best_epoch(history)
        trial.set_user_attr("best_epoch", best["epoch"])
        return best["auprc"]

    # runs only the configurations that have not finished yet
    done = 0
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            done += 1

    if done < size:
        study.optimize(objective, n_trials=size - done)

    return study

# trains the best configuration with each seed
# the result of each seed goes to seeds.json as soon as
# it finishes, so an interrupted run skips the ones already done
def train_seeds(context, scenario, tokenizer, train_set, valid_set, valid_labels, best):
    settings = context.config["transformer"]
    results_file = work_dir(context, scenario) / "seeds.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
    else:
        results = {}

    for seed in settings["seeds"]:
        if str(seed) in results:
            continue

        folder = work_dir(context, scenario, f"seed_{seed}")
        trainer, history = train(
            context,
            tokenizer,
            train_set,
            valid_set,
            folder,
            best["learning_rate"],
            best["batch_size"],
            seed,
            save=True,
            scenario=scenario
        )

        # validation scores with the best epoch
        scores = positive_probability(trainer.predict(valid_set).predictions)

        # saves the model and deletes the checkpoints, which are no longer used
        model_dir = folder / "model"
        trainer.save_model(str(model_dir))
        for checkpoint in folder.glob("checkpoint-*"):
            shutil.rmtree(checkpoint, ignore_errors=True)
        np.save(folder / "validation_scores.npy", scores)

        values = []
        for item in history:
            values.append(item["auprc"])

        results[str(seed)] = {
            "validation_auprc": auprc(valid_labels, scores),
            "best_epoch": best_epoch(history)["epoch"],
            "auprc_per_epoch": values,
        }
        results_file.write_text(json.dumps(results, indent=2))

        # frees the gpu memory before the next seed
        del trainer
        torch.cuda.empty_cache()

    return results

# copies the model of the chosen seed to
# models/<run>/transformer_<scenario>
# the copy goes to a temporary folder first, and the model
# of the seed is only deleted after the copy is in place
# if it was already deleted, the stage was interrupted after the
# copy, and the installed model is kept when it is from the same
# version of the stage and from the same seed
def install_model(context, scenario, folder, record):
    destination = context.path("models", f"transformer_{scenario}", "x").parent
    source = folder / "model"

    if source.exists():
        staging = destination.parent / f".partial_{destination.name}"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging)
        (staging / "threshold.json").write_text(json.dumps(record, indent=2))
        shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)
        shutil.rmtree(source)
        return

    installed = destination / "threshold.json"
    if installed.exists():
        previous = json.loads(installed.read_text())
    else:
        previous = {}

    for key in INSTALLED_KEYS:
        if previous.get(key) != record[key]:
            raise StageFailed(f"the model of seed {record['seed']} is neither in {source} nor in {destination}")

    installed.write_text(json.dumps(record, indent=2))

# transformer stage
def run_transformer(context):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not torch.cuda.is_available():
        context.alert("no GPU with CUDA: the transformer fine-tuning will run on the CPU, very slowly.")

    # takes the ollama models off the gpu to leave vram for the training
    unload_all_models()
    drop_stale_work(context)
    drop_stale_studies(context, "transformer_")

    settings = context.config["transformer"]
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["revision"])
    train_part = load_partition(context, "train")
    valid_part = load_partition(context, "validation")

    # some tokenizers report a huge maximum length
    # when they have no limit, and in that case I used 512 tokens
    if tokenizer.model_max_length < 100000:
        limit = tokenizer.model_max_length
    else:
        limit = 512

    grid_rows = []
    seed_rows = []
    summary = []
    for scenario in SCENARIOS:
        column = TEXT_COLUMN[scenario]

        # warns how many posts will be cut at the token limit
        all_texts = list(train_part[column]) + list(valid_part[column])
        lengths = []
        for ids in tokenizer(all_texts)["input_ids"]:
            lengths.append(len(ids))
        if max(lengths) > limit:
            too_long = 0
            for length in lengths:
                if length > limit:
                    too_long += 1
            context.alert(f"{too_long} posts go over {limit} tokens.")

        train_set = Encoded(tokenizer, train_part[column], train_part["target"], limit)
        valid_set = Encoded(tokenizer, valid_part[column], valid_part["target"], limit)
        valid_labels = valid_part["target"].to_numpy()

        # search over the grid
        print(f"[search] transformer, {scenario} scenario", flush=True)
        study = search(context, scenario, tokenizer, train_set, valid_set)
        for trial in study.trials:
            row = {"scenario": scenario, "trial": trial.number}
            row.update(trial.params)
            row["validation_auprc"] = trial.value
            row.update(trial.user_attrs)
            grid_rows.append(row)
        best = study.best_params

        # training with each seed
        print(f"[seeds] transformer, {scenario} scenario: {best}", flush=True)
        results = train_seeds(context, scenario, tokenizer, train_set, valid_set, valid_labels, best)
        for seed, item in results.items():
            row = {"scenario": scenario, "seed": int(seed)}
            row.update(best)
            row.update(item)
            seed_rows.append(row)

        # seed with the highest auprc on validation
        chosen = None
        for seed in results:
            if chosen is None or results[seed]["validation_auprc"] > results[chosen]["validation_auprc"]:
                chosen = seed

        # threshold chosen on the validation predictions
        folder = work_dir(context, scenario, f"seed_{chosen}")
        scores = np.load(folder / "validation_scores.npy")
        threshold = macro_f1_threshold(valid_labels, scores)

        values = []
        for item in results.values():
            values.append(item["validation_auprc"])

        record = {
            "base_model": settings["model"],
            "revision": settings["revision"],
            "stage_hash": context.digest,
            "scenario": scenario,
            "seed": int(chosen),
        }
        record.update(best)
        record["best_epoch"] = results[chosen]["best_epoch"]
        record["threshold"] = threshold
        record["max_length"] = limit
        record["validation_auprc"] = results[chosen]["validation_auprc"]
        record["validation_auprc_seed_mean"] = float(np.mean(values))
        record["validation_auprc_seed_std"] = float(np.std(values))

        install_model(context, scenario, folder, record)
        summary.append(record)

        # the models of the other seeds are not used later,
        # only the auprc values and the scores stay
        for other in work_dir(context, scenario).glob("seed_*"):
            shutil.rmtree(other / "model", ignore_errors=True)

    context.save_table(pd.DataFrame(grid_rows), "transformer_grid")
    context.save_table(pd.DataFrame(seed_rows), "transformer_seeds")
    context.save_table(pd.DataFrame(summary), "transformer_chosen")
    shutil.rmtree(context.path("cache", "transformer", "x").parent, ignore_errors=True)

# probability of the sexist class for each text,
# with the model saved in model_dir
def predict_scores(model_dir, texts, batch_size=EVAL_BATCH):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
    record = json.loads((model_dir / "threshold.json").read_text())
    collator = DataCollatorWithPadding(tokenizer)
    dataset = Encoded(tokenizer, texts, max_length=record["max_length"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    # prediction in batches, in bf16 when there is a gpu
    pieces = []
    with torch.no_grad():
        for batch in loader:
            moved = {}
            for key, value in batch.items():
                moved[key] = value.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**moved).logits
            pieces.append(positive_probability(logits.float().cpu()))

    # frees the gpu memory
    del model
    torch.cuda.empty_cache()

    return np.concatenate(pieces), record

STAGE_FUNCTIONS = {"transformer": run_transformer}

if __name__ == "__main__":
    run_stage_cli(STAGE_FUNCTIONS)
