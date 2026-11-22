# llm judges served by ollama: choice of the
# prompt, stability and test
#
# usage: python scripts/judges.py <stage> --run-dir ... --config ...
#
# stages:
# - judges_dev: on a random sample of the development set, compares
#   three prompts (minimal, edos definition and definition with examples)
#   by the auprc of each judge and keeps the best one
#   then measures the stability of the judge with the chosen prompt
#   through krippendorff's alpha (krippendorff library): the same sample
#   with the original prompt and its four reformulations (the ones of
#   the minimal prompt, if it was chosen, or the ones of the definition,
#   keeping the examples when there are any)
#   the sample is also judged with three seeds, but, with
#   temperature zero, the decoding is greedy, and this repetition
#   only checks if the judge is deterministic
# - judges_test: classifies the whole test set with the chosen
#   prompt, the vote of the judge is the first word of the answer
#   (YES or NO) and the score is P(YES) / (P(YES) + P(NO)), calculated
#   from the log-probabilities of the first token
#   the consensus is the majority of the votes of the three judges, and
#   its score is the mean of the scores
#   the predictions go to predictions/judges_test.parquet and to one
#   csv per judge (and for the consensus) in results/predictions/<run>/
#   after the votes, each conversation gets a second question
#   (explanation in prompts.yaml) and the judge explains the vote in one
#   sentence, the explanation is conditioned on the vote already given
#   ("label then explanation" format of camburu et al., 2018, and
#   marasovic et al., 2022) and does not change the vote or the score
#   it serves the qualitative analysis: there is no guarantee that it
#   describes the real reason of the vote (turpin et al., 2023; wang et al., 2023)
#
# each answer is saved in
# results/runs/<run>/llm/responses/<stage>/<model>.jsonl,
# with a key that includes the digest of the model weights, the prompt,
# the text and the seed, and with the digest of the stage
# an interrupted run continues from where it stopped, and the answers of
# another version of the stage are discarded, so that the measured cost
# covers every answer used

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats

from pipeline import StageFailed, load_prompts, measure, run_stage_cli
from common import auprc, load_partition

URL = "http://localhost:11434"
POSITIVE = "YES"
NEGATIVE = "NO"
CONSENSUS = "judge_consensus"

# scenario of the judges in the predictions and in the cost,
# they only read the full text
JUDGE_SCENARIO = "judge"

# reformulations used in the stability, according to the
# chosen prompt (keys of prompts.yaml)
REFORMULATIONS = {
    "minimal": "minimal_reformulations",
    "definition": "definition_reformulations",
    "examples": "definition_reformulations",
}
MAX_ATTEMPTS = 5
SERVER_WAIT_S = 30

# sends a json to ollama and returns the answer
def post_json(path, body, timeout=900):
    request = urllib.request.Request(
        URL + path,
        data=json.dumps(body).encode("utf8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

# asks ollama for a json
def get_json(path):
    with urllib.request.urlopen(URL + path, timeout=10) as response:
        return json.load(response)

# checks if the ollama server is answering
def server_is_up():
    try:
        get_json("/api/tags")
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False

# starts the ollama server, if it is not up yet
# I set OLLAMA_NUM_PARALLEL=1 so that ollama
# handles one call at a time
def ensure_server():
    if shutil.which("ollama") is None:
        raise StageFailed("Ollama not found.")
    if server_is_up():
        return

    environment = dict(os.environ, OLLAMA_NUM_PARALLEL="1")
    subprocess.Popen(
        ["ollama", "serve"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # waits for the server to come up
    for _ in range(SERVER_WAIT_S):
        time.sleep(1)
        if server_is_up():
            return

    raise StageFailed("the Ollama server did not answer.")

# digest of the model weights in ollama, or None
# if the model was not downloaded yet
def model_digest(model):
    for item in get_json("/api/tags")["models"]:
        if item["name"] == model:
            return item["digest"]
    return None

# downloads the missing models
def ensure_models(models):
    for model in models:
        if model_digest(model) is None:
            print(f"Downloading {model}", flush=True)
            subprocess.run(["ollama", "pull", model], check=True)

# takes the model out of memory
def unload(model):
    post_json("/api/chat", {"model": model, "stream": False, "keep_alive": 0, "messages": []})

# frees the vram used by ollama models,
# if the server is up
def unload_all_models():
    if shutil.which("ollama") is None or not server_is_up():
        return

    for item in get_json("/api/ps").get("models", []):
        unload(item["name"])

# removes spaces, quotes and punctuation around the word
# and puts it in uppercase
def normalize_word(token):
    return token.strip().strip("\"'`.,:*").upper()

# P(YES) / (P(YES) + P(NO)) among the most likely
# alternatives of the first token
# returns None when the model did not send the probabilities
# or when neither YES nor NO is among the alternatives
def score_from_logprobs(logprobs):
    if not logprobs:
        return None

    positive = 0.0
    negative = 0.0
    for item in logprobs[0]["top_logprobs"]:
        word = normalize_word(item["token"])
        if word == POSITIVE:
            positive += math.exp(item["logprob"])
        elif word == NEGATIVE:
            negative += math.exp(item["logprob"])

    if positive + negative == 0:
        return None
    else:
        return positive / (positive + negative)

# vote of the judge from the first word of the answer:
# 1 for YES, 0 for NO and None for anything else
def parse_verdict(content):
    words = content.split()
    if not words:
        return None

    word = normalize_word(words[0])
    if word == POSITIVE:
        return 1
    elif word == NEGATIVE:
        return 0
    else:
        return None

# conversation sent to the judge: the system prompt, the
# examples (when there are any) and the text to be judged
def judge_messages(system, examples, text):
    messages = [{"role": "system", "content": system}]
    for example_text, answer in examples:
        messages.append({"role": "user", "content": f"Text:\n{example_text}\nAnswer:"})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": f"Text:\n{text}\nAnswer:"})
    return messages

# asks the judge for the vote on a text
def judge_text(model, system, examples, text, settings, seed):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": settings["top_logprobs"],
        "keep_alive": settings["keep_alive"],
        "messages": judge_messages(system, examples, text),
        "options": {
            "temperature": settings["temperature"],
            "seed": seed,
            "num_ctx": settings["num_ctx"],
            "num_predict": settings["num_predict"],
        },
    }
    reply = post_json("/api/chat", body)
    content = reply["message"]["content"]
    return {
        "answer": content,
        "verdict": parse_verdict(content),
        "score": score_from_logprobs(reply.get("logprobs")),
        "input_tokens": reply.get("prompt_eval_count"),
        "output_tokens": reply.get("eval_count"),
        "total_duration_ns": reply.get("total_duration"),
    }

# second question in the conversation of the vote:
# the judge explains the vote it gave
def explain_text(model, system, examples, text, answer, question, settings, seed):
    limit = settings["num_predict_explanation"]
    messages = judge_messages(system, examples, text)
    messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": question})

    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": settings["keep_alive"],
        "messages": messages,
        "options": {
            "temperature": settings["temperature"],
            "seed": seed,
            "num_ctx": settings["num_ctx"],
            "num_predict": limit,
        },
    }
    reply = post_json("/api/chat", body)

    # the explanation was cut when it used every allowed token
    tokens = reply.get("eval_count")
    truncated = tokens is not None and tokens >= limit

    return {
        "explanation": reply["message"]["content"].strip(),
        "explanation_truncated": truncated,
        "output_tokens": tokens,
        "total_duration_ns": reply.get("total_duration"),
    }

# tries the call up to MAX_ATTEMPTS times, waiting longer
# each time, and starts the server again if it went down
# if all of them fail, returns {"failed": True} so that the
# post stays without a vote instead of stopping the stage
def with_retries(call, *args):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call(*args)
        except (urllib.error.URLError, OSError, ValueError) as error:
            print(f"Call failed ({attempt + 1}/{MAX_ATTEMPTS}): {error}", flush=True)
            time.sleep(min(2 * 2**attempt, 30))
            if not server_is_up():
                ensure_server()
    return {"failed": True}

# answers already obtained by this version of the stage, by
# key, in one jsonl file per model
class ResponseStore:

    def __init__(self, context, model):
        name = model.replace(":", "_").replace("/", "_")
        self.path = context.path("llm", "responses", context.name, f"{name}.jsonl")
        self.digest = context.digest
        self.responses = {}
        if not self.path.exists():
            return

        # loads only the successful answers of this version of the stage
        lines = self.path.read_text(encoding="utf8").splitlines()
        kept = []
        for line in lines:
            record = json.loads(line)
            if record.get("stage_hash") == self.digest and not record.get("failed"):
                self.responses[record["key"]] = record
                kept.append(line)

        # answers of another version of the stage (other code,
        # configuration or data) leave the file
        if len(kept) < len(lines):
            text = "".join([line + "\n" for line in kept])
            self.path.write_text(text, encoding="utf8")

    # stored answer, or None if it does not exist yet
    def get(self, key):
        return self.responses.get(key)

    # keeps a new answer
    # the calls that failed stay only in memory, so
    # that they are tried again in the next run
    def add(self, record):
        record = dict(record)
        record["stage_hash"] = self.digest
        self.responses[record["key"]] = record
        if record.get("failed"):
            return

        with open(self.path, "a", encoding="utf8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

# key of an answer: post, text, model weights,
# prompt, examples, seed and question
def response_key(doc_id, text, digest, system, examples, seed, question=None):
    raw = json.dumps([int(doc_id), text, digest, system, examples, seed, question], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()

# classifies the posts of the frame and returns a
# table with vote and score
# with question, a second pass asks the judge for the explanation
# of each vote, and the table also has the explanation
# the two passes are measured separately, so that the cost
# of the vote does not include the cost of the explanation
def judge_batch(context, model, label, system, examples, frame, seed, phase, question=None):
    settings = context.config["judges"]
    store = ResponseStore(context, model)
    digest = model_digest(model)
    texts = frame["text"]

    # posts that do not have a stored vote yet
    keys = []
    for doc, text in zip(frame["id"], texts):
        keys.append(response_key(doc, text, digest, system, examples, seed))

    pending = []
    for position, key in enumerate(keys):
        if store.get(key) is None:
            pending.append(position)

    # asks for the vote of the missing ones
    if pending:
        with measure(
            context,
            method=judge_name(model),
            scenario=JUDGE_SCENARIO,
            phase=phase,
            n_documents=len(pending),
            ollama=True
        ):
            for position in pending:
                result = with_retries(
                    judge_text,
                    model,
                    system,
                    examples,
                    texts.iloc[position],
                    settings,
                    seed
                )
                record = {
                    "key": keys[position],
                    "id": int(frame["id"].iloc[position]),
                    "prompt": label,
                    "seed": seed,
                }
                record.update(result)
                store.add(record)

    # table with the vote and the score of each post
    votes = []
    for key in keys:
        vote = store.get(key)
        if vote is None:
            vote = {}
        votes.append(vote)

    verdicts = []
    scores = []
    for vote in votes:
        verdicts.append(vote.get("verdict"))
        scores.append(vote.get("score"))

    table = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "verdict": pd.to_numeric(verdicts, errors="coerce"),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    )
    if question is None:
        return table

    # posts with a vote that do not have a stored explanation yet
    explain_keys = []
    for doc, text in zip(frame["id"], texts):
        explain_keys.append(response_key(doc, text, digest, system, examples, seed, question))

    pending = []
    for position, key in enumerate(explain_keys):
        if store.get(key) is None and votes[position].get("answer"):
            pending.append(position)

    # asks for the explanation of the missing ones
    if pending:
        with measure(
            context,
            method=judge_name(model),
            scenario=JUDGE_SCENARIO,
            phase="explanation",
            n_documents=len(pending),
            ollama=True
        ):
            for position in pending:
                result = with_retries(
                    explain_text,
                    model,
                    system,
                    examples,
                    texts.iloc[position],
                    votes[position]["answer"],
                    question,
                    settings,
                    seed
                )
                record = {
                    "key": explain_keys[position],
                    "id": int(frame["id"].iloc[position]),
                    "prompt": label,
                    "seed": seed,
                    "kind": "explanation",
                }
                record.update(result)
                store.add(record)

    # adds the explanations to the table
    explanations = []
    truncated = []
    for key in explain_keys:
        item = store.get(key)
        if item is None:
            item = {}
        explanations.append(item.get("explanation"))
        truncated.append(item.get("explanation_truncated"))

    table["explanation"] = explanations
    table["explanation_truncated"] = truncated
    return table

# name of the judge in the predictions and in the cost
def judge_name(model):
    return "judge_" + model.replace(":", "_").replace("/", "_")

# ollama version, model weights, prompts and options
# used in the judges
# the judges_test stage compares it with the manifest of judges_dev
# to make sure nothing changed between the two
def manifest(context):
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    prompts = json.dumps(load_prompts(), sort_keys=True).encode("utf8")

    option_names = [
        "temperature",
        "seed",
        "num_ctx",
        "num_predict",
        "top_logprobs",
        "num_predict_explanation",
    ]
    options = {}
    for key in option_names:
        options[key] = context.config["judges"][key]

    digests = {}
    for model in context.config["judges"]["models"]:
        digests[model] = model_digest(model)

    return {
        "ollama_version": version,
        "digest": digests,
        "prompts_hash": hashlib.sha256(prompts).hexdigest(),
        "options": options,
    }

# samples the examples of the prompt with examples, the same
# number of sexist and not sexist posts
# also returns the ids used, so that they do not
# go into the judged sample
def pick_examples(dev, per_class, seed):
    shuffled = dev.sample(frac=1, random_state=seed)
    positives = shuffled[shuffled["target"] == 1].head(per_class)
    negatives = shuffled[shuffled["target"] == 0].head(per_class)

    examples = []
    for position in range(per_class):
        examples.append([positives["text"].iloc[position], POSITIVE])
        examples.append([negatives["text"].iloc[position], NEGATIVE])

    used = set(positives["id"]) | set(negatives["id"])
    return examples, used

# krippendorff's alpha (nominal) of the verdicts and the
# bootstrap interval over the posts
def alpha_with_interval(verdicts, settings, seed):
    matrix = np.array(verdicts, dtype=float)

    # the bootstrap calls this function with the sampled positions
    # when the sampled data has a single value, krippendorff
    # raises an error, and in that case I returned nan
    def statistic(positions):
        try:
            return krippendorff.alpha(
                reliability_data=matrix[:, positions.astype(int)],
                level_of_measurement="nominal"
            )
        except ValueError:
            return np.nan

    value = statistic(np.arange(matrix.shape[1]))
    result = stats.bootstrap(
        (np.arange(matrix.shape[1]),),
        statistic,
        vectorized=False,
        n_resamples=settings["bootstrap"],
        method="percentile",
        random_state=seed
    )
    interval = result.confidence_interval
    return float(value), float(interval.low), float(interval.high)

# judges_dev stage: choice of the prompt and stability
def run_judges_dev(context):
    settings = context.config["judges"]
    ensure_server()
    ensure_models(settings["models"])
    prompts = load_prompts()
    dev = load_partition(context, "development")

    # judged sample, without the posts used as examples
    examples, used = pick_examples(dev, settings["examples_per_class"], context.seed)
    sample = dev[~dev["id"].isin(used)].sample(n=settings["dev_sample"], random_state=context.seed)
    labels = sample["target"].to_numpy()

    # candidate prompts: (system prompt, examples)
    candidates = {
        "minimal": (prompts["minimal"], []),
        "definition": (prompts["definition"], []),
        "examples": (prompts["definition"], examples),
    }

    selection = []
    stability = []
    chosen = {}
    for model in settings["models"]:
        # picks the prompt with the highest auprc on the sample
        best_label = None
        best_score = None
        for label, (system, shots) in candidates.items():
            table = judge_batch(context, model, label, system, shots, sample, settings["seed"], "selection")

            # the auprc only uses the posts that got a score
            answered = table["score"].notna().to_numpy()
            if answered.any():
                score = auprc(labels[answered], table["score"].to_numpy()[answered])
            else:
                score = float("nan")

            selection.append(
                {
                    "model": model,
                    "prompt": label,
                    "n": len(sample),
                    "coverage": float(answered.mean()),
                    "auprc": score,
                }
            )
            if not np.isnan(score):
                if best_label is None or score > best_score:
                    best_label = label
                    best_score = score

        if best_label is None:
            raise StageFailed(f"{model} did not answer any prompt.")

        system, shots = candidates[best_label]
        chosen[model] = {"prompt": best_label, "system": system, "examples": shots, "auprc_dev": best_score}

        # repetition with other seeds
        # with temperature zero, it only checks if the judge is deterministic
        repetitions = []
        for seed in settings["stability_seeds"]:
            table = judge_batch(context, model, best_label, system, shots, sample, seed, "stability")
            repetitions.append(table["verdict"])

        # reformulation: the chosen prompt and its four
        # rewritten versions, with the same examples
        variants = [system] + prompts[REFORMULATIONS[best_label]]
        reformulations = []
        for index, variant in enumerate(variants):
            table = judge_batch(
                context,
                model,
                f"variant_{index}",
                variant,
                shots,
                sample,
                settings["seed"],
                "stability"
            )
            reformulations.append(table["verdict"])

        # krippendorff's alpha of each kind of stability
        for kind, verdicts in [("repetition", repetitions), ("reformulation", reformulations)]:
            value, low, high = alpha_with_interval(verdicts, context.config["evaluation"], context.seed)
            stability.append(
                {
                    "model": model,
                    "kind": kind,
                    "alpha": value,
                    "ci_low": low,
                    "ci_high": high,
                    "versions": len(verdicts),
                }
            )

            # I wrote the comparison this way so that a nan
            # alpha also raises the alert
            if not value >= settings["min_alpha"]:
                context.alert(f"{model}: {kind} alpha = {value:.2f}, below {settings['min_alpha']}.")

        unload(model)

    context.save_table(pd.DataFrame(selection), "judges_prompt_selection")
    context.save_table(pd.DataFrame(stability), "judges_stability")
    context.save_json({"models": chosen, "sample_ids": sample["id"].tolist()}, "llm", "chosen_prompts.json")
    context.save_json(manifest(context), "llm", "manifest.json")

# judges_test stage: judges and consensus on the test set
def run_judges_test(context):
    settings = context.config["judges"]
    ensure_server()

    # guard: nothing of the judges can have changed since judges_dev
    if manifest(context) != context.read_json("llm", "manifest.json"):
        raise StageFailed("models, prompts or options of the judges changed after the judges_dev stage.")

    chosen = context.read_json("llm", "chosen_prompts.json")["models"]
    question = load_prompts()["explanation"]
    test = load_partition(context, "test")

    tables = []
    explanations = []
    for model in settings["models"]:
        choice = chosen[model]
        table = judge_batch(
            context,
            model,
            choice["prompt"],
            choice["system"],
            choice["examples"],
            test,
            settings["seed"],
            "judgment",
            question
        )

        # warns when the judge did not vote on the whole test
        coverage = table["verdict"].notna().mean()
        if coverage < 1:
            context.alert(f"{model}: answered {coverage:.1%} of the test.")

        # warns when some explanation was cut
        truncated = table["explanation_truncated"].fillna(False).astype(bool).mean()
        if truncated > 0:
            context.alert(f"{model}: {truncated:.1%} of the explanations reached the token limit.")

        explanations.append(table.assign(judge=judge_name(model), target=test["target"].to_numpy()))
        tables.append(
            table[["id", "verdict", "score"]].assign(
                detector=judge_name(model),
                scenario=JUDGE_SCENARIO
            )
        )
        unload(model)

    all_explanations = pd.concat(explanations, ignore_index=True)
    columns = ["id", "judge", "target", "verdict", "score", "explanation", "explanation_truncated"]
    context.save_table(all_explanations[columns], "judges_explanations")

    # consensus: majority of the valid votes and mean of the scores
    # without any valid vote, the consensus stays empty
    vote_rows = []
    score_rows = []
    for table in tables:
        vote_rows.append(table["verdict"].to_numpy(dtype=float))
        score_rows.append(table["score"].to_numpy(dtype=float))
    votes = np.vstack(vote_rows)
    scores = np.vstack(score_rows)

    valid = (~np.isnan(votes)).sum(axis=0)
    positive = np.nansum(votes, axis=0)
    verdict = np.where(valid == 0, np.nan, (positive > valid / 2).astype(float))

    # a post without any score gets a nan mean,
    # which is what I want in that case
    with np.errstate(all="ignore"):
        score = np.nanmean(scores, axis=0)

    consensus = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "verdict": verdict,
            "score": score,
            "detector": CONSENSUS,
            "scenario": JUDGE_SCENARIO,
        }
    )
    predictions = pd.concat(tables + [consensus], ignore_index=True)
    predictions.to_parquet(context.path("predictions", "judges_test.parquet"), index=False)

    # one csv per judge (and for the consensus), in the same
    # format of the other methods (evaluation.py)
    for name, group in predictions.groupby("detector"):
        group[["id", "verdict", "score"]].to_csv(context.path("prediction_tables", f"{name}.csv"), index=False)

STAGE_FUNCTIONS = {"judges_dev": run_judges_dev, "judges_test": run_judges_test}

if __name__ == "__main__":
    run_stage_cli(STAGE_FUNCTIONS)
