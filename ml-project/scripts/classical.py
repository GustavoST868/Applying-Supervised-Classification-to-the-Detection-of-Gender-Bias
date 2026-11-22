# classical classifiers: hyperparameter optimization,
# threshold and final model
#
# usage: python scripts/classical.py classical --run-dir ... --config ...
#
# for each algorithm (complement naive bayes, logistic regression,
# linear svm and xgboost) and each scenario (full and masked text):
# 1. optuna (multivariate tpe sampler) searches, in
#    optimization.trials trials, the tf-idf representation and the
#    hyperparameters of the model together
#    each trial is evaluated by the mean auprc in stratified 5-fold
#    cross-validation on the development set (cross_validate)
#    in xgboost, which has the most expensive fit, the folds run in
#    sequence and the trial is stopped by the median rule
#    (MedianPruner) when the partial mean is below the median
#    of the previous trials at the same fold
#    every trial is saved in the optuna sqlite database in
#    results/runs/<run>/optimization.db, in a study whose
#    name ends with the digest of the stage, so an interrupted
#    run resumes the study, and a change in the code, in the
#    configuration or in the data starts a new study
# 2. with the best configuration, TunedThresholdClassifierCV picks
#    the threshold that maximizes the mean macro f1 over the folds of
#    the cross-validation, in a grid of threshold.grid_points
#    thresholds, and trains the final model on the whole development set
#
# the algorithm is not a variable of the search: the best classical
# model of each scenario is chosen later, by the highest auprc in the
# cross-validation (evaluation.best_classic)

import json
import shutil
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from pipeline import gpu_available, line_chart, measure, run_stage_cli
from common import (
    SCENARIOS,
    TEXT_COLUMN,
    drop_stale_studies,
    load_partition,
    optuna_storage,
    study_name,
    suggest_space,
)

ALGORITHMS = ["naive_bayes", "logistic_regression", "linear_svm", "xgboost"]

# algorithms that run on the gpu: the folds of the
# cross-validation are evaluated one at a time
GPU_ALGORITHMS = {"xgboost"}
NO_CLASS_WEIGHT = "none"

# limit of nodes whose histogram xgboost keeps on the gpu
# without the limit, the cache grows in powers of 2 with the
# depth of the tree and, with character n-grams and a high
# max_depth, it asked for 8 GB on a 12 GB card
# the nodes outside the cache have their histogram recalculated
# instead of obtained by subtracting the sibling, so the model does
# not change (identical predictions on the test with and without the
# limit), only the memory drops (worst case of the search space:
# from 6.4 GB to 1.7 GB)
XGB_MAX_CACHED_HIST_NODE = 32

# hyperparameters that belong to the tf-idf representation
REPRESENTATION_KEYS = {"ngrams", "min_df", "sublinear_tf"}

# creates the tf-idf from the hyperparameters of the representation
# the ngrams value has the format <kind>_<lowest>_<highest>,
# for example word_1_2 or char_1_5
def vectorizer(params):
    kind, low, high = params["ngrams"].split("_")
    if kind == "word":
        analyzer = "word"
    else:
        analyzer = "char"

    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(int(low), int(high)),
        min_df=params["min_df"],
        sublinear_tf=params["sublinear_tf"],
        norm="l2",
        lowercase=True,
        dtype=np.float32,
    )

# creates the model of the algorithm with the sampled hyperparameters
def model(algorithm, params, seed, device):
    params = dict(params)

    # in config.yaml, "none" means no weight on the classes
    if "class_weight" in params and params["class_weight"] == NO_CLASS_WEIGHT:
        params["class_weight"] = None

    if algorithm == "naive_bayes":
        return ComplementNB(**params)
    elif algorithm == "logistic_regression":
        return LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, **params)
    elif algorithm == "linear_svm":
        return LinearSVC(max_iter=10000, random_state=seed, **params)
    elif algorithm == "xgboost":
        # the percentile belongs to the selection step, and not to the model
        params.pop("selection_percentile", None)
        return XGBClassifier(
            device=device,
            tree_method="hist",
            max_cached_hist_node=XGB_MAX_CACHED_HIST_NODE,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

# builds the pipeline tf-idf, feature selection (only in
# xgboost) and model
def build_pipeline(algorithm, params, seed, device, memory=None):
    # separates the hyperparameters of the representation from the ones of the model
    representation = {}
    model_params = {}
    for key, value in params.items():
        if key in REPRESENTATION_KEYS:
            representation[key] = value
        else:
            model_params[key] = value

    steps = [("tfidf", vectorizer(representation))]
    if algorithm == "xgboost":
        selection = SelectPercentile(chi2, percentile=model_params["selection_percentile"])
        steps.append(("selection", selection))
    steps.append(("model", model(algorithm, model_params, seed, device)))

    return Pipeline(steps, memory=memory)

# search space: the one of the representation together with the one of the algorithm
def search_space(context, algorithm):
    space = {}
    space.update(context.config["representation"]["space"])
    space.update(context.config["classical"][algorithm]["space"])
    return space

# stratified folds of the cross-validation
def stratified_folds(context):
    return StratifiedKFold(
        n_splits=context.config["validation"]["folds"],
        shuffle=True,
        random_state=context.seed
    )

# how many folds are evaluated in parallel
def parallel_jobs(context, algorithm):
    if algorithm in GPU_ALGORITHMS:
        return 1
    else:
        return context.config["validation"]["folds"]

# evaluates the folds in sequence and, after each one,
# sends the partial mean to the optuna pruning
def folds_with_pruning(context, trial, estimator, texts, labels, cv):
    scorer = get_scorer(context.config["optimization"]["metric"])
    scores = []
    for step, (train, valid) in enumerate(cv.split(texts, labels)):
        fitted = clone(estimator).fit(texts.iloc[train], labels[train])
        scores.append(scorer(fitted, texts.iloc[valid], labels[valid]))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            values = []
            for value in scores:
                values.append(float(value))
            trial.set_user_attr("auprc_folds", values)
            raise optuna.TrialPruned()

    return np.array(scores)

# evaluates an optuna trial: samples the hyperparameters,
# builds the pipeline and calculates the mean auprc over the folds
def evaluate_trial(context, trial, algorithm, texts, labels, memory, device):
    space = search_space(context, algorithm)
    cv = stratified_folds(context)
    params = suggest_space(trial, space)
    estimator = build_pipeline(algorithm, params, context.seed, device, memory)

    # I kept the convergence warnings to count
    # how many showed up in each trial
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if algorithm in context.config["optimization"]["pruning"]["algorithms"]:
            scores = folds_with_pruning(context, trial, estimator, texts, labels, cv)
        else:
            results = cross_validate(
                estimator,
                texts,
                labels,
                cv=cv,
                scoring=context.config["optimization"]["metric"],
                n_jobs=parallel_jobs(context, algorithm),
                error_score="raise",
            )
            scores = results["test_score"]

    values = []
    for value in scores:
        values.append(float(value))
    trial.set_user_attr("auprc_folds", values)
    trial.set_user_attr("auprc_std", float(np.std(scores)))

    warnings_count = 0
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            warnings_count += 1
    trial.set_user_attr("convergence_warnings", warnings_count)

    return float(np.mean(scores))

# creates (or opens, if it already exists) the optuna study
# of the algorithm in the scenario
def make_study(context, algorithm, scenario):
    settings = context.config["optimization"]
    sampler = optuna.samplers.TPESampler(
        seed=context.seed,
        multivariate=settings["multivariate"],
        n_startup_trials=settings["random_startup_trials"],
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=settings["pruning"]["unpruned_trials"],
        n_warmup_steps=0
    )
    return optuna.create_study(
        study_name=study_name(context, "classic", algorithm, scenario),
        storage=optuna_storage(context),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

# finished trials of the study
def finished_trials(study):
    trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials.append(trial)
    return trials

# trials that count in the budget: finished
# or stopped by the pruning
def counted_trials(study):
    done = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    trials = []
    for trial in study.trials:
        if trial.state in done:
            trials.append(trial)
    return trials

# runs only the trials missing to complete the budget
# this way an interrupted run continues from where it stopped
def optimize(context, algorithm, scenario, texts, labels, memory, device):
    study = make_study(context, algorithm, scenario)
    wanted = context.config["optimization"]["trials"]
    remaining = wanted - len(counted_trials(study))

    if remaining > 0:
        def objective(trial):
            return evaluate_trial(context, trial, algorithm, texts, labels, memory, device)

        with measure(
            context,
            method=method_name(algorithm, scenario),
            scenario=scenario,
            phase="optimization",
            n_documents=len(texts)
        ):
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    return study

# name of the method in the cost and in the test predictions
def method_name(algorithm, scenario):
    return f"classic_{algorithm}_{scenario}"

# picks the threshold through the cross-validation and trains
# the final model on the whole development set
def fit_final(context, algorithm, scenario, params, texts, labels, device):
    estimator = build_pipeline(algorithm, params, context.seed, device)
    tuned = TunedThresholdClassifierCV(
        estimator,
        scoring=context.config["threshold"]["metric"],
        thresholds=context.config["threshold"]["grid_points"],
        cv=stratified_folds(context),
        refit=True,
        random_state=context.seed,
        n_jobs=parallel_jobs(context, algorithm),
    )

    with measure(
        context,
        method=method_name(algorithm, scenario),
        scenario=scenario,
        phase="final_training",
        n_documents=len(texts)
    ):
        tuned.fit(texts, labels)

    return tuned

# chart of the best auprc up to each trial,
# one chart per scenario and one line per algorithm
def plot_history(context, studies):
    for scenario in SCENARIOS:
        series = []
        for algorithm in ALGORITHMS:
            study = studies[(algorithm, scenario)]
            values = []
            for trial in finished_trials(study):
                values.append(trial.value)
            best_so_far = np.maximum.accumulate(values)
            series.append((algorithm, np.arange(1, len(values) + 1), best_so_far))

        line_chart(
            context,
            f"optimization_{scenario}",
            series,
            f"Best cross-validation AUPRC, {scenario} scenario",
            "Trial",
            "AUPRC",
            markers=False,
        )

# classical stage
def run_classical(context):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    drop_stale_studies(context, "classic_")
    dev = load_partition(context, "development")
    labels = dev["target"].to_numpy()

    if gpu_available():
        device = "cuda"
    else:
        device = "cpu"

    # cache of the tf-idf between the trials
    # I used a dummy "x" file only so that context.path
    # creates the folder, and kept its folder
    cache = context.path("cache", "tfidf", "x").parent
    memory = joblib.Memory(location=str(cache), verbose=0)

    studies = {}
    rows = []
    trial_tables = []
    for scenario in SCENARIOS:
        texts = dev[TEXT_COLUMN[scenario]]
        for algorithm in ALGORITHMS:
            print(f"[optimization] {algorithm}, {scenario} scenario", flush=True)
            study = optimize(context, algorithm, scenario, texts, labels, memory, device)
            studies[(algorithm, scenario)] = study

            # final model with the best trial
            best = study.best_trial
            final = fit_final(context, algorithm, scenario, best.params, texts, labels, device)
            joblib.dump(final, context.path("models", f"{method_name(algorithm, scenario)}.joblib"))

            counted = len(counted_trials(study))
            finished = len(finished_trials(study))
            rows.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "trials": counted,
                    "pruned_trials": counted - finished,
                    "best_trial": best.number,
                    "auprc_cv": best.value,
                    "auprc_cv_std": best.user_attrs.get("auprc_std"),
                    "threshold": float(final.best_threshold_),
                    "f1_macro_cv": float(final.best_score_),
                    "params": json.dumps(best.params),
                }
            )

            # every trial of the study
            table = study.trials_dataframe()
            table.insert(0, "scenario", scenario)
            table.insert(0, "algorithm", algorithm)
            trial_tables.append(table)

            # I saved the tables again after each finished algorithm so that
            # a later failure does not erase the results that already finished
            context.save_table(pd.DataFrame(rows), "classical_best")
            context.save_table(pd.concat(trial_tables, ignore_index=True), "classical_trials")

    plot_history(context, studies)
    shutil.rmtree(cache, ignore_errors=True)

STAGE_FUNCTIONS = {"classical": run_classical}

if __name__ == "__main__":
    run_stage_cli(STAGE_FUNCTIONS)
