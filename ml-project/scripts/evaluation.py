# evaluation: test set, comparisons, interpretation,
# cost, report and traceability
#
# usage: python scripts/evaluation.py <stage> --run-dir ... --config ...
#
# stages:
# - test: applies the frozen models to the test set
#   (classical classifiers, transformer, lexicon rule and
#   trivial classifier) and saves the predictions
# - comparison: metrics of every method (including the judges),
#   bootstrap confidence intervals (scipy.stats.bootstrap),
#   mcnemar tests with holm correction (statsmodels),
#   agreement between the judges and auc per subgroup (with and
#   without a lexicon term)
# - interpretation: features with the largest weight of the best
#   classical classifiers (shap values), with the fraction of the
#   occurrences of each feature that falls inside a lexicon
#   term, local explanations with lime of the transformer and of the
#   best classical model and, for the judges, the explanations of the
#   three side by side in cases of each type of the consensus and in
#   cases of disagreement between them
# - cost: preparation time, time per prediction and vram of each
#   method, including the consensus of the judges (sum of the three),
#   only with the records of the current version of each stage
# - report: SUMMARY.md by research question and
#   traceability.csv, which links each value of config.yaml to the
#   references that support it and checks if they are in
#   referencias.bib and are cited in the article

import hashlib
import itertools
import json
import re
from pathlib import Path

import joblib
import krippendorff
import numpy as np
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
from scipy import stats
from scipy.special import expit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from pipeline import (
    PHASES,
    ROOT,
    StageFailed,
    confusion_matrix_figure,
    heatmap,
    horizontal_bars,
    measure,
    run_stage_cli,
)
from common import (
    SCENARIOS,
    TEXT_COLUMN,
    auprc,
    classification_metrics,
    load_lexicon,
    load_partition,
)
from judges import CONSENSUS, judge_name

CLASSIFIERS_FILE = ("predictions", "classifiers_test.parquet")
JUDGES_FILE = ("predictions", "judges_test.parquet")
LOCK_FILE = "test_opened.json"
LEXICON_RULE = "baseline_lexicon"
TRIVIAL = "baseline_trivial"

# the shap values are calculated in blocks of this size
SHAP_CHUNK = 50

# how TfidfVectorizer extracts the features: words through the
# default token_pattern and, in the character n-grams, the text
# with the sequences of spaces replaced by a single space
WORD_TOKEN = re.compile(r"(?u)\b\w\w+\b")
SPACES = re.compile(r"\s\s+")

# phases and stages used in the cost calculation
PREDICTION_PHASES = ["prediction", "judgment"]
TEST_STAGES = ["test", "judges_test"]
TIME_COLUMNS = ["preparation_time_s", "test_time_s", "ms_per_prediction", "explanation_time_s"]
VRAM_COLUMNS = ["mean_vram_mb", "mean_device_vram_mb"]

# sha-256 digest of a file or of every
# file of a folder
def file_hash(path):
    path = Path(path)
    if path.is_dir():
        items = sorted(path.rglob("*"))
    else:
        items = [path]

    digest = hashlib.sha256()
    for item in items:
        if item.is_file():
            digest.update(item.read_bytes())
    return digest.hexdigest()

# records the models used when the test was opened
# and refuses different models later
def lock_test(context, fingerprint):
    path = context.path(LOCK_FILE)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf8"))
        if previous != fingerprint:
            raise StageFailed("the models changed after the test was opened. Use another --run-name.")

    context.save_json(fingerprint, LOCK_FILE)

# score of a classical model: the probability when the model
# estimates it, or the distance to the hyperplane in the linear svm
def classic_scores(estimator, texts):
    inner = estimator.estimator_
    if hasattr(inner, "predict_proba"):
        return inner.predict_proba(texts)[:, 1]
    else:
        return inner.decision_function(texts)

# table of predictions of a method on the test
def prediction_table(test, name, scenario, scores, verdict):
    return pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "detector": name,
            "scenario": scenario,
            "score": np.asarray(scores, dtype=float),
            "verdict": np.asarray(verdict, dtype=float),
        }
    )

# test stage: applies the frozen models to the test set
def run_test(context):
    # I imported it here inside so that the other stages of this
    # script do not need to load pytorch
    from transformer import predict_scores

    # models saved by the classical and transformer stages
    models = context.path("models", "x").parent
    classics = sorted(models.glob("classic_*.joblib"))
    transformers = []
    for path in sorted(models.glob("transformer_*")):
        if path.is_dir():
            transformers.append(path)

    if not classics or not transformers:
        raise StageFailed("missing models: the classical and transformer stages must finish first.")

    # locks the models at the moment the test is opened
    fingerprint = {}
    for path in classics + transformers:
        fingerprint[path.name] = file_hash(path)
    lock_test(context, fingerprint)

    dev = load_partition(context, "development")
    test = load_partition(context, "test")
    rows = []

    # classical classifiers
    # the file name ends with the scenario
    for path in classics:
        estimator = joblib.load(path)
        name = path.stem
        scenario = name.rsplit("_", 1)[1]
        texts = test[TEXT_COLUMN[scenario]]
        with measure(context, method=name, scenario=scenario, phase="prediction", n_documents=len(test)):
            scores = classic_scores(estimator, texts)
            verdict = estimator.predict(texts)
        rows.append(prediction_table(test, name, scenario, scores, verdict))

    # transformer, with the threshold saved with the model
    for folder in transformers:
        scenario = folder.name.rsplit("_", 1)[1]
        texts = test[TEXT_COLUMN[scenario]]
        with measure(context, method=folder.name, scenario=scenario, phase="prediction", n_documents=len(test)):
            scores, record = predict_scores(folder, texts)
        rows.append(prediction_table(test, folder.name, scenario, scores, scores >= record["threshold"]))

    # lexicon rule
    # the rule counts the terms on the test again so
    # that the prediction time is measured
    lexicon = load_lexicon(context.path("interim", "lexicon.txt"))
    rule = context.config["lexicon"]["rule_threshold"]
    with measure(context, method=LEXICON_RULE, scenario="baseline", phase="prediction", n_documents=len(test)):
        counts = test["text"].map(lexicon.count).to_numpy()
    if not np.array_equal(counts, test["lexicon_terms"].to_numpy()):
        raise StageFailed("the lexicon count on the test differs from the one made in the preprocessing")
    rows.append(prediction_table(test, LEXICON_RULE, "baseline", counts, counts >= rule))

    # trivial classifier, which always gives the most frequent class
    with measure(context, method=TRIVIAL, scenario="baseline", phase="training", n_documents=len(dev)):
        trivial = DummyClassifier(strategy="prior").fit(dev[["id"]], dev["target"])
    with measure(context, method=TRIVIAL, scenario="baseline", phase="prediction", n_documents=len(test)):
        scores = trivial.predict_proba(test[["id"]])[:, 1]
        verdict = trivial.predict(test[["id"]])
    rows.append(prediction_table(test, TRIVIAL, "baseline", scores, verdict))

    # saves the predictions together and one csv per method
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(context.path(*CLASSIFIERS_FILE), index=False)
    for name, group in predictions.groupby("detector"):
        group[["id", "verdict", "score"]].to_csv(context.path("prediction_tables", f"{name}.csv"), index=False)

# predictions of the classifiers and, if they exist, of the judges
def load_predictions(context):
    tables = [pd.read_parquet(context.path(*CLASSIFIERS_FILE))]
    if context.path(*JUDGES_FILE).exists():
        tables.append(pd.read_parquet(context.path(*JUDGES_FILE)))
    else:
        context.alert("there are no judge predictions; the comparison only covers the other methods.")
    return pd.concat(tables, ignore_index=True)

# paired bootstrap confidence interval
def interval(data, statistic, context):
    settings = context.config["evaluation"]
    result = stats.bootstrap(
        data,
        statistic,
        paired=True,
        vectorized=False,
        n_resamples=settings["bootstrap"],
        method="percentile",
        confidence_level=1 - settings["alpha"],
        random_state=context.seed
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)

# macro f1, used in the bootstrap
def macro_f1(labels, predicted):
    return f1_score(labels, predicted, average="macro", zero_division=0)

# metrics of each method on the test, with confidence intervals
def metrics_table(context, predictions, truth):
    rows = []
    for (name, scenario), group in predictions.groupby(["detector", "scenario"]):
        merged = group.merge(truth, on="id")

        # metrics of the vote, only on the posts with a vote
        answered = merged.dropna(subset=["verdict"])
        labels = answered["target"].to_numpy()
        verdict = answered["verdict"].to_numpy(dtype=int)
        row = {"detector": name, "scenario": scenario, "coverage": len(answered) / len(merged)}
        row.update(classification_metrics(labels, verdict))
        low, high = interval((labels, verdict), macro_f1, context)
        row["f1_macro_ci_low"] = low
        row["f1_macro_ci_high"] = high

        # metrics of the score, only on the posts with a score
        scored = merged.dropna(subset=["score"])
        score_labels = scored["target"].to_numpy()
        scores = scored["score"].to_numpy(dtype=float)
        row["score_coverage"] = len(scored) / len(merged)
        if len(scored) and len(np.unique(score_labels)) == 2:
            row["auprc"] = auprc(score_labels, scores)
            row["auc_roc"] = roc_auc_score(score_labels, scores)
            low, high = interval((score_labels, scores), auprc, context)
            row["auprc_ci_low"] = low
            row["auprc_ci_high"] = high

        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

# figure of the confusion matrix of each method
def save_confusion_figures(context, predictions, truth):
    for name, group in predictions.groupby("detector"):
        merged = group.merge(truth, on="id").dropna(subset=["verdict"])

        # counts[true][classified], with the order [sexist, not sexist]
        counts = []
        for real in (1, 0):
            line = []
            for decided in (1, 0):
                hits = (merged["target"] == real) & (merged["verdict"] == decided)
                line.append(int(hits.sum()))
            counts.append(line)

        path = context.path("figures", "confusion_matrices", f"{name}.png")
        confusion_matrix_figure(path, counts, name)

# name of the best classical model of the scenario, the one with the
# highest auprc in the cross-validation on the development set
def best_classic(context, scenario):
    table = context.read_table("classical_best")
    table = table[table["scenario"] == scenario].sort_values("auprc_cv", ascending=False)
    return f"classic_{table['algorithm'].iloc[0]}_{scenario}"

# pairs of methods compared by mcnemar, defined
# before opening the test
def planned_pairs(context, names):
    pairs = []

    # transformer, best classical model and lexicon rule in each scenario
    for scenario in SCENARIOS:
        transformer = f"transformer_{scenario}"
        classic = best_classic(context, scenario)
        pairs.append((transformer, classic))
        pairs.append((transformer, LEXICON_RULE))
        pairs.append((classic, LEXICON_RULE))

    # each supervised method with the full text against
    # itself with the masked text
    supervised = set()
    for name in names:
        if name.startswith(("classic_", "transformer_")):
            supervised.add(name.rsplit("_", 1)[0])
    for name in sorted(supervised):
        pairs.append((f"{name}_full", f"{name}_masked"))

    # consensus of the judges against the other methods
    pairs.append((CONSENSUS, "transformer_full"))
    pairs.append((CONSENSUS, best_classic(context, "full")))
    pairs.append((CONSENSUS, LEXICON_RULE))

    # keeps only the pairs whose two methods exist
    existing = []
    for first, second in pairs:
        if first in names and second in names:
            existing.append((first, second))
    return existing

# mcnemar tests of the planned pairs, with holm correction
def mcnemar_tests(context, predictions, truth):
    verdicts = predictions.pivot_table(index="id", columns="detector", values="verdict")
    labels = truth.set_index("id")["target"]

    rows = []
    for first, second in planned_pairs(context, set(verdicts.columns)):
        data = verdicts[[first, second]].dropna().join(labels, how="inner")
        right_first = data[first] == data["target"]
        right_second = data[second] == data["target"]

        # 2x2 table of hits and errors of the two methods
        table = [
            [int((right_first & right_second).sum()), int((right_first & ~right_second).sum())],
            [int((~right_first & right_second).sum()), int((~right_first & ~right_second).sum())],
        ]
        result = mcnemar(table, exact=False, correction=True)
        rows.append(
            {
                "first": first,
                "second": second,
                "n": len(data),
                "only_first_correct": table[0][1],
                "only_second_correct": table[1][0],
                "statistic": float(result.statistic),
                "p": float(result.pvalue),
            }
        )

    table = pd.DataFrame(rows)
    if len(table):
        reject, corrected, _, _ = multipletests(
            table["p"],
            alpha=context.config["evaluation"]["alpha"],
            method="holm"
        )
        table["p_holm"] = corrected
        table["significant"] = reject

    return table

# agreement between the judges: cohen's kappa of each pair
# and krippendorff's alpha of the whole panel
def judge_agreement(context, predictions):
    names = []
    for model in context.config["judges"]["models"]:
        names.append(judge_name(model))

    judges = predictions[predictions["detector"].isin(names)]
    votes = judges.pivot(index="id", columns="detector", values="verdict")

    # without at least two judges there is nothing to compare
    present = []
    for name in names:
        if name in votes.columns:
            present.append(name)
    names = present
    if len(names) < 2:
        return

    # kappa of each pair of judges
    matrix = np.eye(len(names))
    rows = []
    for (i, first), (j, second) in itertools.combinations(enumerate(names), 2):
        pair = votes[[first, second]].dropna()
        kappa = cohen_kappa_score(pair[first], pair[second])
        matrix[i, j] = kappa
        matrix[j, i] = kappa
        rows.append({"first": first, "second": second, "n": len(pair), "kappa": kappa})

    # alpha of the panel, with one row per judge
    data = votes[names].T.to_numpy(dtype=float)

    # the bootstrap calls this function with the sampled positions
    def alpha(positions):
        return krippendorff.alpha(
            reliability_data=data[:, positions.astype(int)],
            level_of_measurement="nominal"
        )

    low, high = interval((np.arange(data.shape[1]),), alpha, context)
    rows.append(
        {
            "first": "panel",
            "second": "panel",
            "n": data.shape[1],
            "krippendorff_alpha": alpha(np.arange(data.shape[1])),
            "ci_low": low,
            "ci_high": high,
        }
    )
    context.save_table(pd.DataFrame(rows), "judges_agreement")
    heatmap(context, "judges_agreement", matrix, names, "Cohen's kappa between the judges")

# roc auc inside the subgroups with and without a
# lexicon term (subgroup auc)
def subgroup_auc(predictions, test):
    frame = test[["id", "target", "lexicon_terms"]]
    rows = []
    for (name, scenario), group in predictions.groupby(["detector", "scenario"]):
        merged = group.merge(frame, on="id").dropna(subset=["score"])
        subgroups = [
            ("with_term", merged["lexicon_terms"] > 0),
            ("without_term", merged["lexicon_terms"] == 0),
        ]
        for label, mask in subgroups:
            part = merged[mask]

            # the auc only exists with both classes in the subgroup
            if part["target"].nunique() == 2:
                value = roc_auc_score(part["target"], part["score"])
            else:
                value = np.nan

            rows.append(
                {
                    "detector": name,
                    "scenario": scenario,
                    "subgroup": label,
                    "n": len(part),
                    "prevalence": part["target"].mean(),
                    "subgroup_auc": value,
                }
            )

    return pd.DataFrame(rows)

# comparison stage
def run_comparison(context):
    test = load_partition(context, "test")
    truth = test[["id", "target"]]
    predictions = load_predictions(context)

    # metrics and chart of the macro f1 with the confidence interval
    table = metrics_table(context, predictions, truth)
    context.save_table(table, "test_results")
    ordered = table.sort_values("f1_macro", ascending=False)
    lower_error = (ordered["f1_macro"] - ordered["f1_macro_ci_low"]).clip(lower=0)
    upper_error = (ordered["f1_macro_ci_high"] - ordered["f1_macro"]).clip(lower=0)
    horizontal_bars(
        context,
        "test_results_f1_macro",
        list(ordered["detector"]),
        ordered["f1_macro"],
        "F1-macro on the test, with 95% CI",
        "F1-macro",
        errors=[lower_error, upper_error]
    )

    save_confusion_figures(context, predictions, truth)
    context.save_table(mcnemar_tests(context, predictions, truth), "comparison_mcnemar")
    context.save_table(subgroup_auc(predictions, test), "subgroup_auc")
    judge_agreement(context, predictions)

    # table with the vote of each method on each test post
    wide = test[["id", "text", "target"]].copy()
    for name, group in predictions.groupby("detector"):
        wide[name] = wide["id"].map(group.set_index("id")["verdict"]).astype("Int64")
    wide.to_csv(context.path("tables", "test_classifications.csv"), index=False)

# linear shap explainer (lundberg and lee, 2017) or
# tree explainer (lundberg et al., 2020)
# in the linear shap, E[x] is the mean of the whole sample,
# and not of a subsample
def shap_explainer(model, background):
    if hasattr(model, "coef_") or hasattr(model, "feature_log_prob_"):
        background = shap.maskers.Independent(background, max_samples=background.shape[0])

    if hasattr(model, "feature_log_prob_"):
        # the complement naive bayes decides linearly on the
        # features (rennie et al., 2003), so the weight of each
        # feature is the one of the sexist class minus the one of the not sexist class
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        return shap.LinearExplainer((weights, 0.0), background)
    elif hasattr(model, "coef_"):
        return shap.LinearExplainer(model, background)
    else:
        return shap.TreeExplainer(model)

# spans (start, end) of each feature in the text, extracted
# the same way TfidfVectorizer extracts them
# also returns the text in the form in which the spans were
# measured (lowercase and, in the character n-grams, with the
# spaces normalized), to look for the lexicon terms in it
def attribute_occurrences(text, attributes, vectorizer):
    found = {}

    # character n-grams: looks for each feature in the text,
    # including the overlapping occurrences
    if vectorizer.analyzer == "char":
        text = SPACES.sub(" ", text.lower())
        for attribute in attributes:
            start = text.find(attribute)
            while start >= 0:
                if attribute not in found:
                    found[attribute] = []
                found[attribute].append((start, start + len(attribute)))
                start = text.find(attribute, start + 1)
        return text, found

    # word n-grams: builds each n-gram of the text
    # and checks if it is one of the features
    text = text.lower()
    tokens = list(WORD_TOKEN.finditer(text))
    low, high = vectorizer.ngram_range
    for size in range(low, high + 1):
        for position in range(len(tokens) - size + 1):
            words = []
            for token in tokens[position:position + size]:
                words.append(token.group())
            gram = " ".join(words)

            if gram in attributes:
                span = (tokens[position].start(), tokens[position + size - 1].end())
                if gram not in found:
                    found[gram] = []
                found[gram].append(span)

    return text, found

# fraction of the occurrences of each feature, in the texts,
# that falls inside a lexicon term
# a character n-gram is rarely a whole term (the n-grams
# have up to five characters, and most terms have more),
# so I compared the feature with the spans of the text
# taken by lexicon terms, and not with the list of terms
def lexicon_share(texts, attributes, vectorizer, lexicon):
    targets = set(attributes)
    inside = dict.fromkeys(attributes, 0)
    total = dict.fromkeys(attributes, 0)

    for text in texts:
        normalized, found = attribute_occurrences(text, targets, vectorizer)
        if not found:
            continue

        spans = lexicon.spans(normalized)
        for attribute, places in found.items():
            for start, end in places:
                total[attribute] += 1

                # the occurrence counts when it overlaps some term
                overlaps = False
                for span_start, span_end in spans:
                    if start < span_end and span_start < end:
                        overlaps = True
                        break
                if overlaps:
                    inside[attribute] += 1

    shares = {}
    for name in attributes:
        if total[name]:
            shares[name] = inside[name] / total[name]
        else:
            shares[name] = float("nan")
    return shares

# global importance and direction of each feature through the shap values
# the importance is the mean of the absolute shap value in the sample
# (lundberg et al., 2020), and the direction is the mean of the shap in
# the posts that contain the feature, positive when the feature
# pushes the decision towards the sexist class
# I calculated the values in blocks because the matrix of values is dense
# and the vocabulary can have hundreds of thousands of n-grams
def feature_weights(estimator, dev_texts, seed, sample_size):
    pipeline = estimator.estimator_
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    matrix = pipeline.named_steps["tfidf"].transform(dev_texts)

    # in xgboost, only the features that passed the selection
    if "selection" in pipeline.named_steps:
        names = names[pipeline.named_steps["selection"].get_support()]
        matrix = pipeline.named_steps["selection"].transform(matrix)

    # sample of the posts of the development set
    rng = np.random.default_rng(seed)
    size = min(sample_size, matrix.shape[0])
    sample = matrix[rng.choice(matrix.shape[0], size=size, replace=False)]
    explainer = shap_explainer(pipeline.named_steps["model"], sample)

    magnitude = np.zeros(len(names))
    signed = np.zeros(len(names))
    present = np.zeros(len(names))
    for start in range(0, sample.shape[0], SHAP_CHUNK):
        part = sample[start:start + SHAP_CHUNK]
        values = explainer.shap_values(part)
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values)

        # some explainers return the values of both
        # classes, and in that case I keep the sexist class
        if values.ndim == 3:
            values = values[..., 1]

        mask = (part != 0).toarray()
        magnitude += np.abs(values).sum(axis=0)
        signed += (values * mask).sum(axis=0)
        present += mask.sum(axis=0)

    # features that do not show up in the sample get a nan direction
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = signed / present

    return names, magnitude / sample.shape[0], direction

# the most important features of the best classical model of the scenario
# and the link of each one with the lexicon
# a feature is a lexicon term when most of its occurrences in the
# development set are inside a lexicon term (fraction above
# interpretation.lexicon_fraction_threshold)
def influential_terms(context, scenario, lexicon, dev):
    settings = context.config["interpretation"]
    name = best_classic(context, scenario)
    estimator = joblib.load(context.path("models", f"{name}.joblib"))
    texts = dev[TEXT_COLUMN[scenario]]
    names, importance, direction = feature_weights(estimator, texts, context.seed, settings["shap_sample"])

    # the features with the highest importance
    order = np.argsort(-importance)[:settings["terms"]]
    attributes = []
    for position in order:
        attributes.append(names[position])

    vectorizer = estimator.estimator_.named_steps["tfidf"]
    share = lexicon_share(texts, attributes, vectorizer, lexicon)

    rows = []
    for position, term in zip(order, attributes):
        if direction[position] > 0:
            side = "sexist"
        else:
            side = "not_sexist"

        rows.append(
            {
                "model": name,
                "scenario": scenario,
                "feature": term,
                "shap_importance": float(importance[position]),
                "direction": side,
                "lexicon_fraction": share[term],
                "lexicon_term": bool(share[term] > settings["lexicon_fraction_threshold"]),
            }
        )

    return pd.DataFrame(rows)

# function lime calls to get the probability of the
# two classes from a classical model
# the linear svm does not estimate probability, so its score
# (the distance to the hyperplane) goes through the logistic function
def classic_probability(estimator):
    inner = estimator.estimator_
    if hasattr(inner, "predict_proba"):
        return inner.predict_proba

    def predict(texts):
        positive = expit(inner.decision_function(list(texts)))
        return np.column_stack([1 - positive, positive])

    return predict

# function lime calls to get the probability of the
# two classes from the transformer saved in folder
# I imported it here inside for the same reason as in the test stage
def transformer_probability(folder):
    from transformer import predict_scores

    def predict(texts):
        scores, _ = predict_scores(folder, list(texts))
        return np.column_stack([1 - scores, scores])

    return predict

# verdicts of a method in the same order as the test posts
def verdicts_in_test_order(predictions, name, test):
    chosen = predictions[predictions["detector"] == name]
    verdicts = chosen.set_index("id").loc[test["id"], "verdict"]
    return verdicts.to_numpy()

# local lime explanations on sampled cases
# of each type (hits and errors)
def lime_cases(context, name, scenario, predict, test, verdict):
    settings = context.config["interpretation"]
    labels = test["target"].to_numpy()
    cases = {
        "true_positive": (labels == 1) & (verdict == 1),
        "false_positive": (labels == 0) & (verdict == 1),
        "false_negative": (labels == 1) & (verdict == 0),
    }
    explainer = LimeTextExplainer(class_names=["not_sexist", "sexist"], random_state=context.seed)
    rng = np.random.default_rng(context.seed)

    rows = []
    for case, mask in cases.items():
        positions = np.flatnonzero(mask)
        size = min(settings["cases_per_type"], len(positions))
        for position in rng.choice(positions, size=size, replace=False):
            text = test[TEXT_COLUMN[scenario]].iloc[position]
            explanation = explainer.explain_instance(
                text,
                predict,
                num_features=settings["lime_words"],
                num_samples=settings["lime_samples"]
            )
            for term, weight in explanation.as_list():
                rows.append(
                    {
                        "model": name,
                        "case": case,
                        "id": int(test["id"].iloc[position]),
                        "text": text,
                        "term": term,
                        "weight": weight,
                    }
                )

    return rows

# interpretation stage
def run_interpretation(context):
    lexicon = load_lexicon(context.path("interim", "lexicon.txt"))
    dev = load_partition(context, "development")
    test = load_partition(context, "test")
    predictions = pd.read_parquet(context.path(*CLASSIFIERS_FILE))

    # most important features of the best classical models
    tables = []
    for scenario in SCENARIOS:
        tables.append(influential_terms(context, scenario, lexicon, dev))
    context.save_table(pd.concat(tables), "influential_terms")

    # lime of the transformer and of the best classical model of each scenario
    rows = []
    for scenario in SCENARIOS:
        folder = context.path("models", f"transformer_{scenario}", "x").parent
        name = folder.name
        verdict = verdicts_in_test_order(predictions, name, test)
        rows += lime_cases(context, name, scenario, transformer_probability(folder), test, verdict)

        classic_name = best_classic(context, scenario)
        estimator = joblib.load(context.path("models", f"{classic_name}.joblib"))
        verdict = verdicts_in_test_order(predictions, classic_name, test)
        rows += lime_cases(context, classic_name, scenario, classic_probability(estimator), test, verdict)

    context.save_table(pd.DataFrame(rows), "lime_explanations")
    context.save_table(judge_explanation_cases(context, test), "judges_explanation_cases")

# explanations of the judges side by side on sampled
# cases of each type
# the types are the ones of lime, by the vote of the consensus, plus the
# cases in which the judges did not all vote the same
# showing explanations of opposite votes side by side reduces the risk
# of a persuasive explanation steering the reading (wang et al., 2023)
def judge_explanation_cases(context, test):
    explanations = context.read_table("judges_explanations")
    judges = pd.read_parquet(context.path(*JUDGES_FILE))
    consensus = verdicts_in_test_order(judges, CONSENSUS, test)

    # cases in which the judges disagreed
    votes = explanations.pivot(index="id", columns="judge", values="verdict").loc[test["id"]]
    split = (votes.nunique(axis=1) > 1).to_numpy()

    labels = test["target"].to_numpy()
    cases = {
        "true_positive": (labels == 1) & (consensus == 1) & ~split,
        "false_positive": (labels == 0) & (consensus == 1),
        "false_negative": (labels == 1) & (consensus == 0),
        "disagreement": split,
    }
    rng = np.random.default_rng(context.seed)
    size = context.config["interpretation"]["cases_per_type"]

    rows = []
    for case, mask in cases.items():
        positions = np.flatnonzero(mask)
        for position in rng.choice(positions, size=min(size, len(positions)), replace=False):
            doc = int(test["id"].iloc[position])

            # one row per judge
            for item in explanations[explanations["id"] == doc].itertuples():
                rows.append(
                    {
                        "case": case,
                        "id": doc,
                        "text": test["text"].iloc[position],
                        "target": int(labels[position]),
                        "consensus_verdict": consensus[position],
                        "judge": item.judge,
                        "verdict": item.verdict,
                        "score": item.score,
                        "explanation": item.explanation,
                    }
                )

    return pd.DataFrame(rows)

# reads the status of the stages
def read_status(context):
    status_file = context.run_dir / "status.json"
    if status_file.exists():
        return json.loads(status_file.read_text())
    else:
        return {}

# cost records of the current version of each finished stage
# records of previous versions of a stage (other code,
# configuration or data) stay out, and the ones of a run
# interrupted and resumed in the same version add up
def current_cost(context, status):
    cost = pd.read_csv(context.path("cost", "cost_runs.csv"))

    # current digest of each finished stage
    current = {}
    for stage, item in status.items():
        if item.get("state") == "ok":
            current[stage] = item.get("hash")

    keep = []
    for stage, digest in zip(cost["stage"], cost["stage_hash"]):
        keep.append(current.get(stage) == digest)

    cost = cost[keep].copy()
    cost["n_documents"] = pd.to_numeric(cost["n_documents"], errors="coerce")
    return cost[cost["method"].notna()]

# mean of a column weighted by the time of each unit
def weighted_mean(units, column):
    measured = units.dropna(subset=[column])
    measured = measured[measured["seconds"] > 0]
    if measured.empty:
        return np.nan
    else:
        return float(np.average(measured[column], weights=measured["seconds"]))

# cost of a method: preparation (search, training, choice of the prompt),
# prediction on the test and explanations
def cost_row(method, scenario, units):
    prediction = units[units["phase"].isin(PREDICTION_PHASES) & units["stage"].isin(TEST_STAGES)]
    preparation = units[~units["phase"].isin(PREDICTION_PHASES + ["explanation"])]
    explaining = units[units["phase"] == "explanation"]

    documents = prediction["n_documents"].sum()
    if documents:
        ms_per_prediction = 1000 * prediction["seconds"].sum() / documents
    else:
        ms_per_prediction = np.nan

    row = {
        "method": method,
        "scenario": scenario,
        "preparation_time_s": preparation["seconds"].sum(),
        "test_time_s": prediction["seconds"].sum(),
        "ms_per_prediction": ms_per_prediction,
        "explanation_time_s": explaining["seconds"].sum(),
    }
    for column in VRAM_COLUMNS:
        row[column] = weighted_mean(units, column)
    return row

# cost of the consensus, which needs the three judges
# the times are the sum of theirs, and the vram is the mean weighted
# by the time, since the judges run one after the other
def consensus_cost(context, cost, rows):
    names = []
    for model in context.config["judges"]["models"]:
        names.append(judge_name(model))

    judges = []
    for row in rows:
        if row["method"] in names:
            judges.append(row)

    if len(judges) < len(names):
        context.alert("cost records of some judge are missing; the consensus has no cost.")
        return None

    units = cost[cost["method"].isin(names)]
    row = {"method": CONSENSUS, "scenario": judges[0]["scenario"]}
    for column in TIME_COLUMNS:
        total = 0
        for item in judges:
            total += item[column]
        row[column] = total
    for column in VRAM_COLUMNS:
        row[column] = weighted_mean(units, column)
    return row

# cost stage
def run_cost(context):
    status = read_status(context)
    cost = current_cost(context, status)

    # cost of each method and of the consensus
    rows = []
    for (method, scenario), units in cost.groupby(["method", "scenario"], dropna=False):
        rows.append(cost_row(method, scenario, units))

    consensus = consensus_cost(context, cost, rows)
    if consensus is not None:
        rows.append(consensus)
    context.save_table(pd.DataFrame(rows), "cost_by_method")

    # time of each stage and of the whole run
    stage_rows = []
    for _, title, stages in PHASES:
        for stage in stages:
            seconds = status.get(stage, {}).get("seconds")
            stage_rows.append({"phase": title, "stage": stage, "seconds": seconds})

    times = pd.DataFrame(stage_rows)
    total = pd.DataFrame([{"phase": "Total", "stage": "(whole run)", "seconds": times["seconds"].sum()}])
    context.save_table(pd.concat([times, total], ignore_index=True), "execution_time")

# turns the blocks of the config into a list (name.with.dots, value)
# a block with "type" is a search space and stays whole
def flatten(block, prefix=""):
    items = []
    for key, value in block.items():
        if key == "references":
            continue

        if prefix:
            name = f"{prefix}.{key}"
        else:
            name = key

        if isinstance(value, dict) and "type" not in value:
            items.extend(flatten(value, name))
        else:
            items.append((name, value))

    return items

# references of the innermost block that defines
# references on the path of the parameter
def nearest_references(config, path):
    node = config
    found = []
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            break
        if "references" in node:
            found = node["references"]

        node = node[part]
        if isinstance(node, dict) and "references" in node:
            found = node["references"]

    return found

# links each value of config.yaml to the references that support it
# and checks if they are in referencias.bib and are cited in the article
def traceability(context):
    config = context.config

    # loose values at the top of the config, without references
    rows = []
    for key, value in config.items():
        if not isinstance(value, dict):
            rows.append({"parameter": key, "value": json.dumps(value), "references": ""})

    # values inside the blocks, with the nearest references
    blocks = {}
    for key, value in config.items():
        if isinstance(value, dict):
            blocks[key] = value
    for name, value in flatten(blocks):
        rows.append(
            {
                "parameter": name,
                "value": json.dumps(value, ensure_ascii=False),
                "references": ", ".join(nearest_references(config, name)),
            }
        )
    table = pd.DataFrame(rows)

    # every reference key used in the config
    used = set()
    for item in table["references"]:
        for key in item.split(","):
            if key.strip():
                used.add(key.strip())

    # references missing from referencias.bib
    bib = ROOT.parent / "Artigo" / "referencias.bib"
    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf8")))
        missing = sorted(used - keys)
        if missing:
            context.alert(f"config references missing from referencias.bib: {missing}")

    # references the article does not cite
    article = ROOT.parent / "Artigo" / "artigo.tex"
    if article.exists():
        groups = re.findall(r"\\cite\w*\{([^}]*)\}", article.read_text(encoding="utf8"))
        cited = set()
        for group in groups:
            for key in group.split(","):
                cited.add(key.strip())
        missing = sorted(used - cited)
        if missing:
            context.alert(f"config references the article does not cite: {missing}")

    return table

# markdown table cell, without pipes and line breaks
def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")

# markdown table, only with the requested columns that exist
def markdown_table(table, columns=None, digits=3):
    if columns:
        chosen = []
        for column in columns:
            if column in table.columns:
                chosen.append(column)
        table = table[chosen]
    table = table.round(digits)

    # header and separator line
    header = []
    for column in table.columns:
        header.append(str(column))
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(table.columns)]

    for row in table.itertuples(index=False):
        values = []
        for value in row:
            values.append(cell(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)

# section of SUMMARY.md with the requested tables
# tables that were not generated get a warning in their place
def section(context, title, tables):
    lines = [f"## {title}", ""]
    for name, columns in tables:
        if context.has_table(name):
            lines += [f"**{name}**", "", markdown_table(context.read_table(name), columns), ""]
        else:
            lines += [f"`{name}` was not generated. See `status.json` and `logs/`.", ""]
    return lines

# report stage: SUMMARY.md and traceability
def run_report(context):
    context.save_table(traceability(context), "traceability")
    lines = ["# Run summary", ""]

    # one section per research question
    lines += section(context, "RQ1: which approach detects sexism best?", [
        ("test_results", ["detector", "scenario", "coverage", "f1_macro", "f1_macro_ci_low",
                          "f1_macro_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high", "mcc"]),
        ("comparison_mcnemar", None),
    ])
    lines += section(context, "RQ2: is there signal beyond the lexicon?", [
        ("subgroup_auc", None),
        ("influential_terms", ["model", "feature", "shap_importance", "direction", "lexicon_fraction",
                               "lexicon_term"]),
    ])
    lines += section(context, "RQ3: are the judges reliable?", [
        ("judges_prompt_selection", None),
        ("judges_stability", None),
        ("judges_agreement", None),
        ("judges_explanation_cases", ["case", "id", "target", "judge", "verdict", "explanation"]),
    ])
    lines += section(context, "RQ4: how much does each approach cost?", [
        ("cost_by_method", None),
        ("execution_time", None),
    ])
    lines += section(context, "Hyperparameter choices", [
        ("classical_best", None),
        ("transformer_chosen", None),
    ])

    # alerts of every stage
    alerts = context.path("alerts.csv")
    lines += ["## Alerts", ""]
    if alerts.exists():
        for row in pd.read_csv(alerts).itertuples():
            lines.append(f"- **{row.stage}:** {row.message}")

    context.path("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("SUMMARY.md saved", flush=True)

STAGE_FUNCTIONS = {
    "test": run_test,
    "comparison": run_comparison,
    "interpretation": run_interpretation,
    "cost": run_cost,
    "report": run_report,
}

if __name__ == "__main__":
    run_stage_cli(STAGE_FUNCTIONS)
