# code shared by the stages of the experiment: splits,
# lexicon, metrics, search spaces and optuna studies
# the algorithms come from libraries (the porter stemmer
# of nltk, the metrics of scikit-learn and the
# distributions of optuna), here I only connect them
# to the data of the experiment

import re

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from pipeline import StageFailed

# text read by the classifiers in each scenario
TEXT_COLUMN = {"full": "text", "masked": "masked_text"}
SCENARIOS = list(TEXT_COLUMN)

# splits saved by the split stage
# the development set is the union of the official
# train and validation sets of edos
PARTITIONS = {
    "train": ["train"],
    "validation": ["dev"],
    "development": ["train", "dev"],
    "test": ["test"],
}

# only these stages can read the test set,
# and only after the models and prompts are saved
STAGES_ALLOWED_TO_READ_TEST = {"test", "judges_test", "comparison", "interpretation"}

# words of the text, used in the comparison with the lexicon
TOKEN = re.compile(r"\w+", re.UNICODE)

# loads a split of the already divided data
# I put the test guard here so that no stage before
# the evaluation can read the test by accident
def load_partition(context, name):
    if name == "test" and context.name not in STAGES_ALLOWED_TO_READ_TEST:
        raise StageFailed(f"the {context.name} stage cannot read the test set")

    data = context.read_data("divided")
    chosen = data[data["split"].isin(PARTITIONS[name])]
    return chosen.reset_index(drop=True)

# hurtlex lemmas kept as sequences of porter stems
# an occurrence is a sequence of words of the text with the
# same stems of a lemma (lemmas with more than one word,
# like "loose woman", take several positions)
# the search goes from left to right and prefers
# the longest occurrence
class Lexicon:

    def __init__(self, lemmas):
        self.stemmer = PorterStemmer()

        # I kept the stems already calculated in a dictionary
        # because the same words repeat a lot in the texts
        self.stem_cache = {}

        # each term becomes a tuple with the stems of the words
        self.terms = set()
        for lemma in lemmas:
            words = TOKEN.findall(str(lemma).lower())
            if words:
                stems = []
                for word in words:
                    stems.append(self.stem(word))
                self.terms.add(tuple(stems))

        # number of words of the longest term
        self.longest = max(len(term) for term in self.terms)

    # porter stem of a word, using the cache
    def stem(self, word):
        if word not in self.stem_cache:
            self.stem_cache[word] = self.stemmer.stem(word)
        return self.stem_cache[word]

    # spans (start, end) of the text taken by lexicon terms
    def spans(self, text):
        tokens = list(TOKEN.finditer(text.lower()))
        stems = []
        for token in tokens:
            stems.append(self.stem(token.group()))

        spans = []
        position = 0
        while position < len(tokens):
            # tests the largest possible size first,
            # to keep the longest occurrence
            matched = 0
            largest_size = min(self.longest, len(tokens) - position)
            for size in range(largest_size, 0, -1):
                if tuple(stems[position:position + size]) in self.terms:
                    matched = size
                    break

            if matched > 0:
                start = tokens[position].start()
                end = tokens[position + matched - 1].end()
                spans.append((start, end))
                position += matched
            else:
                position += 1

        return spans

    # number of lexicon terms in the text
    def count(self, text):
        return len(self.spans(text))

    # removes the terms from the text, without putting a marker
    # in their place, and normalizes the spaces
    # I repeated the removal a few times because, when a term
    # is removed, the neighbor words can form a new term
    def mask(self, text):
        for _ in range(self.longest + 1):
            spans = self.spans(text)
            if not spans:
                break

            pieces = []
            last = 0
            for start, end in spans:
                pieces.append(text[last:start])
                last = end
            pieces.append(text[last:])

            text = " ".join("".join(pieces).split())

        return text

# reads the lexicon file, ignoring the empty lines
# and the comment lines of the header
def load_lexicon(path):
    lines = path.read_text(encoding="utf8").splitlines()
    lemmas = []
    for line in lines:
        if line and not line.startswith("#"):
            lemmas.append(line)
    return Lexicon(lemmas)

# auprc, or nan when the labels have a single class
def auprc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    else:
        return float(average_precision_score(labels, scores))

# threshold that maximizes the macro f1 among the
# observed scores (lipton et al., 2014)
# with more than 2000 different scores, I used only the quantiles
# so that the search would not get too slow
def macro_f1_threshold(labels, scores):
    candidates = np.unique(scores)
    if len(candidates) > 2000:
        candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 2001)))

    values = []
    for value in candidates:
        values.append(f1_score(labels, scores >= value, average="macro"))

    best_position = int(np.argmax(values))
    return float(candidates[best_position])

# binary classification metrics
# the scores, when they exist, feed the auprc and the roc auc
def classification_metrics(labels, predicted, scores=None):
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    row = {"n": len(labels), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    row["f1_macro"] = f1_score(labels, predicted, average="macro", zero_division=0)

    # metrics of each class, with the sexist class in position 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        labels=[1, 0],
        zero_division=0
    )
    row["precision_sexist"] = precision[0]
    row["recall_sexist"] = recall[0]
    row["f1_sexist"] = f1[0]
    row["precision_not_sexist"] = precision[1]
    row["recall_not_sexist"] = recall[1]
    row["f1_not_sexist"] = f1[1]

    # general metrics
    row["accuracy"] = accuracy_score(labels, predicted)
    row["balanced_accuracy"] = balanced_accuracy_score(labels, predicted)
    row["mcc"] = matthews_corrcoef(labels, predicted)
    row["kappa"] = cohen_kappa_score(labels, predicted)

    # I only calculate the auprc and the roc auc when both classes
    # show up and no score is missing
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(labels)) == 2 and not np.isnan(scores).any():
            row["auprc"] = average_precision_score(labels, scores)
            row["auc_roc"] = roc_auc_score(labels, scores)

    return row

# samples a hyperparameter according to the type
# defined in config.yaml
def suggest(trial, name, spec):
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["values"])
    elif kind == "integer":
        return trial.suggest_int(name, spec["min"], spec["max"])
    elif kind == "real":
        return trial.suggest_float(name, spec["min"], spec["max"])
    elif kind == "real_log":
        return trial.suggest_float(name, spec["min"], spec["max"], log=True)
    else:
        raise ValueError(f"unknown hyperparameter type: {kind}")

# samples every hyperparameter of a search space
def suggest_space(trial, space):
    params = {}
    for name, spec in space.items():
        params[name] = suggest(trial, name, spec)
    return params

# address of the sqlite database where optuna keeps the studies
def optuna_storage(context):
    return f"sqlite:///{context.path('optimization.db')}"

# name of the optuna study ending with the digest of the stage
# this way an interrupted run resumes its own study, and a
# change in the code, in the configuration or in the data starts a new one
def study_name(context, *parts):
    pieces = list(parts)
    pieces.append(context.short_hash)
    return "_".join(pieces)

# deletes the studies with the prefix of the stage that were
# created by another version of it
# I imported optuna here inside because not every stage
# that uses this file needs it
def drop_stale_studies(context, prefix):
    import optuna

    storage = optuna_storage(context)
    for name in optuna.get_all_study_names(storage=storage):
        if name.startswith(prefix) and not name.endswith(context.short_hash):
            optuna.delete_study(study_name=name, storage=storage)
