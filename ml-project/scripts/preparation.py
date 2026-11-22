# data preparation: from the raw edos to the
# train, validation and test splits
#
# usage: python scripts/preparation.py <stage> --run-dir ... --config ...
#
# stages: hardware, data, cleaning, lexicon, split and eda
# - data: downloads edos and hurtlex at the pinned commits (only if
#   they do not exist yet) and records the sha-256 digests of the files
# - cleaning: normalizes the unicode (nfkc) and finds near duplicates
#   with minhash (datasketch library) over word 5-grams,
#   confirmed by the exact jaccard index above 0.8
#   (lee et al., 2022), the duplicates form groups (connected
#   components, networkx library)
#   to avoid leaking the test, train or validation posts that
#   are in the group of a test post are removed, and in the other
#   groups the first post stays
# - lexicon: counts the hurtlex terms in each post and creates
#   the masked text, without the terms
# - split: keeps the official edos split and saves the splits
#   as csv
# - eda: describes the development set, including the
#   agreement of the lexicon rule with the human label
#
# the test posts go through the normalization, the removal
# of duplicates and the mask, which do not depend on the label, but
# no table of these stages uses the test labels, which are only
# read in phase 4

import hashlib
import unicodedata
import urllib.request
from datetime import date

import networkx
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from pipeline import (
    ROOT,
    StageFailed,
    describe_hardware,
    histogram,
    make_vram_sampler,
    measure,
    run_stage_cli,
)
from common import TEXT_COLUMN, load_lexicon, load_partition

SEXIST = "sexist"

# order of preference when one post of a
# group of duplicates has to stay
SPLIT_ORDER = {"test": 0, "train": 1, "dev": 2}

# columns and names of the splits saved as csv
FEATURE_COLUMNS = ["id", "rewire_id", "text", "masked_text", "lexicon_terms"]
EXPORT_NAMES = {"train": "train", "dev": "validation", "test": "test"}

# hardware stage: describes the machine and checks
# if the vram can be read
def run_hardware(context):
    info = describe_hardware()
    sampler = make_vram_sampler()
    info["vram_readable"] = sampler is not None

    if sampler is None:
        context.alert("VRAM cannot be read: the cost will only have the execution times.")
    else:
        info["vram_used_at_start_mb"] = round(sampler.read_mb(), 1)

    items = []
    values = []
    for item, value in info.items():
        items.append(item)
        values.append(str(value))

    table = pd.DataFrame({"item": items, "value": values})
    context.save_table(table, "hardware")

# sha-256 digest of a file
# reads the file in chunks of 1 MB to avoid
# loading the whole file into memory
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        chunk = handle.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(1024 * 1024)
    return digest.hexdigest()

# downloads a file, if it does not exist yet
# the download goes to a .part file first, so that
# an interrupted download does not look like a complete file
def download(url, destination):
    if destination.exists():
        print(f"[already exists] {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[downloading] {url}")
    partial = destination.with_suffix(destination.suffix + ".part")

    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    partial.replace(destination)

# posts per split and, outside the test, the
# count and the proportion of sexist posts
def partition_summary(data):
    rows = []
    for split, part in data.groupby("split"):
        row = {"split": split, "posts": len(part)}
        if split != "test":
            row["sexist"] = int(part["target"].sum())
            row["sexist_proportion"] = float(part["target"].mean())
        rows.append(row)

    return pd.DataFrame(rows, columns=["split", "posts", "sexist", "sexist_proportion"])

# hurtlex lemmas in the requested categories and level,
# without repetition and in lowercase
def read_hurtlex(path, categories, level):
    table = pd.read_csv(path, sep="\t", keep_default_na=False)
    in_categories = table["category"].isin(categories)
    in_level = table["level"] == level
    chosen = table[in_categories & in_level]

    if chosen.empty:
        raise StageFailed(f"no HurtLex term in {categories} with level {level}")

    lemmas = chosen["lemma"].str.strip().str.lower().unique()
    return sorted(lemmas)

# data stage: download and reading of edos and hurtlex
def run_data(context):
    data_settings = context.config["data"]
    lexicon_settings = context.config["lexicon"]
    base = ROOT / data_settings["raw_dataset"]
    lexicon_source = ROOT / lexicon_settings["raw"]
    download(data_settings["base_url"], base)
    download(lexicon_settings["url"], lexicon_source)

    # table with the columns the experiment uses
    edos = pd.read_csv(base)
    data = pd.DataFrame(
        {
            "id": np.arange(len(edos)),
            "rewire_id": edos["rewire_id"],
            "original_text": edos["text"],
            "target": (edos["label_sexist"] == SEXIST).astype(int),
            "category": edos["label_category"],
            "split": edos["split"],
        }
    )
    context.save_data(data, "raw")

    # lexicon with a header saying where the terms came from
    lemmas = read_hurtlex(lexicon_source, lexicon_settings["categories"], lexicon_settings["level"])
    header = [
        f"# HurtLex (Bassignana et al., 2018): {lexicon_settings['url']}",
        f"# categories {', '.join(lexicon_settings['categories'])}, level {lexicon_settings['level']}",
    ]
    lexicon_text = "\n".join(header + lemmas) + "\n"
    context.path("interim", "lexicon.txt").write_text(lexicon_text, encoding="utf8")

    # digest of the downloaded files, to check
    # later if they are the same
    hashes = {}
    for path in [base, lexicon_source]:
        if path.is_relative_to(ROOT):
            name = str(path.relative_to(ROOT))
        else:
            name = str(path)
        hashes[name] = file_sha256(path)

    record = {
        "retrieval_date": date.today().isoformat(),
        "posts": len(data),
        "sexist_in_train_and_validation": int(data.loc[data["split"] != "test", "target"].sum()),
        "splits": data["split"].value_counts().to_dict(),
        "lexicon_lemmas": len(lemmas),
        "sha256": hashes,
    }
    context.save_json(record, "data.json")
    context.save_table(partition_summary(data), "data_splits")

# set of word n-grams (shingles) of the text
# a text with fewer words than n becomes a single shingle
def shingles(text, n_words):
    words = text.split()
    if len(words) < n_words:
        return {" ".join(words)}
    else:
        result = set()
        for start in range(len(words) - n_words + 1):
            result.add(" ".join(words[start:start + n_words]))
        return result

# jaccard index between two sets
def jaccard(first, second):
    return len(first & second) / len(first | second)

# pairs of posts with exact jaccard above the minimum,
# found by minhash lsh
def duplicate_pairs(texts, settings):
    n_words = settings["shingle_words"]
    sets = []
    for text in texts:
        sets.append(shingles(text, n_words))

    index = MinHashLSH(
        num_perm=settings["minhash_permutations"],
        params=(settings["minhash_bands"], settings["minhash_rows"]),
    )

    # minhash signature of each post
    # I used the same seed in every signature
    # so that they can be compared with each other
    signatures = []
    for position, items in enumerate(sets):
        signature = MinHash(num_perm=settings["minhash_permutations"], seed=1)
        encoded = []
        for item in items:
            encoded.append(item.encode("utf8"))
        signature.update_batch(encoded)
        signatures.append(signature)
        index.insert(position, signature)

    # lsh only suggests candidates, so I confirmed
    # each pair with the exact jaccard
    pairs = []
    for position, signature in enumerate(signatures):
        for other in index.query(signature):
            if other <= position:
                continue
            similarity = jaccard(sets[position], sets[other])
            if similarity > settings["min_jaccard"]:
                pairs.append((position, other, similarity))

    return pairs

# positions that leave the data
# in the group of a test post, the train and validation
# posts leave, and in the other groups the first post stays
# (train before validation, then by id)
def to_remove(data, pairs):
    graph = networkx.Graph()
    for first, second, _ in pairs:
        graph.add_edge(first, second)

    removed = set()
    for component in networkx.connected_components(graph):
        # sorts the group by the order of preference
        ranked = []
        for position in component:
            split = data["split"].iloc[position]
            ranked.append((SPLIT_ORDER[split], position))
        ranked.sort()

        members = []
        for _, position in ranked:
            members.append(position)

        has_test = False
        for position in members:
            if data["split"].iloc[position] == "test":
                has_test = True

        # the test posts never leave
        for position in members:
            if data["split"].iloc[position] == "test":
                continue
            if has_test or position != members[0]:
                removed.add(position)

    return sorted(removed)

# cleaning stage: normalization and near duplicates
def run_cleaning(context):
    settings = context.config["cleaning"]
    data = context.read_data("raw")
    form = settings["unicode_normalization"]

    # I measured it without the method name because the preprocessing
    # goes into the time of the stages, and not into the cost of a detector
    with measure(context, phase="preprocessing", n_documents=len(data)):
        data["text"] = data["original_text"].map(lambda text: unicodedata.normalize(form, text))
        pairs = duplicate_pairs(data["text"].tolist(), settings)
        removed = to_remove(data, pairs)

    # table with the pairs of near duplicates
    pair_rows = []
    for first, second, similarity in pairs:
        pair_rows.append(
            {
                "id_a": int(data["id"].iloc[first]),
                "split_a": data["split"].iloc[first],
                "id_b": int(data["id"].iloc[second]),
                "split_b": data["split"].iloc[second],
                "jaccard": similarity,
            }
        )
    pair_table = pd.DataFrame(pair_rows, columns=["id_a", "split_a", "id_b", "split_b", "jaccard"])
    context.save_table(pair_table, "cleaning_duplicate_pairs")

    # removed posts
    removed_rows = data.iloc[removed]
    context.save_table(
        removed_rows[["id", "rewire_id", "split", "target", "text"]],
        "cleaning_removed"
    )

    # how many posts went through each step of the cleaning
    normalized = int((data["text"] != data["original_text"]).sum())
    funnel = pd.DataFrame(
        [
            {"step": "original", "posts": len(data)},
            {"step": "normalized (NFKC)", "posts": normalized},
            {"step": "near-duplicate pairs", "posts": len(pairs)},
            {"step": "removed", "posts": len(removed)},
            {"step": "final", "posts": len(data) - len(removed)},
        ]
    )
    context.save_table(funnel, "cleaning_funnel")

    data = data.drop(index=data.index[removed]).reset_index(drop=True)
    context.save_data(data, "clean")

# lexicon stage: term count and masked text
def run_lexicon(context):
    data = context.read_data("clean")
    lexicon = load_lexicon(context.path("interim", "lexicon.txt"))

    with measure(context, phase="preprocessing", n_documents=len(data)):
        data["lexicon_terms"] = data["text"].map(lexicon.count)
        data["masked_text"] = data["text"].map(lexicon.mask)

    # checks if the mask removed every term
    remaining_counts = data["masked_text"].map(lexicon.count)
    remaining = int((remaining_counts > 0).sum())
    if remaining:
        raise StageFailed(f"{remaining} masked texts still have lexicon terms")

    context.save_data(data, "labeled")

# split stage: keeps the official split and
# saves the splits as csv
def run_split(context):
    data = context.read_data("labeled")
    context.save_data(data, "divided")
    context.save_table(partition_summary(data), "split_summary")

    # one X and one y per split, separated by folder
    folder = ROOT / context.config["data"]["processed"]
    for split, name in EXPORT_NAMES.items():
        part = data[data["split"] == split]
        (folder / name).mkdir(parents=True, exist_ok=True)
        part[FEATURE_COLUMNS].to_csv(folder / name / f"X_{name}.csv", index=False)
        part[["id", "target"]].to_csv(folder / name / f"y_{name}.csv", index=False)

    print(f"CSV splits saved in {folder}", flush=True)

# the 30 lexicon terms that show up the most in the texts
def term_frequencies(texts, lexicon):
    counts = {}
    for text in texts:
        for start, end in lexicon.spans(text):
            term = text[start:end].lower()
            counts[term] = counts.get(term, 0) + 1

    table = pd.DataFrame(list(counts.items()), columns=["term", "occurrences"])
    return table.sort_values("occurrences", ascending=False, kind="stable").head(30)

# eda stage: exploratory analysis of the development set
def run_eda(context):
    dev = load_partition(context, "development")
    lexicon = load_lexicon(context.path("interim", "lexicon.txt"))

    # measures of each post
    dev = dev.assign(
        characters=dev["text"].str.len(),
        words=dev["text"].str.split().str.len(),
        has_term=dev["lexicon_terms"] > 0,
        removed_fraction=1 - dev["masked_text"].str.len() / dev["text"].str.len(),
    )

    # summary by class
    by_class = dev.groupby("target").agg(
        posts=("id", "size"),
        median_characters=("characters", "median"),
        median_words=("words", "median"),
        with_lexicon_term=("has_term", "mean"),
        mean_removed_fraction=("removed_fraction", "mean"),
    )
    context.save_table(by_class.reset_index(), "eda_by_class")

    categories = (
        dev["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="posts")
    )
    context.save_table(categories, "eda_categories")
    context.save_table(term_frequencies(dev["text"], lexicon), "eda_frequent_terms")

    # agreement of the lexicon rule with the human
    # label, only in the development set
    rule = (dev["lexicon_terms"] >= context.config["lexicon"]["rule_threshold"]).astype(int)
    agreement = pd.crosstab(dev["target"], rule)
    agreement.index.name = "target"

    columns = []
    for column in agreement.columns:
        columns.append(f"rule_{column}")
    agreement.columns = columns
    context.save_table(agreement.reset_index(), "lexicon_rule_vs_target")

    histogram(context, "eda_length", dev["words"], "Length of the posts", "Words")

    # warns when the mask erased the whole text
    empty = int((dev[TEXT_COLUMN["masked"]].str.strip() == "").sum())
    if empty:
        context.alert(f"{empty} posts became empty after the mask.")

STAGE_FUNCTIONS = {
    "hardware": run_hardware,
    "data": run_data,
    "cleaning": run_cleaning,
    "lexicon": run_lexicon,
    "split": run_split,
    "eda": run_eda,
}

if __name__ == "__main__":
    run_stage_cli(STAGE_FUNCTIONS)
