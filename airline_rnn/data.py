from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import digest, initialize, object_digest, read_json, save_json, utc_now
from .text import WordTokenizer, canonical_text, clean_text, pad_sequences, source_clean

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/crowdflower/twitter-airline-sentiment"
DATASET_SHA256 = "ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e"
DATASET_PAGE = "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment"


def load_raw(root):
    root = initialize(root)
    target = root / "data/raw/Tweets.csv"
    if not target.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "airline-rnn-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
            member = next(name for name in archive.namelist() if name.endswith("Tweets.csv"))
            target.write_bytes(archive.read(member))
    if digest(target) != DATASET_SHA256:
        raise ValueError("Dataset checksum changed. Audit the new version before running the frozen protocol.")
    frame = pd.read_csv(target, dtype={"tweet_id": "string"})
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


class UnionFind:
    def __init__(self, ids):
        self.parent = {int(i): int(i) for i in ids}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def quality_filter(frame, cleaner):
    frame = frame.copy()
    excluded = []
    valid = frame["text"].notna() & frame["text"].astype(str).str.strip().ne("")
    valid &= frame["airline_sentiment"].isin(["negative", "neutral", "positive"])
    for row_id in frame.loc[~valid, "row_id"]:
        excluded.append({"row_id": int(row_id), "reason": "missing_or_invalid_text_or_label"})
    frame = frame.loc[valid].copy()
    original_columns = [c for c in frame if c != "row_id"]
    exact_duplicate = frame.duplicated(original_columns, keep="first")
    for row_id in frame.loc[exact_duplicate, "row_id"]:
        excluded.append({"row_id": int(row_id), "reason": "exact_duplicate_record"})
    frame = frame.loc[~exact_duplicate].copy()
    frame["clean_text"] = frame["text"].map(cleaner)
    frame["canonical_text"] = frame["text"].map(canonical_text)
    groups = UnionFind(frame["row_id"])
    for column in ("tweet_id", "canonical_text", "clean_text"):
        first = {}
        for row_id, value in frame[["row_id", column]].itertuples(index=False, name=None):
            if pd.isna(value) or value in ("", "emptymarker"):
                continue
            if value in first:
                groups.union(int(row_id), first[value])
            else:
                first[value] = int(row_id)
    frame["group_id"] = frame["row_id"].map(lambda i: groups.find(int(i)))
    conflicts = frame.groupby("group_id")["airline_sentiment"].nunique()
    conflict_groups = conflicts[conflicts > 1].index
    conflict_rows = frame[frame["group_id"].isin(conflict_groups)].copy()
    for row_id in conflict_rows["row_id"]:
        excluded.append({"row_id": int(row_id), "reason": "conflicting_labels_in_duplicate_group"})
    frame = frame[~frame["group_id"].isin(conflict_groups)].copy()
    return frame, pd.DataFrame(excluded, columns=["row_id", "reason"]), conflict_rows


def stratified_groups(frame, ratios, seed):
    groups = frame.groupby("group_id", sort=True)["airline_sentiment"].first().reset_index()
    train_groups, remaining = train_test_split(
        groups, test_size=ratios[1] + ratios[2], random_state=seed,
        stratify=groups["airline_sentiment"])
    val_groups, test_groups = train_test_split(
        remaining, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=seed,
        stratify=remaining["airline_sentiment"])
    assignment = {int(group): split for split, subset in
                  (("train", train_groups), ("validation", val_groups), ("test", test_groups))
                  for group in subset["group_id"]}
    frame["split"] = frame["group_id"].map(assignment)
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    return frame


def prepare_data(root):
    root = initialize(root)
    raw = load_raw(root)
    profile = {"source": DATASET_PAGE, "sha256": digest(root / "data/raw/Tweets.csv"),
               "raw_rows": len(raw), "raw_columns": len(raw.columns) - 1,
               "class_counts": raw["airline_sentiment"].value_counts().to_dict(),
               "airline_counts": raw["airline"].value_counts().to_dict(),
               "missing_counts": raw.drop(columns="row_id").isna().sum().astype(int).to_dict(),
               "duplicate_tweet_ids": int(raw["tweet_id"].duplicated().sum()),
               "duplicate_raw_texts": int(raw["text"].duplicated().sum()), "benchmarks": {}}
    source = raw[raw["airline_sentiment"] != "neutral"].copy()
    source = source.drop(source[source["airline_sentiment"] == "negative"].iloc[:5000].index)
    if len(source) != 6541:
        raise ValueError("Source population differs from the audited notebook")
    source["clean_text"] = source["text"].map(source_clean)
    train_ids, test_ids = train_test_split(source["row_id"].to_numpy(), test_size=0.3,
                                          stratify=source["airline_sentiment"].to_numpy(), random_state=1)
    cut = int(len(train_ids) * 0.8)
    ordered = [*train_ids[:cut], *train_ids[cut:], *test_ids]
    source = source.set_index("row_id").loc[ordered].reset_index()
    source["split"] = ["train"] * cut + ["validation"] * (len(train_ids) - cut) + ["test"] * len(test_ids)
    source["group_id"] = source["row_id"]  # Legacy split intentionally has no group guard.
    clean_binary, excluded_binary, conflict_binary = quality_filter(
        raw.loc[raw["row_id"].isin(source["row_id"])], source_clean)
    clean_binary = stratified_groups(clean_binary, (0.56, 0.14, 0.30), 1)
    clean_three, excluded_three, conflict_three = quality_filter(raw, clean_text)
    clean_three = stratified_groups(clean_three, (0.70, 0.15, 0.15), 42)
    for benchmark, frame in (("binary_source", source), ("binary_clean", clean_binary),
                             ("three_class", clean_three)):
        frame.to_csv(root / f"data/interim/{benchmark}.csv", index=False)
        manifest_columns = ["row_id", "tweet_id", "airline_sentiment", "group_id", "split"]
        path = root / f"data/splits/{benchmark}.csv"
        frame[manifest_columns].to_csv(path, index=False)
        profile["benchmarks"][benchmark] = {
            "rows": len(frame), "groups": int(frame["group_id"].nunique()),
            "counts": pd.crosstab(frame["split"], frame["airline_sentiment"]).to_dict(orient="index"),
            "manifest_sha256": digest(path), "legacy_contamination": benchmark == "binary_source"}
    excluded_binary.to_csv(root / "data/interim/exclusions_binary.csv", index=False)
    excluded_three.to_csv(root / "data/interim/exclusions_three_class.csv", index=False)
    conflict_binary.to_csv(root / "data/interim/conflicts_binary.csv", index=False)
    conflict_three.to_csv(root / "data/interim/conflicts_three_class.csv", index=False)
    save_json(root / "results/metrics/dataset_profile.json", profile)
    save_json(root / "data/raw/manifest.json", {"source": DATASET_PAGE, "download_url": DATASET_URL,
              "sha256": DATASET_SHA256, "rows": len(raw), "checked_at_utc": utc_now()})
    return profile


def load_benchmark(root, benchmark):
    root = initialize(root)
    path = root / f"data/interim/{benchmark}.csv"
    if not path.exists():
        prepare_data(root)
    return pd.read_csv(path, dtype={"tweet_id": "string"}, keep_default_na=False)


def encode_frame(frame, tokenizer, length, legacy):
    sequences, stats = [], []
    for row in frame.itertuples(index=False):
        sequence, info = tokenizer.encode(row.clean_text)
        sequences.append(sequence)
        info.update({"row_id": int(row.row_id), "truncated": len(sequence) > length,
                     "empty_sequence": len(sequence) == 0})
        stats.append(info)
    return pad_sequences(sequences, length, legacy), pd.DataFrame(stats)


def prepare_representation(root, config):
    root = initialize(root)
    frame = load_benchmark(root, config.benchmark)
    legacy = config.benchmark != "three_class"
    # Source Tokenizer was fitted before splitting: preserve original row order,
    # including the tie-breaking order of equally frequent words.
    fit = frame.sort_values("row_id") if config.benchmark == "binary_source" else frame[frame["split"] == "train"]
    tokenizer = WordTokenizer(config.vocabulary_size, legacy=legacy).fit(fit["clean_text"])
    length = config.sequence_length
    if length is None:
        length = max(len(tokenizer.encode(text)[0]) for text in fit["clean_text"])
    length = max(length, 1)
    folder = root / f"data/processed/{config.id}"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "tokenizer.json", tokenizer.to_dict())
    save_json(folder / "label_map.json", config.labels)
    arrays, stats_by_split = {}, {}
    for split in ("train", "validation"):
        subset = frame[frame["split"] == split]
        matrix, stats = encode_frame(subset, tokenizer, length, legacy)
        arrays[f"x_{split}"] = matrix
        arrays[f"y_{split}"] = subset["airline_sentiment"].map(config.labels).to_numpy(dtype="int32")
        arrays[f"ids_{split}"] = subset["row_id"].to_numpy(dtype="int64")
        stats.to_csv(folder / f"{split}_representation_stats.csv", index=False)
        stats_by_split[split] = {"rows": len(subset), "oov_token_rate": float(stats.unknown_tokens.sum() / max(stats.token_length.sum(), 1)),
                                "truncation_rate": float(stats.truncated.mean()),
                                "empty_sequence_count": int(stats.empty_sequence.sum()),
                                "length_percentiles": {str(q): float(stats.token_length.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)}}
    np.savez_compressed(folder / "train_validation.npz", **arrays)
    metadata = {"config_id": config.id, "benchmark": config.benchmark, "sequence_length": length,
                "padding": "pre" if legacy else "post", "truncating": "pre" if legacy else "post",
                "legacy": legacy, "tokenizer_sha256": digest(folder / "tokenizer.json"),
                "split_sha256": digest(root / f"data/splits/{config.benchmark}.csv"),
                "dataset_sha256": DATASET_SHA256, "fit_row_ids_sha256": object_digest([int(i) for i in fit.row_id]),
                "fit_scope": "all_source_population" if config.benchmark == "binary_source" else "train_only",
                "word_index_size": len(tokenizer.word_index), "stats": stats_by_split}
    save_json(folder / "representation.json", metadata)
    return metadata
