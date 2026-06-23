# D:\other\TRIDENT-APT\Code\dataset_inspector_v2.py

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

warnings.filterwarnings("ignore")

DATASET_DIR = Path(r"D:\other\TRIDENT-APT\Dataset")
OUTPUT_DIR = DATASET_DIR / "TRIDENT_PROCESSED_V2"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

LABEL_HINTS = [
    "label", "class", "target", "attack", "attack_detected",
    "evil", "malware", "ransomware", "category", "incidentgrade",
    "classification", "result", "type"
]

BAD_LABELS = [
    "time", "timestamp", "date", "datetime", "id", "uid",
    "src_ip", "dst_ip", "source_ip", "destination_ip",
    "user", "username", "userid", "useridentityusername",
    "ip", "port", "src_port", "dst_port"
]


def read_csv_smart(path):
    """
    Handles normal CSV and CTU/Bro-style separator files.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.shape[1] > 1:
            return df, ","
    except Exception:
        pass

    separators = ["\t", ";", "|", r"\s+"]
    for sep in separators:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", low_memory=False)
            if df.shape[1] > 1:
                return df, sep
        except Exception:
            continue

    # CTU files often have one header line with spaces.
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#", low_memory=False)
        return df, r"\s+"
    except Exception:
        pass

    df = pd.read_csv(path, low_memory=False)
    return df, ","


def clean_col_name(col):
    return str(col).strip().lower().replace(" ", "").replace("_", "")


def detect_label_column(df, file_name):
    cols = list(df.columns)
    normalized = {clean_col_name(c): c for c in cols}

    # dataset-specific fixes
    fname = file_name.lower()

    if "beth" in fname or "labelled" in fname:
        for c in cols:
            if clean_col_name(c) == "evil":
                return c

    if "android_malware" in fname or "android_ransomeware" in fname or "android_ransomware" in fname:
        for c in cols:
            if clean_col_name(c) == "label":
                return c

    if "cybersecurity_intrusion" in fname:
        for c in cols:
            if clean_col_name(c) == "attackdetected":
                return c

    if "a_train" in fname:
        for c in cols:
            if clean_col_name(c) == "class":
                return c

    if "a_test" in fname:
        # Some NSL-KDD test files do not contain final class.
        # Do not use is_guest_login as label.
        for c in cols:
            if clean_col_name(c) == "class":
                return c
        return None

    if "guide_train" in fname or "guide_test" in fname:
        for candidate in ["incidentgrade", "category"]:
            if candidate in normalized:
                return normalized[candidate]

    if "cloudwatch" in fname:
        for candidate in ["label", "attack", "classification", "class", "category"]:
            if candidate in normalized:
                return normalized[candidate]
        return None

    if "dec12" in fname or "nineteenfeatures" in fname:
        for candidate in ["label", "attack", "classification", "class", "category"]:
            if candidate in normalized:
                return normalized[candidate]
        return None

    # general search by exact known label names
    for c in cols:
        cn = clean_col_name(c)
        if cn in BAD_LABELS:
            continue
        if cn in [clean_col_name(x) for x in LABEL_HINTS]:
            return c

    # fallback: low-cardinality non-bad column near the end
    candidates = []
    for c in cols:
        cn = clean_col_name(c)
        if cn in BAD_LABELS:
            continue

        nunique = df[c].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)

        if 2 <= nunique <= 30 and ratio <= 0.20:
            candidates.append((c, nunique, ratio))

    if candidates:
        return candidates[-1][0]

    return None


def label_distribution(df, label_col):
    counts = df[label_col].value_counts(dropna=False)
    pct = df[label_col].value_counts(normalize=True, dropna=False) * 100
    return counts.to_dict(), pct.round(5).to_dict(), float(pct.min())


def clean_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("missing").astype(str)
        else:
            median = df[col].median() if df[col].notna().any() else 0
            df[col] = df[col].fillna(median)

    return df


def can_stratify(y, test_size):
    counts = y.value_counts()
    if len(counts) < 2:
        return False
    return counts.min() >= 2


def safe_split(df, label_col):
    """
    Handles tiny classes safely.
    If class has only 1 sample, no stratification is used.
    """

    y = df[label_col]

    stratify_1 = y if can_stratify(y, 0.30) else None

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify_1
    )

    temp_y = temp_df[label_col]
    stratify_2 = temp_y if can_stratify(temp_y, 0.50) else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=stratify_2
    )

    return train_df, val_df, test_df


def balance_train_only(train_df, label_col, max_per_class=None):
    """
    Balances training set only.
    Uses undersampling by default.
    Prevents memory explosion on huge files.
    """

    counts = train_df[label_col].value_counts()

    if len(counts) < 2:
        return train_df

    min_count = int(counts.min())

    if min_count < 2:
        # Cannot create reliable balanced supervised training.
        # Keep original train and flag later.
        return train_df

    if max_per_class is not None:
        n = min(min_count, max_per_class)
    else:
        n = min_count

    parts = []

    for cls in counts.index:
        cls_df = train_df[train_df[label_col] == cls]
        parts.append(
            resample(
                cls_df,
                replace=False,
                n_samples=n,
                random_state=RANDOM_STATE
            )
        )

    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)


def inspect_and_prepare(path):
    try:
        df, sep = read_csv_smart(path)
    except Exception as e:
        return {
            "file": str(path),
            "status": "failed_read",
            "error": str(e)
        }, None

    label_col = detect_label_column(df, path.name)

    stats = {
        "file": str(path),
        "file_name": path.name,
        "status": "ok",
        "separator_used": sep,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns)),
        "label_column": label_col,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include=[np.number]).shape[1]),
        "categorical_columns": int(df.select_dtypes(exclude=[np.number]).shape[1]),
    }

    if label_col is None:
        stats.update({
            "class_distribution": None,
            "class_percentage": None,
            "minority_class_percentage": None,
            "is_imbalanced": None,
            "split_status": "skipped_no_valid_label"
        })
        return stats, None

    counts, pct, minority_pct = label_distribution(df, label_col)
    stats["class_distribution"] = counts
    stats["class_percentage"] = pct
    stats["minority_class_percentage"] = round(minority_pct, 5)
    stats["is_imbalanced"] = bool(minority_pct < 20.0)

    if df[label_col].nunique(dropna=True) < 2:
        stats["split_status"] = "skipped_single_class"
        return stats, None

    try:
        clean_df = clean_dataframe(df)
        train_df, val_df, test_df = safe_split(clean_df, label_col)

        # cap huge balanced training per class to keep experiments practical
        balanced_train_df = balance_train_only(
            train_df,
            label_col,
            max_per_class=200000
        )

        safe_name = path.stem.replace(" ", "_").replace(".", "_")
        out_dir = OUTPUT_DIR / safe_name
        out_dir.mkdir(exist_ok=True)

        balanced_train_df.to_csv(out_dir / "train_balanced.csv", index=False)
        val_df.to_csv(out_dir / "validation_original.csv", index=False)
        test_df.to_csv(out_dir / "test_original.csv", index=False)

        split_report = {
            "dataset": safe_name,
            "label_column": label_col,
            "original_rows": int(len(clean_df)),
            "train_balanced_rows": int(len(balanced_train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "original_distribution": clean_df[label_col].value_counts().to_dict(),
            "train_balanced_distribution": balanced_train_df[label_col].value_counts().to_dict(),
            "validation_distribution": val_df[label_col].value_counts().to_dict(),
            "test_distribution": test_df[label_col].value_counts().to_dict(),
            "output_folder": str(out_dir)
        }

        stats["split_status"] = "created"
        return stats, split_report

    except Exception as e:
        stats["split_status"] = "failed_split"
        stats["split_error"] = str(e)
        return stats, None


def main():
    csv_files = list(DATASET_DIR.rglob("*.csv"))
    csv_files = [p for p in csv_files if "TRIDENT_PROCESSED" not in str(p)]

    print(f"Found {len(csv_files)} CSV files")

    all_stats = []
    split_reports = []

    for path in csv_files:
        print(f"\nInspecting: {path.name}")

        stats, report = inspect_and_prepare(path)
        all_stats.append(stats)

        print(f"Rows: {stats.get('rows')}")
        print(f"Columns: {stats.get('columns')}")
        print(f"Separator: {stats.get('separator_used')}")
        print(f"Label column: {stats.get('label_column')}")
        print(f"Imbalanced: {stats.get('is_imbalanced')}")
        print(f"Split status: {stats.get('split_status')}")

        if report:
            split_reports.append(report)

    summary_df = pd.DataFrame(all_stats)
    summary_path = OUTPUT_DIR / "dataset_inspection_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with open(OUTPUT_DIR / "dataset_inspection_full.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4, default=str)

    with open(OUTPUT_DIR / "balanced_split_report.json", "w", encoding="utf-8") as f:
        json.dump(split_reports, f, indent=4, default=str)

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Full report: {OUTPUT_DIR / 'dataset_inspection_full.json'}")
    print(f"Split report: {OUTPUT_DIR / 'balanced_split_report.json'}")


if __name__ == "__main__":
    main()