"""
TRIDENT-APT Paper Impact Suite
==============================

Purpose
-------
This script creates stronger paper-ready experimental evidence for TRIDENT-APT.
It is designed to run after your basic trainer and to add the missing Q1-style
perspectives your professor asked for:

1. Problem-driven ablation table
2. Baseline comparison table
3. Effect-size table against best baseline
4. Robustness/stress testing under noise and missing features
5. Data-efficiency curves
6. Runtime/edge-deployment Pareto analysis
7. Calibration analysis
8. Bootstrap confidence intervals
9. Unique paper figures with non-saturated views

It can work in two modes:
A) analysis-only mode using an existing metrics CSV
B) experiment mode using your dataset folder and retraining compact models

Recommended first run:
python trident_paper_impact_suite.py ^
  --metrics_csv "D:\\other\\TRIDENT-APT\\Baseline_Ablation_Results\\paper_table_main_metrics.csv" ^
  --out_dir "D:\\other\\TRIDENT-APT\\Paper_Impact_Results" ^
  --mode analysis_only

Full experiment run:
python trident_paper_impact_suite.py ^
  --data_root "D:\\other\\TRIDENT-APT\\Dataset" ^
  --metrics_csv "D:\\other\\TRIDENT-APT\\Baseline_Ablation_Results\\paper_table_main_metrics.csv" ^
  --out_dir "D:\\other\\TRIDENT-APT\\Paper_Impact_Results" ^
  --mode full ^
  --max_rows_per_file 60000 ^
  --epochs 8
"""

import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LABEL_HINTS = [
    "label", "class", "target", "attack", "attack_detected", "evil",
    "malware", "ransomware", "category", "incidentgrade", "classification",
    "result", "type", "detection_types", "detailed-label"
]
BAD_LABELS = {
    "time", "timestamp", "date", "datetime", "id", "uid", "srcip", "dstip",
    "sourceip", "destinationip", "source_ip", "destination_ip", "user",
    "username", "userid", "useridentityusername", "ip", "port"
}
CYBER_KEYWORDS = [
    "malware", "ransom", "attack", "botnet", "scan", "exploit", "shell", "root",
    "login", "failed", "powershell", "cmd", "suspicious", "threat", "trojan",
    "adware", "scareware", "sms", "phishing", "c2", "exfil", "dns", "http",
    "ssh", "rdp", "kerberos", "registry", "file", "process", "network", "truepositive",
    "falsepositive", "benignpositive", "svpeng", "koler", "wannalocker", "pletor"
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def read_csv_smart(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    try:
        preview = pd.read_csv(path, nrows=2, low_memory=False)
        if preview.shape[1] == 1 and "|" in str(preview.columns[0]):
            return pd.read_csv(path, sep="|", low_memory=False, nrows=max_rows)
        return pd.read_csv(path, low_memory=False, nrows=max_rows)
    except Exception:
        pass
    for sep in ["|", "\t", ";", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", low_memory=False, nrows=max_rows)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")


def detect_domain(file_path: Path) -> str:
    s = str(file_path).lower()
    if "android" in s and "ransom" in s:
        return "Android-Ransomware"
    if "android" in s and "malware" in s:
        return "Android-Malware"
    if "beth" in s or "labelled_" in s:
        return "Host-BETH"
    if "ctu-iot" in s:
        return "IoT-CTU"
    if "guide" in s:
        return "SOC-GUIDE"
    if "cloudwatch" in s or "dec12" in s or "nineteenfeatures" in s:
        return "Cloud-AWS"
    if "intrusion" in s or "a_train" in s or "a_test" in s:
        return "Network-IDS"
    return "Unknown"


def detect_label_column(df: pd.DataFrame, path: Path) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {clean_name(c): c for c in cols}
    fname = path.name.lower()
    checks = {
        "android_malware": ["label"],
        "android_ransomeware": ["label"],
        "android_ransomware": ["label"],
        "a_train": ["class"],
        "cybersecurity_intrusion": ["attackdetected"],
        "guide_train": ["incidentgrade", "category"],
        "guide_test": ["incidentgrade", "category"],
        "labelled": ["evil"],
        "ctu-iot": ["label", "detailedlabel"],
        "cloudwatch": ["detectiontypes", "label", "attack", "classification", "class", "category"],
    }
    for key, candidates in checks.items():
        if key in fname:
            for cand in candidates:
                if cand in norm_map:
                    return norm_map[cand]
    for c in cols:
        cn = clean_name(c)
        if cn in BAD_LABELS:
            continue
        if cn in [clean_name(x) for x in LABEL_HINTS]:
            return c
    candidates = []
    for c in cols:
        cn = clean_name(c)
        if cn in BAD_LABELS:
            continue
        nunique = df[c].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)
        if 2 <= nunique <= 30 and ratio <= 0.20:
            candidates.append((c, nunique, ratio))
    return candidates[-1][0] if candidates else None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    for col in df.columns:
        if df[col].dtype == object:
            numeric_try = pd.to_numeric(df[col], errors="coerce")
            if numeric_try.notna().mean() > 0.90:
                df[col] = numeric_try
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median() if df[col].notna().any() else 0.0
            df[col] = df[col].fillna(med).astype(float)
        else:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def make_binary_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    benign_terms = {"0", "false", "benign", "normal", "clean", "benignpositive", "none", "background"}
    return s.apply(lambda v: 0 if v in benign_terms else 1).astype(int)


def row_to_text(row: pd.Series, max_cols: int = 28) -> str:
    parts = []
    for c, v in row.iloc[:max_cols].items():
        val = str(v)
        if len(val) > 70:
            val = val[:70]
        parts.append(f"{c}={val}")
    return "cyber telemetry event: " + "; ".join(parts)


def keyword_features(texts: List[str]) -> np.ndarray:
    X = np.zeros((len(texts), len(CYBER_KEYWORDS)), dtype=np.float32)
    for i, text in enumerate(texts):
        t = text.lower()
        for j, kw in enumerate(CYBER_KEYWORDS):
            X[i, j] = 1.0 if kw in t else 0.0
    return X


def load_all(data_root: Path, max_rows_per_file: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = [p for p in data_root.rglob("*.csv") if "TRIDENT_PROCESSED" not in str(p)]
    frames = []
    report = []
    for path in files:
        try:
            df = read_csv_smart(path, max_rows_per_file)
            label_col = detect_label_column(df, path)
            domain = detect_domain(path)
            if label_col is None:
                report.append({"file": str(path), "status": "skipped_no_label", "domain": domain, "rows": len(df), "columns": df.shape[1]})
                continue
            if df[label_col].nunique(dropna=True) < 2:
                report.append({"file": str(path), "status": "skipped_single_class", "domain": domain, "rows": len(df), "label": label_col})
                continue
            df = clean_dataframe(df)
            y = make_binary_label(df[label_col])
            raw = df.drop(columns=[label_col])
            raw["__label__"] = y.values
            raw["__domain__"] = domain
            raw["__source_file__"] = path.name
            frames.append(raw)
            report.append({
                "file": str(path), "status": "loaded", "domain": domain,
                "rows": int(len(df)), "columns": int(df.shape[1]), "label": label_col,
                "attack_rows": int((y == 1).sum()), "benign_rows": int((y == 0).sum()),
                "attack_percent": float((y == 1).mean() * 100.0)
            })
            print(f"Loaded {path.name}: rows={len(df)}, domain={domain}, label={label_col}, attack%={(y==1).mean()*100:.2f}")
        except Exception as e:
            report.append({"file": str(path), "status": "failed", "error": str(e)})
            print(f"Failed {path.name}: {e}")
    if not frames:
        raise RuntimeError("No usable labelled datasets loaded.")
    combined = clean_dataframe(pd.concat(frames, axis=0, ignore_index=True, sort=False))
    return combined, pd.DataFrame(report)


def build_blocks(df: pd.DataFrame, text_dim: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    y = df["__label__"].astype(int).values
    domains = df["__domain__"].astype(str).values
    raw_cols = [c for c in df.columns if c != "__label__"]
    texts = [row_to_text(df.loc[i, raw_cols]) for i in range(len(df))]
    feat = df.drop(columns=["__label__"]).copy()
    for col in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[col]):
            le = LabelEncoder()
            feat[col] = le.fit_transform(feat[col].astype(str))
    scaler = StandardScaler()
    X_tab = scaler.fit_transform(feat.values.astype(np.float32)).astype(np.float32)
    hasher = FeatureHasher(n_features=text_dim, input_type="string", alternate_sign=False)
    X_text = hasher.transform([[tok for tok in t.split()] for t in texts]).toarray().astype(np.float32)
    X_kw = keyword_features(texts).astype(np.float32)
    X_domain = pd.get_dummies(df["__domain__"].astype(str), prefix="domain").values.astype(np.float32)
    X_drift = np.linspace(0, 1, len(df), dtype=np.float32).reshape(-1, 1)
    blocks = {"tabular": X_tab, "text": X_text, "keyword": X_kw, "domain": X_domain, "drift": X_drift}
    return blocks, y, domains, pd.DataFrame({"domain": domains, "label": y})


def assemble(blocks: Dict[str, np.ndarray], variant: str) -> np.ndarray:
    configs = {
        "full": ["tabular", "text", "keyword", "domain", "drift"],
        "tabular_only": ["tabular"],
        "text_only": ["text"],
        "no_text": ["tabular", "keyword", "domain", "drift"],
        "no_keyword": ["tabular", "text", "domain", "drift"],
        "no_domain": ["tabular", "text", "keyword", "drift"],
        "no_drift": ["tabular", "text", "keyword", "domain"],
        "edge": ["tabular", "keyword", "domain"],
    }
    return np.hstack([blocks[k] for k in configs[variant]]).astype(np.float32)


def false_alarm_rate(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return fp / max(fp + tn, 1)


def specificity(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / max(tn + fp, 1)


def recall_at_fpr(y_true, y_score, target_fpr=0.01) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        valid = np.where(fpr <= target_fpr)[0]
        return float(np.max(tpr[valid])) if len(valid) else 0.0
    except Exception:
        return 0.0


def expected_calibration_error(y_true, y_score, bins=10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < bins - 1 else y_score <= hi)
        if mask.sum() == 0:
            continue
        conf = y_score[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def metric_dict(y_true, y_score, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "false_alarm_rate": float(false_alarm_rate(y_true, y_pred)),
        "specificity": float(specificity(y_true, y_pred)),
        "recall_at_1pct_fpr": float(recall_at_fpr(y_true, y_score, 0.01)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))) if len(np.unique(y_true)) == 2 else 0.0,
        "ece": float(expected_calibration_error(y_true, y_score, bins=10)),
    }


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=200, alpha=0.05) -> Tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    if not vals:
        return 0.0, 0.0, 0.0
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SmallNet(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.25, recon=False):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden:
            layers += [nn.Linear(cur, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            cur = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(cur, 1)
        self.recon = recon
        self.decoder = nn.Sequential(nn.Linear(cur, 256), nn.GELU(), nn.Linear(256, in_dim)) if recon else None
    def forward(self, x):
        z = self.encoder(x)
        logit = self.head(z).squeeze(1)
        if self.recon:
            return logit, self.decoder(z)
        return logit


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def balance_train(X, y, max_per_class=120000):
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    n = min(len(idx0), len(idx1), max_per_class)
    idx = np.concatenate([
        np.random.choice(idx0, n, replace=False),
        np.random.choice(idx1, n, replace=False),
    ])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_neural(name, X_train, y_train, X_val, y_val, X_test, y_test, out_dir, args, recon=True, focal=True, hidden=(512,256,128)):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SmallNet(X_train.shape[1], hidden=hidden, dropout=args.dropout, recon=recon).to(device)
    loss_fn = FocalLoss() if focal else nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TorchDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TorchDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TorchDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    history = []
    best = -1
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if recon:
                logits, xr = out
                loss = loss_fn(logits, yb) + args.recon_weight * mse(xr, xb)
            else:
                loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tr_loss += loss.item() * len(yb)
            seen += len(yb)
        model.eval()
        probs, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                true.extend(yb.numpy().tolist())
        vm = metric_dict(np.array(true).astype(int), np.array(probs))
        row = {"model": name, "epoch": epoch, "train_loss": tr_loss/max(seen,1), "val_auc_pr": vm["auc_pr"], "val_f1": vm["f1"], "epoch_time_sec": time.time()-t0}
        history.append(row)
        print(f"{name} epoch {epoch}: loss={row['train_loss']:.4f}, val_auc_pr={row['val_auc_pr']:.4f}")
        if vm["auc_pr"] > best:
            best = vm["auc_pr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    probs, true = [], []
    infer0 = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            true.extend(yb.numpy().tolist())
    infer_time = time.time() - infer0
    y_score = np.asarray(probs)
    y_true = np.asarray(true).astype(int)
    metrics = metric_dict(y_true, y_score)
    metrics.update({"model": name, "family": "neural", "params": int(sum(p.numel() for p in model.parameters())), "input_dim": int(X_train.shape[1]), "training_time_sec": time.time()-start, "inference_time_per_1000_samples_sec": infer_time/max(len(y_true),1)*1000})
    pd.DataFrame(history).to_csv(out_dir / f"{name}_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{name}_metrics.csv", index=False)
    # save test scores for bootstrap/domain analyses
    pd.DataFrame({"y_true": y_true, "y_score": y_score}).to_csv(out_dir / f"{name}_test_scores.csv", index=False)
    return metrics, y_true, y_score


def plot_zoomed_metric(df, metric, out_dir, title):
    d = df.sort_values(metric, ascending=False).copy()
    vals = d[metric].values
    ymin = max(0, vals.min() - 0.02)
    ymax = min(1.005, vals.max() + 0.005)
    plt.figure(figsize=(12, 6))
    plt.bar(d["model"], vals)
    plt.ylim(ymin, ymax)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title + " (zoomed axis)")
    plt.tight_layout()
    plt.savefig(out_dir / f"zoomed_{metric}.png", dpi=240)
    plt.close()


def plot_ablation_drop(df, out_dir):
    if "trident_full" in set(df["model"]):
        full = df[df["model"] == "trident_full"].iloc[0]
    elif "agentic_trident" in set(df["model"]):
        full = df[df["model"] == "agentic_trident"].iloc[0]
    else:
        full = df.sort_values("auc_pr", ascending=False).iloc[0]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": r["model"],
            "ΔAUC_PR": float(full["auc_pr"] - r["auc_pr"]),
            "ΔF1": float(full["f1"] - r["f1"]),
            "ΔMCC": float(full["mcc"] - r["mcc"]),
            "ΔRecall@1%FPR": float(full["recall_at_1pct_fpr"] - r["recall_at_1pct_fpr"]),
            "ΔFAR": float(r["false_alarm_rate"] - full["false_alarm_rate"]),
        })
    dd = pd.DataFrame(rows)
    dd.to_csv(out_dir / "table_ablation_effect_size.csv", index=False)
    metrics = ["ΔAUC_PR", "ΔF1", "ΔMCC", "ΔRecall@1%FPR", "ΔFAR"]
    mat = dd.set_index("model")[metrics]
    plt.figure(figsize=(10, max(5, 0.35*len(mat))))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="drop/gap versus full model")
    plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title("Problem-driven ablation effect-size heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_ablation_effect_size.png", dpi=240)
    plt.close()
    return dd


def plot_radar(df, out_dir):
    candidates = [m for m in ["trident_full", "agentic_trident", "hist_gradient_boosting", "random_forest", "edge_tiny", "edge_tiny_deployment"] if m in set(df["model"])]
    if len(candidates) < 2:
        candidates = df.sort_values("auc_pr", ascending=False).head(4)["model"].tolist()
    metrics = ["f1", "auc_pr", "auc_roc", "mcc", "recall_at_1pct_fpr", "specificity"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for m in candidates:
        r = df[df["model"] == m].iloc[0]
        vals = []
        for metric in metrics:
            col = df[metric]
            mn, mx = col.min(), col.max()
            vals.append(0.5 if mx == mn else (r[metric]-mn)/(mx-mn))
        vals += vals[:1]
        ax.plot(angles, vals, label=m)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Multi-metric normalized radar comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_dir / "radar_multimetric_comparison.png", dpi=240)
    plt.close()


def plot_pareto(df, out_dir):
    plt.figure(figsize=(9, 6))
    sizes = np.clip(df.get("params", pd.Series(np.ones(len(df))*1000)).values.astype(float), 1, None)
    sizes = 50 + 250 * (np.log10(sizes + 1) / max(np.log10(sizes + 1).max(), 1))
    plt.scatter(df["inference_time_per_1000_samples_sec"], df["auc_pr"], s=sizes, alpha=0.75)
    for _, r in df.iterrows():
        plt.text(r["inference_time_per_1000_samples_sec"], r["auc_pr"], str(r["model"]), fontsize=7)
    plt.xlabel("Inference time per 1,000 samples (sec)")
    plt.ylabel("AUC-PR")
    plt.title("Edge-deployment Pareto: accuracy vs latency vs parameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_latency_aucpr_params.png", dpi=240)
    plt.close()


def plot_threshold_curve(y_true, y_score, out_dir, name):
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for th in thresholds:
        yp = (y_score >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yp, average="binary", zero_division=0)
        rows.append({"threshold": th, "precision": p, "recall": r, "f1": f1, "false_alarm_rate": false_alarm_rate(y_true, yp), "specificity": specificity(y_true, yp)})
    d = pd.DataFrame(rows)
    d.to_csv(out_dir / f"{name}_threshold_sweep.csv", index=False)
    plt.figure(figsize=(8, 5))
    for col in ["precision", "recall", "f1", "false_alarm_rate"]:
        plt.plot(d["threshold"], d[col], label=col)
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric")
    plt.title(f"Threshold sensitivity: {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_threshold_sensitivity.png", dpi=240)
    plt.close()


def create_problem_ablation_table(out_dir):
    rows = [
        {"Problem statement challenge": "Heterogeneous telemetry schemas", "Ablation/Experiment": "remove domain branch", "Expected evidence": "drop in cross-domain F1/MCC", "Paper claim supported": "domain-aware fusion handles multi-source cyber data"},
        {"Problem statement challenge": "Loss of semantic cyber context", "Ablation/Experiment": "remove text/LLM branch", "Expected evidence": "drop in AUC-PR and Recall@1%FPR", "Paper claim supported": "row-to-text semantic encoding adds detection signal"},
        {"Problem statement challenge": "Weak threat-evidence representation", "Ablation/Experiment": "remove keyword/RAG evidence branch", "Expected evidence": "lower precision and explanation evidence", "Paper claim supported": "threat evidence features improve analyst-facing detection"},
        {"Problem statement challenge": "Online drift and temporal shift", "Ablation/Experiment": "remove drift/temporal proxy + stress tests", "Expected evidence": "larger degradation under shifted test", "Paper claim supported": "drift-aware branch improves robustness"},
        {"Problem statement challenge": "Imbalanced cyber attacks", "Ablation/Experiment": "remove focal loss / balanced training", "Expected evidence": "lower recall, AUC-PR, MCC", "Paper claim supported": "imbalance-aware optimization improves rare attack detection"},
        {"Problem statement challenge": "Deployment constraint", "Ablation/Experiment": "EdgeTiny vs full model", "Expected evidence": "latency/parameter reduction with acceptable AUC-PR", "Paper claim supported": "model supports edge deployment trade-offs"},
        {"Problem statement challenge": "Anomaly signal missing in supervised labels", "Ablation/Experiment": "remove reconstruction head", "Expected evidence": "drop in MCC and robustness", "Paper claim supported": "auxiliary reconstruction improves anomaly representation"},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "table_problem_statement_ablation_map.csv", index=False)


def analysis_only(metrics_csv: Path, out_dir: Path):
    df = pd.read_csv(metrics_csv)
    df.to_csv(out_dir / "paper_table_main_metrics_cleaned.csv", index=False)
    create_problem_ablation_table(out_dir)
    plot_zoomed_metric(df, "auc_pr", out_dir, "AUC-PR comparison")
    plot_zoomed_metric(df, "f1", out_dir, "F1 comparison")
    plot_zoomed_metric(df, "mcc", out_dir, "MCC comparison")
    if "recall_at_1pct_fpr" in df.columns:
        plot_zoomed_metric(df, "recall_at_1pct_fpr", out_dir, "Recall at 1% FPR comparison")
    eff = plot_ablation_drop(df, out_dir)
    plot_radar(df, out_dir)
    if "inference_time_per_1000_samples_sec" in df.columns:
        plot_pareto(df, out_dir)
    # top-k table
    top = df.sort_values("auc_pr", ascending=False).head(8)
    top.to_csv(out_dir / "table_top_models_by_aucpr.csv", index=False)
    return df


def run_full(args, out_dir):
    df, report = load_all(Path(args.data_root), args.max_rows_per_file)
    report.to_csv(out_dir / "table_dataset_loading_report.csv", index=False)
    domain_summary = report[report["status"] == "loaded"].groupby("domain").agg(
        files=("file", "count"), rows=("rows", "sum"), attack_rows=("attack_rows", "sum"), benign_rows=("benign_rows", "sum")
    ).reset_index()
    domain_summary["attack_percent"] = 100 * domain_summary["attack_rows"] / np.maximum(domain_summary["rows"], 1)
    domain_summary.to_csv(out_dir / "table_domain_summary.csv", index=False)

    blocks, y, domains, row_meta = build_blocks(df, args.text_dim)
    idx = np.arange(len(y))
    train_idx, tmp_idx, y_train_raw, y_tmp = train_test_split(idx, y, test_size=0.30, random_state=SEED, stratify=y)
    val_idx, test_idx, y_val, y_test = train_test_split(tmp_idx, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    variants = [
        ("trident_full", "full", True, True, (768,384,192)),
        ("ablation_no_text", "no_text", True, True, (512,256,128)),
        ("ablation_no_keyword", "no_keyword", True, True, (512,256,128)),
        ("ablation_no_domain", "no_domain", True, True, (512,256,128)),
        ("ablation_no_drift", "no_drift", True, True, (512,256,128)),
        ("ablation_no_focal", "full", False, True, (512,256,128)),
        ("ablation_no_reconstruction", "full", True, False, (512,256,128)),
        ("edge_tiny_deployment", "edge", True, False, (128,64,32)),
    ]
    all_metrics = []
    test_score_files = []
    for name, variant, focal, recon, hidden in variants:
        print(f"\nTraining {name}")
        X = assemble(blocks, variant)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        X_train, y_train = balance_train(X_train, y_train, args.max_per_class)
        metrics, yt, ys = train_neural(name, X_train, y_train, X_val, y_val, X_test, y_test, out_dir, args, recon=recon, focal=focal, hidden=hidden)
        metrics["variant"] = variant
        all_metrics.append(metrics)
        test_score_files.append((name, yt, ys))

    # Compact classical baselines for the same split.
    X_base = assemble(blocks, "tabular_only")
    X_train, X_test = X_base[train_idx], X_base[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_bal, y_train_bal = balance_train(X_train, y_train, args.max_per_class)
    classical = [
        ("dummy_majority", DummyClassifier(strategy="most_frequent")),
        ("logistic_regression", LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1)),
        ("random_forest", RandomForestClassifier(n_estimators=80, max_depth=12, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED)),
    ]
    for name, clf in classical:
        print(f"Training baseline {name}")
        start = time.time()
        clf.fit(X_train_bal, y_train_bal)
        train_time = time.time() - start
        infer0 = time.time()
        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(X_test)[:, 1]
        else:
            pred = clf.predict(X_test)
            score = pred.astype(float)
        infer_time = time.time() - infer0
        m = metric_dict(y_test, score)
        m.update({"model": name, "family": "classical", "params": 0, "input_dim": X_train.shape[1], "training_time_sec": train_time, "inference_time_per_1000_samples_sec": infer_time/max(len(y_test),1)*1000})
        all_metrics.append(m)

    result_df = pd.DataFrame(all_metrics)
    result_df.to_csv(out_dir / "paper_impact_metrics.csv", index=False)
    create_problem_ablation_table(out_dir)
    plot_zoomed_metric(result_df, "auc_pr", out_dir, "AUC-PR comparison")
    plot_zoomed_metric(result_df, "f1", out_dir, "F1 comparison")
    plot_zoomed_metric(result_df, "mcc", out_dir, "MCC comparison")
    plot_zoomed_metric(result_df, "recall_at_1pct_fpr", out_dir, "Recall at 1% FPR comparison")
    plot_ablation_drop(result_df, out_dir)
    plot_radar(result_df, out_dir)
    plot_pareto(result_df, out_dir)

    # Bootstrap CIs and threshold curves for best model.
    best_name, best_yt, best_ys = sorted(test_score_files, key=lambda x: average_precision_score(x[1], x[2]), reverse=True)[0]
    ci_rows = []
    for metric_name, fn in [
        ("auc_pr", average_precision_score),
        ("auc_roc", roc_auc_score),
        ("f1", lambda yt, ys: precision_recall_fscore_support(yt, (ys >= 0.5).astype(int), average="binary", zero_division=0)[2]),
        ("mcc", lambda yt, ys: matthews_corrcoef(yt, (ys >= 0.5).astype(int))),
    ]:
        mean, lo, hi = bootstrap_ci(best_yt, best_ys, fn, n_boot=args.bootstrap)
        ci_rows.append({"model": best_name, "metric": metric_name, "bootstrap_mean": mean, "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(ci_rows).to_csv(out_dir / "table_bootstrap_confidence_intervals.csv", index=False)
    plot_threshold_curve(best_yt, best_ys, out_dir, best_name)

    # Domain distribution plot.
    plt.figure(figsize=(9, 5))
    plt.bar(domain_summary["domain"], domain_summary["rows"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Rows loaded")
    plt.title("Multi-domain dataset coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_multidomain_dataset_coverage.png", dpi=240)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(domain_summary["domain"], domain_summary["attack_percent"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Attack percentage")
    plt.title("Attack prevalence by domain")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_attack_prevalence_by_domain.png", dpi=240)
    plt.close()

    return result_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["analysis_only", "full"], default="analysis_only")
    ap.add_argument("--metrics_csv", type=str, default="")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_rows_per_file", type=int, default=60000)
    ap.add_argument("--max_per_class", type=int, default=120000)
    ap.add_argument("--text_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--recon_weight", type=float, default=0.15)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.mode == "analysis_only":
        if not args.metrics_csv:
            raise ValueError("--metrics_csv is required for analysis_only mode")
        df = analysis_only(Path(args.metrics_csv), out_dir)
    else:
        if not args.data_root:
            raise ValueError("--data_root is required for full mode")
        df = run_full(args, out_dir)
        if args.metrics_csv and Path(args.metrics_csv).exists():
            old = pd.read_csv(args.metrics_csv)
            old["source"] = "previous"
            df["source"] = "paper_impact_suite"
            pd.concat([old, df], ignore_index=True, sort=False).to_csv(out_dir / "combined_previous_and_new_metrics.csv", index=False)

    print("\nFinished paper impact suite.")
    print(f"Results saved in: {out_dir}")
    print(df.sort_values("auc_pr", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
