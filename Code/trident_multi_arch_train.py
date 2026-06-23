"""
TRIDENT-APT Multi-Architecture Trainer
Author: ChatGPT for TRIDENT-AP

What this script does:
1. Reads your cyber CSV datasets from a root folder.
2. Fixes CTU-IoT pipe-separated files.
3. Detects valid label columns safely.
4. Cleans numeric/categorical/string columns without median-on-string errors.
5. Creates balanced training data only.
6. Trains 5 architectures:
   - agentic_trident
   - edge_tiny
   - rag_feature
   - continual_drift
   - ensemble_multiview
7. Runs ablations.
8. Saves metrics, figures, confusion matrix, ROC, PR, and training curves.

Example:
python trident_multi_arch_train.py --data_root "D:\\other\\TRIDENT-APT\\Dataset" --out_dir "D:\\other\\TRIDENT-APT\\Results" --mode ablation --epochs 10 --max_rows_per_file 100000
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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
    "ssh", "rdp", "kerberos", "registry", "file", "process", "network"
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def read_csv_smart(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read normal CSV or CTU pipe-separated CSV."""
    # CTU files have header like ts|uid|...|label|detailed-label but extension csv.
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
    raise RuntimeError(f"Could not read file: {path}")


def detect_domain(file_path: Path) -> str:
    s = str(file_path).lower()
    if "android" in s and "ransom" in s:
        return "android_ransomware"
    if "android" in s and "malware" in s:
        return "android_malware"
    if "beth" in s or "labelled_" in s:
        return "host_beth"
    if "ctu-iot" in s:
        return "iot_ctu"
    if "guide" in s:
        return "soc_guide"
    if "cloudwatch" in s or "dec12" in s or "nineteenfeatures" in s:
        return "cloud_aws"
    if "intrusion" in s or "a_train" in s or "a_test" in s:
        return "network_ids"
    return "unknown"


def detect_label_column(df: pd.DataFrame, path: Path) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {clean_name(c): c for c in cols}
    fname = path.name.lower()

    dataset_specific = {
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
    for key, candidates in dataset_specific.items():
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
    if candidates:
        return candidates[-1][0]
    return None


def make_binary_label(series: pd.Series) -> pd.Series:
    """Map labels to binary: benign/normal/0/false -> 0, everything else -> 1."""
    s = series.astype(str).str.strip().str.lower()
    benign_terms = {"0", "false", "benign", "normal", "clean", "benignpositive", "none", "background"}
    return s.apply(lambda v: 0 if v in benign_terms else 1).astype(int)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()

    for col in df.columns:
        # Convert numeric-looking object columns safely.
        if df[col].dtype == object:
            numeric_try = pd.to_numeric(df[col], errors="coerce")
            non_null_ratio = numeric_try.notna().mean()
            if non_null_ratio > 0.90:
                df[col] = numeric_try

        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median() if df[col].notna().any() else 0.0
            df[col] = df[col].fillna(med).astype(float)
        else:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def row_to_text(row: pd.Series, max_cols: int = 30) -> str:
    parts = []
    for c, v in row.iloc[:max_cols].items():
        val = str(v)
        if len(val) > 80:
            val = val[:80]
        parts.append(f"{c}={val}")
    return " cyber event with " + "; ".join(parts)


def cyber_keyword_features(texts: List[str]) -> np.ndarray:
    arr = np.zeros((len(texts), len(CYBER_KEYWORDS)), dtype=np.float32)
    lower_texts = [t.lower() for t in texts]
    for i, t in enumerate(lower_texts):
        for j, kw in enumerate(CYBER_KEYWORDS):
            arr[i, j] = 1.0 if kw in t else 0.0
    return arr


def false_alarm_rate(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return fp / max(fp + tn, 1)


def specificity(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return tn / max(tn + fp, 1)


def recall_at_fpr(y_true, y_score, target_fpr=0.01) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        valid = np.where(fpr <= target_fpr)[0]
        if len(valid) == 0:
            return 0.0
        return float(np.max(tpr[valid]))
    except Exception:
        return 0.0


class CyberDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            cur = h
        self.net = nn.Sequential(*layers)
        self.out_dim = cur

    def forward(self, x):
        return self.net(x)


class EdgeTinyNet(nn.Module):
    """Deployment/edge perspective: fast, small, low-parameter model."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class AgenticTRIDENT(nn.Module):
    """Agentic perspective: full multi-view model with gated fusion style MLP."""
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = MLPBlock(in_dim, [1024, 512, 256], dropout=0.25)
        self.recon = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, in_dim))
        self.head = nn.Linear(256, 1)
    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z).squeeze(1)
        recon = self.recon(z)
        return logits, recon


class RAGFeatureNet(nn.Module):
    """RAG/SOC perspective: adds threat-keyword and evidence features before detection."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = MLPBlock(in_dim, [768, 384, 192], dropout=0.2)
        self.head = nn.Linear(192, 1)
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


class ContinualDriftNet(nn.Module):
    """Continual/drift perspective: robust representation with smaller bottleneck."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = MLPBlock(in_dim, [512, 256, 128], dropout=0.35)
        self.head = nn.Linear(128, 1)
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


class EnsembleMultiViewNet(nn.Module):
    """Multi-perspective ensemble: three parallel neural views fused late."""
    def __init__(self, in_dim):
        super().__init__()
        self.a = MLPBlock(in_dim, [512, 128], dropout=0.2)
        self.b = MLPBlock(in_dim, [256, 128], dropout=0.3)
        self.c = MLPBlock(in_dim, [128, 128], dropout=0.1)
        self.head = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, x):
        z = torch.cat([self.a(x), self.b(x), self.c(x)], dim=1)
        return self.head(z).squeeze(1)


def build_model(name: str, in_dim: int) -> nn.Module:
    if name == "edge_tiny":
        return EdgeTinyNet(in_dim)
    if name == "agentic_trident":
        return AgenticTRIDENT(in_dim)
    if name == "rag_feature":
        return RAGFeatureNet(in_dim)
    if name == "continual_drift":
        return ContinualDriftNet(in_dim)
    if name == "ensemble_multiview":
        return EnsembleMultiViewNet(in_dim)
    raise ValueError(f"Unknown model: {name}")


def load_all_datasets(data_root: Path, max_rows_per_file: Optional[int]) -> pd.DataFrame:
    files = [p for p in data_root.rglob("*.csv") if "TRIDENT_PROCESSED" not in str(p)]
    frames = []
    report = []

    for path in files:
        try:
            df = read_csv_smart(path, max_rows=max_rows_per_file)
            label_col = detect_label_column(df, path)
            domain = detect_domain(path)
            if label_col is None:
                report.append({"file": str(path), "status": "skipped_no_label", "rows": len(df), "columns": df.shape[1]})
                continue
            if df[label_col].nunique(dropna=True) < 2:
                report.append({"file": str(path), "status": "skipped_single_class", "rows": len(df), "label": label_col})
                continue

            df = clean_dataframe(df)
            y = make_binary_label(df[label_col])
            df = df.drop(columns=[label_col])
            df["__label__"] = y.values
            df["__domain__"] = domain
            df["__source_file__"] = path.name
            frames.append(df)
            report.append({
                "file": str(path), "status": "loaded", "rows": len(df), "columns": df.shape[1],
                "label": label_col, "domain": domain, "class_distribution": y.value_counts().to_dict()
            })
            print(f"Loaded {path.name}: rows={len(df)}, label={label_col}, domain={domain}, dist={y.value_counts().to_dict()}")
        except Exception as e:
            report.append({"file": str(path), "status": "failed", "error": str(e)})
            print(f"Failed {path.name}: {e}")

    if not frames:
        raise RuntimeError("No usable labelled datasets were loaded.")

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return clean_dataframe(combined), report


def prepare_features(df: pd.DataFrame, text_dim: int, model_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    y = df["__label__"].astype(int).values

    # Build row text from non-label columns.
    raw_cols = [c for c in df.columns if c not in ["__label__"]]
    texts = [row_to_text(df.loc[i, raw_cols]) for i in range(len(df))]

    # Numeric features.
    feature_df = df.drop(columns=["__label__"]).copy()
    for col in feature_df.columns:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    scaler = StandardScaler()
    X_numcat = scaler.fit_transform(feature_df.values.astype(np.float32))

    # Hashing text branch, lightweight and offline. This is the working substitute for LLM embeddings.
    # Later you can replace this with MiniLM embeddings without changing the trainer.
    hasher = FeatureHasher(n_features=text_dim, input_type="string", alternate_sign=False)
    X_text = hasher.transform([[tok for tok in t.split()] for t in texts]).toarray().astype(np.float32)

    # RAG/SOC evidence branch.
    X_kw = cyber_keyword_features(texts)

    # Domain branch is already encoded in numeric/categorical, but keep domain indicator stronger.
    domain_dummies = pd.get_dummies(df["__domain__"].astype(str), prefix="domain").values.astype(np.float32)

    if model_name == "edge_tiny":
        X = X_numcat.astype(np.float32)
    elif model_name == "rag_feature":
        X = np.hstack([X_numcat, X_text, X_kw, domain_dummies]).astype(np.float32)
    elif model_name == "continual_drift":
        # Add simple drift proxy: row index percentile inside mixed stream.
        drift = np.linspace(0, 1, len(df), dtype=np.float32).reshape(-1, 1)
        X = np.hstack([X_numcat, X_text, domain_dummies, drift]).astype(np.float32)
    else:
        X = np.hstack([X_numcat, X_text, X_kw, domain_dummies]).astype(np.float32)

    meta = {"input_dim": int(X.shape[1]), "text_dim": text_dim, "rows": int(len(X))}
    return X, y, meta


def balance_training_data(X_train: np.ndarray, y_train: np.ndarray, max_per_class: int = 200000):
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X_train, y_train
    n = min(len(idx0), len(idx1), max_per_class)
    idx0_s = np.random.choice(idx0, n, replace=False)
    idx1_s = np.random.choice(idx1, n, replace=False)
    idx = np.concatenate([idx0_s, idx1_s])
    np.random.shuffle(idx)
    return X_train[idx], y_train[idx]


def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.0
    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except Exception:
        auc_pr = 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "false_alarm_rate": float(false_alarm_rate(y_true, y_pred)),
        "specificity": float(specificity(y_true, y_pred)),
        "recall_at_1pct_fpr": float(recall_at_fpr(y_true, y_prob, 0.01)),
    }


def plot_curves(history: List[Dict], out_dir: Path, model_name: str):
    if not history:
        return
    hist = pd.DataFrame(history)
    for metric in ["train_loss", "val_loss", "val_f1", "val_auc_pr", "epoch_time_sec"]:
        if metric not in hist.columns:
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(hist["epoch"], hist[metric], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{model_name}: {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{metric}.png", dpi=200)
        plt.close()


def plot_eval_figures(y_true, y_prob, out_dir: Path, model_name: str):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"{model_name}: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Benign", "Attack"])
    plt.yticks([0, 1], ["Benign", "Attack"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name}: ROC Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_roc_curve.png", dpi=200)
        plt.close()
    except Exception:
        pass

    try:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(r, p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name}: Precision-Recall Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_pr_curve.png", dpi=200)
        plt.close()
    except Exception:
        pass


def train_one_model(model_name: str, X: np.ndarray, y: np.ndarray, out_dir: Path, args) -> Dict:
    ensure_dir(out_dir)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    if args.balance_train:
        X_train, y_train = balance_training_data(X_train, y_train, max_per_class=args.max_per_class)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(model_name, X.shape[1]).to(device)

    train_loader = DataLoader(CyberDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CyberDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CyberDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos = max((y_train == 1).sum(), 1)
    neg = max((y_train == 0).sum(), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    history = []
    best_val = -1
    best_state = None
    patience_left = args.patience
    start_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            if model_name == "agentic_trident":
                logits, recon = model(xb)
                loss = bce(logits, yb) + args.recon_weight * mse(recon, xb)
            else:
                logits = model(xb)
                loss = bce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += loss.item() * len(yb)
            n_seen += len(yb)
        scheduler.step()

        # validation
        model.eval()
        val_probs = []
        val_true = []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if model_name == "agentic_trident":
                    logits, recon = model(xb)
                    loss = bce(logits, yb) + args.recon_weight * mse(recon, xb)
                else:
                    logits = model(xb)
                    loss = bce(logits, yb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                val_probs.extend(probs.tolist())
                val_true.extend(yb.detach().cpu().numpy().tolist())
                val_loss += loss.item() * len(yb)
        val_probs = np.array(val_probs)
        val_true = np.array(val_true).astype(int)
        val_metrics = compute_metrics(val_true, val_probs)
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_seen, 1),
            "val_loss": val_loss / max(len(val_true), 1),
            "val_f1": val_metrics["f1"],
            "val_auc_pr": val_metrics["auc_pr"],
            "val_auc_roc": val_metrics["auc_roc"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        print(f"{model_name} epoch {epoch:03d}: train_loss={row['train_loss']:.4f} val_f1={row['val_f1']:.4f} val_auc_pr={row['val_auc_pr']:.4f} time={epoch_time:.1f}s")

        if val_metrics["auc_pr"] > best_val:
            best_val = val_metrics["auc_pr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping {model_name} at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    test_probs = []
    test_true = []
    infer_t0 = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if model_name == "agentic_trident":
                logits, _ = model(xb)
            else:
                logits = model(xb)
            test_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            test_true.extend(yb.numpy().tolist())
    infer_time = time.time() - infer_t0
    test_probs = np.array(test_probs)
    test_true = np.array(test_true).astype(int)
    metrics = compute_metrics(test_true, test_probs)
    metrics.update({
        "model": model_name,
        "input_dim": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "training_time_sec": float(time.time() - start_total),
        "inference_time_sec": float(infer_time),
        "inference_time_per_1000_samples_sec": float(infer_time / max(len(X_test), 1) * 1000),
        "device": str(device),
    })

    pd.DataFrame(history).to_csv(out_dir / f"{model_name}_training_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{model_name}_test_metrics.csv", index=False)
    torch.save(model.state_dict(), out_dir / f"{model_name}.pt")
    plot_curves(history, out_dir, model_name)
    plot_eval_figures(test_true, test_probs, out_dir, model_name)
    with open(out_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def plot_ablation_summary(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["auc_pr"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("AUC-PR")
    plt.title("Ablation Comparison: AUC-PR")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_auc_pr_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(df["training_time_sec"], df["auc_pr"])
    for _, r in df.iterrows():
        plt.text(r["training_time_sec"], r["auc_pr"], r["model"], fontsize=8)
    plt.xlabel("Training Time (sec)")
    plt.ylabel("AUC-PR")
    plt.title("Training Time vs AUC-PR")
    plt.tight_layout()
    plt.savefig(out_dir / "training_time_vs_auc_pr.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="ablation", choices=["single", "ablation"])
    parser.add_argument("--model", type=str, default="agentic_trident", choices=["agentic_trident", "edge_tiny", "rag_feature", "continual_drift", "ensemble_multiview"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--text_dim", type=int, default=256)
    parser.add_argument("--max_rows_per_file", type=int, default=100000)
    parser.add_argument("--max_per_class", type=int, default=200000)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--recon_weight", type=float, default=0.2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("Loading datasets...")
    combined, load_report = load_all_datasets(data_root, args.max_rows_per_file)
    with open(out_dir / "load_report.json", "w", encoding="utf-8") as f:
        json.dump(load_report, f, indent=2, default=str)
    combined_sample_path = out_dir / "combined_loaded_sample.csv"
    combined.head(1000).to_csv(combined_sample_path, index=False)

    models = [args.model] if args.mode == "single" else [
        "agentic_trident",
        "edge_tiny",
        "rag_feature",
        "continual_drift",
        "ensemble_multiview",
    ]

    all_metrics = []
    for model_name in models:
        print("\n" + "=" * 80)
        print(f"Preparing features for {model_name}")
        X, y, meta = prepare_features(combined, text_dim=args.text_dim, model_name=model_name)
        with open(out_dir / f"{model_name}_feature_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Training {model_name}: X={X.shape}, y_dist={pd.Series(y).value_counts().to_dict()}")
        model_out = out_dir / model_name
        metrics = train_one_model(model_name, X, y, model_out, args)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "all_model_metrics.csv", index=False)
    plot_ablation_summary(metrics_df, out_dir)

    print("\nFinished.")
    print(f"Results saved in: {out_dir}")
    print(metrics_df[["model", "f1", "auc_pr", "auc_roc", "mcc", "false_alarm_rate", "training_time_sec", "params"]])


if __name__ == "__main__":
    main()
