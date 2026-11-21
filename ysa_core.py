"""YSA vize projesi — ortak kod (02_nested_cv.ipynb ve 03_final_test.ipynb tarafından kullanılır).

İçerik:
    1. Config ve veri yükleme (development / test ayrımı results/dataset_split.csv dosyasından)
    2. Kat-içi (fold-local) ön işleme: pdays=999 dönüşümü, yüzdelik kırpma, imputation, ölçekleme, OHE
    3. PyTorch MLP (tek logit çıkışı) ve parametre sayımı
    4. Eğitim döngüsü: %15 early-stop alt kümesi, early stopping (patience 15), ReduceLROnPlateau
    5. Aday eğitimi (fit/e-stop bölmesi + ön işleme + eğitim) ve olasılık tahmini
    6. Metrikler ve otomatik (random search) aday örnekleme

Kısaltmalar: MLP (Multi-Layer Perceptron), OHE (One-Hot Encoding), BCE (Binary Cross-Entropy),
LR (Learning Rate), AP (Average Precision), ROC-AUC (Receiver Operating Characteristic - Area Under Curve).
"""
from __future__ import annotations

import copy
import json
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from torch import nn

# Kat içinde görülmeyen nadir kategoriler (ör. default='yes' tek satır) OHE'de sıfır vektöre düşer;
# handle_unknown="ignore" bunun için seçildi, her transform'da tekrarlanan uyarı bilgi taşımaz.
warnings.filterwarnings("ignore", message="Found unknown categories")

PDAYS_CODE = 999
UNKNOWN = "unknown"


# ---------------------------------------------------------------------------------------------
# 1. Config ve veri
# ---------------------------------------------------------------------------------------------
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_configs(root="."):
    root = Path(root)
    return {name: load_json(root / "configs" / f"{name}.json")
            for name in ["training", "seeds", "manual_candidates", "search_space"]}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _read_raw(root, data_cfg):
    return pd.read_csv(Path(root) / data_cfg["file"], sep=data_cfg["sep"])


def load_development(root, data_cfg):
    """Development özellikleri (DataFrame, orijinal satır indeksiyle) ve hedef (0/1 Series)."""
    raw = _read_raw(root, data_cfg)
    split = pd.read_csv(Path(root) / "results" / "dataset_split.csv")
    dev = raw.loc[split.loc[split["set"] == "development", "orig_index"]]
    X = dev.drop(columns=[data_cfg["target"], *data_cfg["dropped_features"]])
    y = (dev[data_cfg["target"]] == data_cfg["positive_label"]).astype(int)
    return X, y


def load_test_features(root, data_cfg):
    """Test özellikleri — hedef sütunu OKUNMADAN döndürülür (kilitli tahmin aşaması için)."""
    raw = _read_raw(root, data_cfg)
    split = pd.read_csv(Path(root) / "results" / "dataset_split.csv")
    test = raw.loc[split.loc[split["set"] == "test", "orig_index"]]
    return test.drop(columns=[data_cfg["target"], *data_cfg["dropped_features"]])


def load_test_labels(root, data_cfg):
    """Test etiketleri — yalnızca kilitli tahminler commit'lendikten SONRA çağrılır."""
    raw = _read_raw(root, data_cfg)
    split = pd.read_csv(Path(root) / "results" / "dataset_split.csv")
    test = raw.loc[split.loc[split["set"] == "test", "orig_index"]]
    return (test[data_cfg["target"]] == data_cfg["positive_label"]).astype(int)


def feature_types(X):
    return X.select_dtypes("number").columns.tolist(), X.select_dtypes("object").columns.tolist()


# ---------------------------------------------------------------------------------------------
# 2. Kat-içi ön işleme
# ---------------------------------------------------------------------------------------------
def pdays_code_to_nan(X):
    """pdays=999 ('son temas tarihi kayıtlı değil') → NaN. Durumsuz (stateless), hedefe bakmaz."""
    X = X.astype(float).copy()
    if "pdays" in X.columns:
        X["pdays"] = X["pdays"].mask(X["pdays"] == PDAYS_CODE)
    return X


def unknown_to_nan(X):
    """Kategorik 'unknown' → NaN (yalnızca A1 ek analizi: en sık sınıf imputasyonu)."""
    return X.mask(X == UNKNOWN)


class PercentileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Eşikleri yalnızca fit verisinde öğrenilen yüzdelik kırpma (NaN değerler korunur)."""

    def __init__(self, low=1.0, high=99.0):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lower_ = np.nanpercentile(X, self.low, axis=0)
        self.upper_ = np.nanpercentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def build_preprocessor(cand, num_cols, cat_cols, clip_percentiles=(1, 99)):
    """Adayın ön işleme ayarlarından (scaler, clipping, unknown stratejisi) ColumnTransformer kurar."""
    num_pipe = Pipeline([
        ("pdays_999", FunctionTransformer(pdays_code_to_nan, feature_names_out="one-to-one")),
        ("clip", PercentileClipper(*clip_percentiles) if cand.get("clipping", False) else "passthrough"),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", SCALERS[cand["scaler"]]()),
    ])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cand.get("unknown_strategy", "category") == "most_frequent":
        cat_pipe = Pipeline([
            ("unknown_nan", FunctionTransformer(unknown_to_nan, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])
    else:
        cat_pipe = Pipeline([("ohe", ohe)])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


# ---------------------------------------------------------------------------------------------
# 3. MLP
# ---------------------------------------------------------------------------------------------
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Girdi → [Linear → aktivasyon → (Dropout)] × gizli katman → Linear(1) = tek logit."""

    def __init__(self, n_in, hidden_layers, activation="relu", dropout=0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(cand, params, sgd_momentum=0.9):
    name, lr, wd = cand["optimizer"], cand["lr"], cand["weight_decay"]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=sgd_momentum, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        # PyTorch'ta AdamW'nin varsayılan weight_decay'i 0.01'dir; değer her zaman config'ten açıkça verilir
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


# ---------------------------------------------------------------------------------------------
# 4. Eğitim döngüsü
# ---------------------------------------------------------------------------------------------
def train_mlp(cand, X_fit, y_fit, X_es, y_es, seed, train_cfg):
    """Tek bir MLP eğitimi. X_* önceden işlenmiş numpy dizileridir.

    Ağırlıklar yalnızca fit kümesiyle güncellenir; e-stop kümesindeki BCE kaybı her epoch sonunda
    early stopping ve ReduceLROnPlateau tarafından izlenir. Dönüşte en iyi epoch'un ağırlıkları yüklüdür.
    """
    mlp_cfg, es_cfg, sch_cfg = train_cfg["mlp"], train_cfg["early_stopping"], train_cfg["lr_scheduler"]
    torch.manual_seed(seed)  # aynı başlangıç ağırlık politikası + dropout rastgeleliği
    Xf = torch.as_tensor(X_fit, dtype=torch.float32)
    yf = torch.as_tensor(np.asarray(y_fit), dtype=torch.float32)
    Xe = torch.as_tensor(X_es, dtype=torch.float32)
    ye = torch.as_tensor(np.asarray(y_es), dtype=torch.float32)

    model = MLP(Xf.shape[1], cand["hidden_layers"], cand["activation"], cand["dropout"])
    opt = make_optimizer(cand, model.parameters(), mlp_cfg["sgd_momentum"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=sch_cfg["mode"], factor=sch_cfg["factor"], patience=sch_cfg["patience"], min_lr=sch_cfg["min_lr"])
        if cand["scheduler"] else None)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)  # mini-batch karıştırma sırası (DataLoader(shuffle=True) eşdeğeri)
    n, bs = len(Xf), cand["batch_size"]

    history, warn = [], []
    best_loss, best_epoch, best_state, wait = np.inf, 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, mlp_cfg["max_epochs"] + 1):
        lr_now = opt.param_groups[0]["lr"]
        # 1) Öğrenme: fit kümesi üzerinde mini-batch geçişi
        model.train()
        perm, running = torch.randperm(n, generator=gen), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n
        # 2) Ölçüm: e-stop kümesinde BCE (gradyan yok, dropout kapalı)
        model.eval()
        with torch.no_grad():
            es_loss = loss_fn(model(Xe), ye).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "es_loss": es_loss, "lr": lr_now})
        if not (np.isfinite(train_loss) and np.isfinite(es_loss)):
            warn.append("nan_loss")
            break
        # 3) Karar: en iyi mi? → LR scheduler → durdurma
        if es_loss < best_loss:
            best_loss, best_epoch, wait = es_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if sched is not None:
            sched.step(es_loss)
        if wait >= es_cfg["patience"]:
            break

    stopped_epoch = history[-1]["epoch"]
    early_stopped = wait >= es_cfg["patience"]
    if best_state is not None:
        model.load_state_dict(best_state)
    if not early_stopped and "nan_loss" not in warn:
        warn.append("max_epoch_reached")
    if best_epoch == 1:
        warn.append("best_epoch_1")

    hist = pd.DataFrame(history)
    info = {"best_epoch": best_epoch, "stopped_epoch": stopped_epoch, "early_stopped": early_stopped,
            "best_es_loss": best_loss, "final_lr": opt.param_groups[0]["lr"],
            "n_lr_drops": int((hist["lr"].diff() < 0).sum()), "n_params": count_params(model),
            "train_time_s": time.perf_counter() - t0, "warnings": ";".join(warn)}
    return model, hist, info


# ---------------------------------------------------------------------------------------------
# 5. Aday eğitimi: fit / e-stop bölmesi + kat-içi ön işleme + eğitim
# ---------------------------------------------------------------------------------------------
@dataclass
class FittedMLP:
    cand: dict
    preprocessor: ColumnTransformer
    model: MLP
    history: pd.DataFrame
    info: dict = field(default_factory=dict)

    def predict_proba(self, X):
        Z = torch.as_tensor(self.preprocessor.transform(X), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(Z)).numpy()


def fit_candidate(cand, X, y, cfgs, weight_seed=None):
    """Bir adayı verilen eğitim bölümünde (inner-train, outer-train veya tüm development) eğitir.

    (i) %85 fit alt kümesi: ön işleme parametreleri öğrenilir + MLP ağırlıkları güncellenir
    (ii) %15 stratified e-stop alt kümesi: yalnızca early stopping ve LR scheduler için izlenir
    Değerlendirme verisi bu fonksiyona hiç girmez.
    """
    train_cfg, seeds = cfgs["training"], cfgs["seeds"]
    t0 = time.perf_counter()
    fit_idx, es_idx = train_test_split(np.arange(len(X)), test_size=train_cfg["cv"]["early_stop_fraction"],
                                       stratify=y, random_state=seeds["early_stop_subset"])
    X_fit, X_es = X.iloc[fit_idx], X.iloc[es_idx]
    y_fit, y_es = np.asarray(y)[fit_idx], np.asarray(y)[es_idx]

    num_cols, cat_cols = feature_types(X)
    pre = build_preprocessor(cand, num_cols, cat_cols, train_cfg["preprocessing"]["clipping_percentiles"])
    Z_fit = pre.fit_transform(X_fit)  # fit YALNIZCA fit alt kümesinde
    Z_es = pre.transform(X_es)

    seed = seeds["weight_init_primary"] if weight_seed is None else weight_seed
    model, hist, info = train_mlp(cand, Z_fit, y_fit, Z_es, y_es, seed, train_cfg)
    info.update({"weight_seed": seed, "n_fit": len(fit_idx), "n_es": len(es_idx), "n_es_pos": int(y_es.sum()),
                 "n_features": Z_fit.shape[1], "total_time_s": time.perf_counter() - t0})
    return FittedMLP(cand, pre, model, hist, info)


def reference_preprocessor(X):
    """Referans modeller (Logistic Regression) için sabit ön işleme kuralı: M1 ile aynı (Standard, kırpma yok)."""
    num_cols, cat_cols = feature_types(X)
    return build_preprocessor({"scaler": "standard", "clipping": False, "unknown_strategy": "category"},
                              num_cols, cat_cols)


# ---------------------------------------------------------------------------------------------
# 6. Metrikler ve otomatik aday örnekleme
# ---------------------------------------------------------------------------------------------
def classification_metrics(y_true, p, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"AP": average_precision_score(y_true, p), "ROC_AUC": roc_auc_score(y_true, p),
            "Accuracy": (tp + tn) / len(y_true), "Precision": prec, "Sensitivity": sens, "Specificity": spec,
            "F1": f1, "Balanced_Accuracy": (sens + spec) / 2, "TP": int(tp), "FP": int(fp), "TN": int(tn),
            "FN": int(fn)}


def sample_random_candidates(space, seed):
    """Her optimizer için tam 2 aday (R1–R2: SGD, R3–R4: Adam, R5–R6: AdamW), log-ölçekli LR."""
    cands = []
    for k, opt_name in enumerate(space["optimizers"]):
        lo, hi = space["lr_log_uniform"][opt_name]
        dist = {"lr": loguniform(lo, hi), "hidden_layers": space["hidden_layers"],
                "activation": space["activation"], "weight_decay": space["weight_decay"],
                "batch_size": space["batch_size"], "scaler": space["scaler"], "dropout": space["dropout"],
                "clipping": space["clipping"]}
        for params in ParameterSampler(dist, n_iter=space["candidates_per_optimizer"], random_state=seed + k):
            params = {key: (v.item() if hasattr(v, "item") else v) for key, v in params.items()}
            cands.append({"id": f"R{len(cands) + 1}", "optimizer": opt_name, **params,
                          "lr": float(f"{params['lr']:.3g}"), "scheduler": space["fixed"]["scheduler"],
                          "unknown_strategy": "category"})
    return cands


OPTIMIZER_NAMES = {"sgd": "SGD+M", "adam": "Adam", "adamw": "AdamW"}

# Grafik stili (02 ve 03 ortak) — yöntem renkleri tüm şekillerde sabit
METHOD_COLORS = {"MLP_manual": "#2a78d6", "MLP_random": "#1baf7a", "LogReg": "#eb6834", "Dummy": "#a9a8a2"}
METHOD_LABELS = {"MLP_manual": "MLP (manuel arama)", "MLP_random": "MLP (otomatik arama)",
                 "LogReg": "Logistic Regression", "Dummy": "Dummy (prior)"}
INK, MUTED = "#52514e", "#a9a8a2"


def set_plot_style():
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook", rc={
        "grid.color": "#e6e5e0", "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold",
        "text.color": "#0b0b0b", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def describe_candidate(c):
    """Tablolar için kısa açıklama, ör. 'AdamW (32,) relu lr=0.001 wd=1e-04 bs=64 standard do=0.0 clip=0 sch=1'."""
    return (f"{OPTIMIZER_NAMES[c['optimizer']]} {tuple(c['hidden_layers'])} {c['activation']} lr={c['lr']:g} "
            f"wd={c['weight_decay']:.0e} bs={c['batch_size']} {c['scaler']} do={c['dropout']} "
            f"clip={int(c.get('clipping', False))} sch={int(c['scheduler'])}")
