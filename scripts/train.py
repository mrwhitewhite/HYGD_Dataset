from pyprojroot import here
import warnings
from pathlib import Path
from typing import Tuple, Dict, List

import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
import timm

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    Flipd,
    Rotate90d,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandAffined,
    NormalizeIntensityd,
    ToTensord,
    MapTransform,
)
from monai.utils import set_determinism
from monai.config import KeysCollection

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")


# ==================== CONFIG ====================
class Config:
    # ==================== Paths ====================
    PARENTS = here()
    DATA_DIR = PARENTS / Path("data")
    LABELS_CSV = DATA_DIR / "Labels.csv"
    IMAGES_DIR = DATA_DIR / "Images"
    OUTPUT_DIR = PARENTS / Path("outputs")
    FINAL_OUTPUT_DIR = PARENTS / Path("outputs/final")

    # ==================== Dataset ====================
    IMG_SIZE = 380
    LABEL_COL = "Label"
    IMAGE_COL = "Image Name"
    PATIENT_COL = "Patient"
    QUALITY_COL = "Quality Score"

    # ==================== Training Params ====================
    SEED = 42
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 40
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.4
    EARLY_STOP = 8

    # ==================== Augmentation ====================
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # ==================== Model ====================
    BACKBONE = "efficientnet_b3"
    PRETRAINED = True

    # ==================== Inference ====================
    USE_TTA = True
    TTA_STEPS = 5

    # ==================== Classes ====================
    CLASS_NAMES = ["GON-", "GON+"]
    POS_LABEL = "GON+"

    # ==================== MONAI CacheDataset ====================
    CACHE_RATE = 1.0

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(seed=cfg.SEED)


# ==================== DATA ====================
def load_and_clean_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[DATA] Raw: {len(df)} images, {df[cfg.PATIENT_COL].nunique()} patients")
    print(f"Classes:\n{df[cfg.LABEL_COL].value_counts()}\n")

    df["label"] = (df[cfg.LABEL_COL] == cfg.POS_LABEL).astype(int)
    df["image_path"] = df[cfg.IMAGE_COL].apply(lambda x: str(cfg.IMAGES_DIR / x))

    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        print(f"[WARNING] {len(missing)} image(s) not found. First 3:")
        for p in missing[:3]:
            print(f"  {p}")
    else:
        print(f"[DATA] All {len(df)} image paths verified OK.")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def df_to_monai_list(df: pd.DataFrame) -> List[Dict]:
    return [
        {"image": row["image_path"], "label": int(row["label"])}
        for _, row in df.iterrows()
    ]


def patient_stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pat = (
        df.groupby(cfg.PATIENT_COL)["label"]
        .agg(lambda x: int(x.mean() >= 0.5))
        .reset_index()
    )

    p_tr, p_tmp, _, y_tmp = train_test_split(
        pat[cfg.PATIENT_COL].values,
        pat["label"].values,
        test_size=val_size + test_size,
        stratify=pat["label"].values,
        random_state=cfg.SEED,
    )
    p_val, p_test = train_test_split(
        p_tmp, test_size=0.5, stratify=y_tmp, random_state=cfg.SEED
    )

    tr = df[df[cfg.PATIENT_COL].isin(p_tr)].reset_index(drop=True)
    vl = df[df[cfg.PATIENT_COL].isin(p_val)].reset_index(drop=True)
    te = df[df[cfg.PATIENT_COL].isin(p_test)].reset_index(drop=True)

    for name, split in [("Train", tr), ("Val  ", vl), ("Test ", te)]:
        print(
            f"[SPLIT] {name}: {len(split):4d} images | "
            f"{split[cfg.PATIENT_COL].nunique():3d} patients | "
            f"GON+: {split['label'].sum():3d} | "
            f"GON-: {(split['label'] == 0).sum():3d}"
        )
    return tr, vl, te


# ==================== SELF-DEFINED TRANSFORMS ====================
class FovMaskd(MapTransform):
    """Zero out pixels outside the circular fundus FOV."""

    def __init__(self, keys: KeysCollection, threshold: int = 15):
        super().__init__(keys)
        self.threshold = threshold

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            arr = img.numpy() if hasattr(img, "numpy") else np.array(img)
            C, H, W = arr.shape

            gray = (arr.mean(axis=0) * 255).astype(np.uint8)
            _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            mask = np.zeros((H, W), dtype=np.uint8)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest)
                cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, thickness=-1)
            else:
                mask[:] = 255

            mask_float = (mask / 255.0).astype(np.float32)
            arr = arr * mask_float[np.newaxis, :, :]

            d[key] = torch.from_numpy(arr) if hasattr(img, "numpy") else arr
        return d


class CLAHEd(MapTransform):
    """
    Per-channel CLAHE to boost disc interior contrast.
    cv2.CLAHE is created inside __call__ as a local variable (NOT self.clahe)
    so the object is never stored on the instance and stays picklable
    across DataLoader worker processes.
    """

    def __init__(
        self,
        keys: KeysCollection,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ):
        super().__init__(keys)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, data: dict) -> dict:
        # local variable — never assigned to self, never pickled
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        d = dict(data)
        for key in self.keys:
            img = d[key]
            is_tensor = hasattr(img, "numpy")
            arr = img.numpy() if is_tensor else np.array(img)

            arr_uint8 = (arr * 255).astype(np.uint8)
            result = np.zeros_like(arr_uint8)
            for c in range(arr_uint8.shape[0]):
                result[c] = clahe.apply(arr_uint8[c])
            arr = result.astype(np.float32) / 255.0

            d[key] = torch.from_numpy(arr) if is_tensor else arr
        return d


# ==================== TRANSFORM PIPELINES ====================
def get_transforms(mode: str) -> Compose:
    transform_head = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(
            keys=["image"], spatial_size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode="bicubic"
        ),
        FovMaskd(keys=["image"], threshold=10),
        CLAHEd(keys=["image"], clip_limit=2.0),
    ]

    augment = [
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        RandRotated(
            keys=["image"],
            prob=0.4,
            range_x=0.52,
            range_y=0.0,
            range_z=0.0,
            mode="bilinear",
            padding_mode="reflection",
            keep_size=True,
        ),
        RandZoomd(
            keys=["image"], prob=0.3, min_zoom=0.85, max_zoom=1.15, mode="bilinear"
        ),
        RandAffined(
            keys=["image"],
            prob=0.3,
            translate_range=(0.05, 0.05),
            mode="bilinear",
            padding_mode="reflection",
        ),
        RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)
        ),
    ]

    transform_tail = [
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=[0.485, 0.456, 0.406],
            divisor=[0.229, 0.224, 0.225],
            channel_wise=True,
        ),
        ToTensord(keys=["image", "label"]),
    ]

    if mode == "train":
        return Compose(transform_head + augment + transform_tail)
    else:
        return Compose(transform_head + transform_tail)


def get_tta_transforms() -> list:
    transform_head = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(
            keys=["image"], spatial_size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode="bicubic"
        ),
        FovMaskd(keys=["image"], threshold=10),
        CLAHEd(keys=["image"], clip_limit=2.0),
    ]
    transform_tail = [
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=[0.485, 0.456, 0.406],
            divisor=[0.229, 0.224, 0.225],
            channel_wise=True,
        ),
        ToTensord(keys=["image", "label"]),
    ]
    augment_ls = [
        [],
        [Flipd(keys=["image"], spatial_axis=0)],
        [Flipd(keys=["image"], spatial_axis=1)],
        [Rotate90d(keys=["image"], k=1)],
        [Rotate90d(keys=["image"], k=3)],
    ]
    return [Compose(transform_head + aug + transform_tail) for aug in augment_ls]


# ==================== LOADERS ====================
def make_loader(
    df: pd.DataFrame, transform: Compose, mode: str, sampler=None
) -> DataLoader:
    data_list = df_to_monai_list(df)
    cache = cfg.CACHE_RATE if mode != "train" else 0.0

    dataset = CacheDataset(
        data=data_list,
        transform=transform,
        cache_rate=cache,
        num_workers=cfg.NUM_WORKERS,
        copy_cache=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=(mode == "train" and sampler is None),
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=(mode == "train"),
    )


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    weights = df["label"].map(lambda x: total / counts[x]).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=total,
        replacement=True,
    )


# ==================== MIXUP ====================
def mixup_data(x, y, alpha=cfg.MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya.float()) + (1 - lam) * criterion(pred, yb.float())


# ==================== MODEL ====================
class GlaucomaModel(nn.Module):
    def __init__(
        self,
        backbone: str = cfg.BACKBONE,
        pretrained: bool = cfg.PRETRAINED,
        dropout: float = cfg.DROPOUT,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)).squeeze(1)


# ==================== TRAINING ====================
class EarlyStopping:
    def __init__(self, patience=cfg.EARLY_STOP, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(
    model: GlaucomaModel, loader: DataLoader, optimizer, criterion
) -> Dict:
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Train", leave=False):
        imgs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if cfg.USE_MIXUP:
            imgs, ya, yb, lam = mixup_data(imgs, labels)

        logits = model(imgs)
        loss = (
            mixup_loss(criterion, logits, ya, yb, lam)
            if cfg.USE_MIXUP
            else criterion(logits, labels.float())
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds_bin = (np.array(all_probs) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, preds_bin),
    }


@torch.no_grad()
def evaluate(model: GlaucomaModel, loader: DataLoader, criterion) -> Dict:
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        imgs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels.float())

        total_loss += loss.item() * imgs.size(0)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds_bin = (np.array(all_probs) >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": roc_auc_score(all_labels, all_probs),
        "acc": accuracy_score(all_labels, preds_bin),
        "f1": f1_score(all_labels, preds_bin),
        "preds": np.array(all_probs),
        "labels": np.array(all_labels),
    }


def train(train_df, val_df, fold=None):
    tag = f"fold{fold}" if fold is not None else "model"
    ckpt_path = cfg.OUTPUT_DIR / f"best_{tag}.pt"

    sampler = make_weighted_sampler(train_df)
    train_loader = make_loader(
        train_df, get_transforms("train"), mode="train", sampler=sampler
    )
    val_loader = make_loader(val_df, get_transforms("val"), mode="val")

    model = GlaucomaModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(train_df["label"] == 0).sum() / max((train_df["label"] == 1).sum(), 1)],
            device=DEVICE,
        )
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )
    stopper = EarlyStopping()
    best_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}

    print(f"\n{'=' * 60}\n  {tag.upper()}  —  {cfg.BACKBONE}\n{'=' * 60}")

    for epoch in range(1, cfg.EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion)
        vl = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_auc"].append(tr["auc"])
        history["val_auc"].append(vl["auc"])

        flag = ""
        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- best"

        print(
            f"  Ep {epoch:03d}/{cfg.EPOCHS} "
            f"| Loss {tr['loss']:.4f}/{vl['loss']:.4f} "
            f"| AUC  {tr['auc']:.4f}/{vl['auc']:.4f} "
            f"| Acc  {tr['acc']:.4f}/{vl['acc']:.4f}{flag}"
        )

        if stopper(vl["auc"]):
            print(f"  [EARLY STOP] No gain for {cfg.EARLY_STOP} epochs.")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model, history


# ==================== TTA INFERENCE ====================
@torch.no_grad()
def tta_predict(model: GlaucomaModel, df: pd.DataFrame) -> np.ndarray:
    model.eval()

    transform_head = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            Resized(
                keys=["image"],
                spatial_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                mode="bicubic",
            ),
            FovMaskd(keys=["image"], threshold=10),
            CLAHEd(keys=["image"], clip_limit=2.0),
        ]
    )
    transform_tail = Compose(
        [
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=[0.485, 0.456, 0.406],
                divisor=[0.229, 0.224, 0.225],
                channel_wise=True,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    geom_passes = [
        [],
        [Flipd(keys=["image"], spatial_axis=0)],
        [Flipd(keys=["image"], spatial_axis=1)],
        [Rotate90d(keys=["image"], k=1)],
        [Rotate90d(keys=["image"], k=3)],
    ]
    geom_names = ["original", "flip-V", "flip-H", "rot90", "rot270"]

    head_dataset = CacheDataset(
        data=df_to_monai_list(df),
        transform=transform_head,
        cache_rate=1.0,
        num_workers=cfg.NUM_WORKERS,
        copy_cache=False,
    )
    _ = [head_dataset[i] for i in range(len(head_dataset))]

    all_probs = []
    for geom, name in zip(geom_passes, geom_names):
        pass_tfm = Compose(geom + [transform_tail]) if geom else transform_tail
        pass_probs = []

        for start in tqdm(
            range(0, len(head_dataset), cfg.BATCH_SIZE),
            desc=f"  TTA [{name}]",
            leave=False,
        ):
            end = min(start + cfg.BATCH_SIZE, len(head_dataset))
            imgs = torch.stack(
                [pass_tfm(head_dataset[i])["image"] for i in range(start, end)]
            ).to(DEVICE)

            logits = model(imgs)
            pass_probs.extend(torch.sigmoid(logits).cpu().numpy())

        all_probs.append(np.array(pass_probs))

    return np.mean(all_probs, axis=0)


# ==================== PLOTS ====================
def plot_training_curves(history: Dict, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    for ax, tr_k, vl_k, title, ylabel in [
        (axes[0], "train_loss", "val_loss", "Loss", "Loss"),
        (axes[1], "train_auc", "val_auc", "AUC-ROC", "AUC"),
    ]:
        ax.plot(
            ep,
            history[tr_k],
            "o-",
            label="Train",
            color="#2196F3",
            linewidth=2,
            markersize=4,
        )
        ax.plot(
            ep,
            history[vl_k],
            "o-",
            label="Val",
            color="#F44336",
            linewidth=2,
            markersize=4,
        )

        if "auc" in vl_k:
            best_ep = int(np.argmax(history[vl_k])) + 1
            best_v = max(history[vl_k])
            ax.scatter(
                [best_ep],
                [best_v],
                color="#FF9800",
                s=120,
                zorder=5,
                label=f"Best (ep {best_ep}, {best_v:.4f})",
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves {tag}", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(cfg.OUTPUT_DIR / f"training_curves_{tag}.png")
    plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    tag: str = "test",
):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # row-normalised matrix for colour scale — shows rates not raw counts
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    f1 = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=False,  # we draw custom annotations below
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=cfg.CLASS_NAMES,
        yticklabels=cfg.CLASS_NAMES,
        linewidths=0.5,
        ax=ax,
    )

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.55 else "#1a1a1a"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )

    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=11)
    ax.set_ylabel("True Label", fontweight="bold", fontsize=11)
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=13)

    fig.text(
        0.5,
        -0.04,
        f"F1 (macro) = {f1:.4f}   |   AUC-ROC = {auc:.4f}"
        f"   |   Sensitivity = {sensitivity:.4f}   |   Specificity = {specificity:.4f}",
        ha="center",
        fontsize=9,
        color="#444",
    )

    plt.tight_layout()
    out_path = cfg.OUTPUT_DIR / f"confusion_matrix_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Confusion matrix saved → {out_path}")


def visualise_gradcam(model: GlaucomaModel, df: pd.DataFrame):
    gradcam_dir = cfg.OUTPUT_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    tfm = get_transforms("val")
    target_layer = model.backbone.blocks[-1]

    _activations: Dict = {}
    _gradients: Dict = {}

    def forward_hook(module, input, output):
        _activations["value"] = output.detach()  # (1, C, H', W')

    def backward_hook(module, grad_in, grad_out):
        _gradients["value"] = grad_out[0].detach()  # (1, C, H', W')

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    saved = 0

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Grad-CAM"):
            item = {"image": row["image_path"], "label": int(row["label"])}
            tensor = tfm(item)["image"].unsqueeze(0).to(DEVICE)

            # forward + backward
            model.zero_grad()
            logit = model(tensor)
            logit.backward()

            acts = _activations["value"]  # (1, C, H', W')
            grads = _gradients["value"]  # (1, C, H', W')

            # Grad-CAM weighting
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1).squeeze(0)  # (H', W')
            cam = torch.clamp(cam, min=0)
            cam -= cam.min()
            cam /= cam.max() + 1e-8
            saliency = cam.cpu().numpy()  # guaranteed 2-D

            prob = float(torch.sigmoid(logit).detach().cpu())
            pred_cls = cfg.CLASS_NAMES[int(prob >= 0.5)]
            true_cls = cfg.CLASS_NAMES[int(row["label"])]
            correct = pred_cls == true_cls

            orig = np.array(
                Image.open(row["image_path"])
                .convert("RGB")
                .resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
            )
            heat = cv2.resize(saliency, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)

            title_color = "#66BB6A" if correct else "#EF5350"

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))

            axes[0].imshow(orig)
            axes[0].set_title("Original Fundus", fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title(
                f"Grad-CAM  |  True: {true_cls}  Pred: {pred_cls} ({prob:.2f})",
                fontweight="bold",
                color=title_color,
            )
            axes[1].axis("off")

            plt.suptitle(row[cfg.IMAGE_COL], fontsize=9)
            plt.tight_layout()

            stem = Path(row["image_path"]).stem
            outcome = "correct" if correct else "wrong"
            fname = f"{stem}__{true_cls}__{outcome}.png"

            out_path = gradcam_dir / fname
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            saved += 1

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    print(f"[Grad-CAM] Saved {saved} overlays to {gradcam_dir}/")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    print("=" * 60)
    print("  HYGD GLAUCOMA DETECTION — train.py")
    print(f"  Device : {DEVICE}")
    print(f"  Labels : {cfg.LABELS_CSV}")
    print(f"  Images : {cfg.IMAGES_DIR}")
    print("=" * 60)

    df = load_and_clean_labels(cfg.LABELS_CSV)
    train_df, val_df, test_df = patient_stratified_split(df)

    model, history = train(train_df, val_df)

    torch.save(model.state_dict(), cfg.FINAL_OUTPUT_DIR / "final_model.pt")
    json.dump(history, open(cfg.OUTPUT_DIR / "training_history.json", "w"))

    plot_training_curves(history, tag="final")

    # ──────────── test set evaluation ────────────
    print("\n[INFERENCE] Running TTA on test set...")
    probs = tta_predict(model, test_df)
    labels = test_df["label"].values

    fpr, tpr, thresholds = roc_curve(labels, probs)
    threshold = float(thresholds[np.argmax(tpr - fpr)])
    preds = (probs >= threshold).astype(int)
    print(f"  Optimal threshold : {threshold:.4f}")
    print(f"  AUC-ROC           : {roc_auc_score(labels, probs):.4f}")
    print(f"  F1 (macro)        : {f1_score(labels, preds, average='macro'):.4f}")
    print(f"  Accuracy          : {accuracy_score(labels, preds):.4f}")

    plot_confusion_matrix(labels, preds, probs, tag="final")

    # ──────────── grad-cam for all test images ────────────
    print("\n[GRAD-CAM] Generating saliency maps for all test images...")
    visualise_gradcam(model, test_df)

    print(f"\n[DONE] All outputs saved to {cfg.OUTPUT_DIR}/")
