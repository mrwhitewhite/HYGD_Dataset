import argparse
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

_ml_ready = threading.Event()
_ml_error: str = ""


def _load_ml():
    global _ml_error
    try:
        import torch
        import timm
        import cv2
        import monai

        _ml_ready.set()
    except Exception as e:
        _ml_error = str(e)
        _ml_ready.set()


threading.Thread(target=_load_ml, daemon=True).start()


BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GON_POS = "#f85149"
GON_NEG = "#3fb950"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
TEXT_DIM = "#484f58"

THUMB_SIZE = (300, 300)
CAM_SIZE = (300, 300)
WIN_W, WIN_H = 1080, 800


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_path: str, cache_attr: str):
    import torch
    from train import DEVICE, GlaucomaModel

    cache = getattr(_load_model, cache_attr, None)
    if cache is None or cache[0] != model_path:
        model = GlaucomaModel()
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval().to(DEVICE)
        setattr(_load_model, cache_attr, (model_path, model))
    return getattr(_load_model, cache_attr)[1]


def _compute_gradcam(model, tensor):
    """Return (saliency: np.ndarray H×W in [0,1], prob: float)."""
    import torch

    activations, gradients = {}, {}
    target_layer = model.backbone.blocks[-1]

    fwd = target_layer.register_forward_hook(
        lambda m, i, o: activations.__setitem__("v", o.detach())
    )
    bwd = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.__setitem__("v", go[0].detach())
    )
    try:
        model.zero_grad()
        logit = model(tensor)
        logit.backward()
    finally:
        fwd.remove()
        bwd.remove()

    acts = activations["v"]
    grads = gradients["v"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze(0)
    cam = torch.clamp(cam, min=0)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    prob = float(torch.sigmoid(logit).detach().cpu())
    return cam.cpu().numpy(), prob


def _make_overlay(image_path: str, saliency: np.ndarray, img_size: int) -> np.ndarray:
    import cv2

    orig = np.array(Image.open(image_path).convert("RGB").resize((img_size, img_size)))
    heat = cv2.resize(saliency, (img_size, img_size))
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig, 0.55, heat, 0.45, 0)


def _run_inference(image_path: str, model_path: str) -> dict:
    import torch
    from train import cfg, DEVICE, get_transforms

    model = _load_model(model_path, "_cache_single")
    tfm = get_transforms("val")
    tensor = tfm({"image": image_path, "label": 0})["image"].unsqueeze(0).to(DEVICE)

    saliency, prob = _compute_gradcam(model, tensor)
    pred_cls = cfg.CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1.0 - prob
    overlay = _make_overlay(image_path, saliency, cfg.IMG_SIZE)

    return {
        "prob": prob,
        "pred_cls": pred_cls,
        "confidence": confidence,
        "gradcam_overlay": overlay,
    }


def _run_tta_inference(image_path: str, model_path: str) -> dict:
    import torch
    from monai.data import CacheDataset
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        ScaleIntensityd,
        Resized,
        NormalizeIntensityd,
        ToTensord,
        Flipd,
        Rotate90d,
    )
    from train import cfg, DEVICE, get_transforms

    model = _load_model(model_path, "_cache_tta")

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

    head_ds = CacheDataset(
        data=[{"image": image_path, "label": 0}],
        transform=transform_head,
        cache_rate=1.0,
        num_workers=0,
    )
    _ = head_ds[0]

    all_probs = []
    with torch.no_grad():
        for geom in geom_passes:
            pass_tfm = Compose(geom + [transform_tail]) if geom else transform_tail
            img = pass_tfm(head_ds[0])["image"].unsqueeze(0).to(DEVICE)
            all_probs.append(float(torch.sigmoid(model(img)).cpu()))

    prob = float(np.mean(all_probs))
    pred_cls = cfg.CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1.0 - prob

    tensor = (
        get_transforms("val")({"image": image_path, "label": 0})["image"]
        .unsqueeze(0)
        .to(DEVICE)
    )
    saliency, _ = _compute_gradcam(model, tensor)
    overlay = _make_overlay(image_path, saliency, cfg.IMG_SIZE)

    return {
        "prob": prob,
        "pred_cls": pred_cls,
        "confidence": confidence,
        "tta_probs": all_probs,
        "gradcam_overlay": overlay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────


class GlaucomaApp(tk.Tk):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.image_path = None
        self._photo = None
        self._cam_photo = None
        self._busy = False

        self.title("HYGD Glaucoma Predictor")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._build_fonts()
        self._build_ui()
        self._set_status("Loading ML backend…", TEXT_SEC)
        self.after(100, self._wait_for_ml)

    def _build_fonts(self):
        self.f_title = tkfont.Font(family="Courier", size=14, weight="bold")
        self.f_label = tkfont.Font(family="Courier", size=10)
        self.f_result = tkfont.Font(family="Courier", size=26, weight="bold")
        self.f_sub = tkfont.Font(family="Courier", size=10)
        self.f_log = tkfont.Font(family="Courier New", size=9)
        self.f_btn = tkfont.Font(family="Courier", size=10, weight="bold")

    def _build_ui(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(
            top, text="◈ HYGD GLAUCOMA PREDICTOR", font=self.f_title, bg=BG, fg=ACCENT
        ).pack(side="left")
        self._status_lbl = tk.Label(top, text="", font=self.f_sub, bg=BG, fg=TEXT_SEC)
        self._status_lbl.pack(side="right")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=10)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        # ── left column ───────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")

        tk.Label(left, text="INPUT", font=self.f_label, bg=BG, fg=TEXT_SEC).pack(
            anchor="w"
        )

        self._drop_zone = tk.Canvas(
            left,
            width=THUMB_SIZE[0],
            height=THUMB_SIZE[1],
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            cursor="hand2",
        )
        self._drop_zone.pack()
        self._draw_drop_placeholder()
        self._drop_zone.bind("<Button-1>", self._on_click_drop)
        self._drop_zone.bind("<Enter>", self._on_drop_enter)
        self._drop_zone.bind("<Leave>", self._on_drop_leave)
        try:
            self._setup_dnd()
        except Exception:
            pass

        self._fname_lbl = tk.Label(
            left,
            text="No image loaded",
            font=self.f_sub,
            bg=BG,
            fg=TEXT_DIM,
            wraplength=THUMB_SIZE[0],
            justify="center",
        )
        self._fname_lbl.pack(pady=(4, 8))

        tk.Label(left, text="GRAD-CAM", font=self.f_label, bg=BG, fg=TEXT_SEC).pack(
            anchor="w"
        )

        self._cam_zone = tk.Canvas(
            left,
            width=CAM_SIZE[0],
            height=CAM_SIZE[1],
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self._cam_zone.pack()
        self._draw_cam_placeholder()

        self._cam_lbl = tk.Label(left, text="", font=self.f_sub, bg=BG, fg=TEXT_DIM)
        self._cam_lbl.pack(pady=(4, 0))

        # ── right column ──────────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG, padx=20)
        right.pack(side="left", fill="both", expand=True)

        pred_frame = tk.Frame(
            right, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        pred_frame.pack(fill="x", pady=(0, 10))
        tk.Label(
            pred_frame, text="PREDICTION", font=self.f_label, bg=PANEL, fg=TEXT_SEC
        ).pack(anchor="w", padx=14, pady=(12, 0))
        self._pred_lbl = tk.Label(
            pred_frame, text="—", font=self.f_result, bg=PANEL, fg=TEXT_DIM
        )
        self._pred_lbl.pack(anchor="w", padx=14)
        self._conf_lbl = tk.Label(
            pred_frame, text="", font=self.f_sub, bg=PANEL, fg=TEXT_SEC
        )
        self._conf_lbl.pack(anchor="w", padx=14, pady=(0, 12))

        bar_frame = tk.Frame(
            right, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        bar_frame.pack(fill="x", pady=(0, 10))
        tk.Label(
            bar_frame,
            text="PROBABILITY  GON+ / GON−",
            font=self.f_label,
            bg=PANEL,
            fg=TEXT_SEC,
        ).pack(anchor="w", padx=14, pady=(10, 4))
        self._bar_canvas = tk.Canvas(
            bar_frame, height=22, bg=PANEL, highlightthickness=0
        )
        self._bar_canvas.pack(fill="x", padx=14, pady=(0, 4))
        self._bar_lbl = tk.Label(
            bar_frame, text="", font=self.f_sub, bg=PANEL, fg=TEXT_SEC
        )
        self._bar_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        ctrl = tk.Frame(right, bg=BG)
        ctrl.pack(fill="x", pady=(0, 10))
        self._use_tta = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl,
            text="Use TTA (5-pass, slower)",
            variable=self._use_tta,
            font=self.f_sub,
            bg=BG,
            fg=TEXT_SEC,
            selectcolor=PANEL,
            activebackground=BG,
            activeforeground=TEXT_PRI,
        ).pack(side="left")
        self._run_btn = tk.Button(
            ctrl,
            text="▶  PREDICT",
            font=self.f_btn,
            bg=ACCENT,
            fg=BG,
            activebackground="#79b8ff",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._on_predict,
        )
        self._run_btn.pack(side="right")

        tk.Label(right, text="LOG", font=self.f_label, bg=BG, fg=TEXT_SEC).pack(
            anchor="w"
        )
        log_frame = tk.Frame(
            right, bg=PANEL, highlightthickness=1, highlightbackground=BORDER
        )
        log_frame.pack(fill="both", expand=True)
        self._log = tk.Text(
            log_frame,
            font=self.f_log,
            bg=PANEL,
            fg=TEXT_PRI,
            insertbackground=ACCENT,
            relief="flat",
            wrap="word",
            state="disabled",
            padx=10,
            pady=8,
        )
        self._log.pack(fill="both", expand=True)
        self._log.tag_config("dim", foreground=TEXT_DIM)
        self._log.tag_config("accent", foreground=ACCENT)
        self._log.tag_config("pos", foreground=GON_POS)
        self._log.tag_config("neg", foreground=GON_NEG)
        self._log.tag_config("warn", foreground="#e3b341")

    def _setup_dnd(self):
        import tkinterdnd2  # noqa: F401

        self._drop_zone.drop_target_register("DND_Files")  # type: ignore
        self._drop_zone.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore

    def _on_dnd_drop(self, event):
        self._load_image(event.data.strip().strip("{}"))

    def _draw_drop_placeholder(self):
        self._drop_zone.delete("all")
        cx, cy = THUMB_SIZE[0] // 2, THUMB_SIZE[1] // 2
        pad = 18
        self._drop_zone.create_rectangle(
            pad,
            pad,
            THUMB_SIZE[0] - pad,
            THUMB_SIZE[1] - pad,
            outline=BORDER,
            dash=(6, 4),
            width=1,
        )
        self._drop_zone.create_text(
            cx,
            cy - 30,
            text="⊕",
            fill=TEXT_DIM,
            font=tkfont.Font(family="Courier", size=36),
        )
        self._drop_zone.create_text(
            cx,
            cy + 20,
            text="Drop fundus image here",
            fill=TEXT_DIM,
            font=tkfont.Font(family="Courier", size=10),
        )
        self._drop_zone.create_text(
            cx,
            cy + 40,
            text="or click to browse",
            fill=TEXT_DIM,
            font=tkfont.Font(family="Courier", size=9),
        )

    def _draw_cam_placeholder(self):
        self._cam_zone.delete("all")
        cx, cy = CAM_SIZE[0] // 2, CAM_SIZE[1] // 2
        self._cam_zone.create_text(
            cx,
            cy,
            text="Grad-CAM will appear here\nafter prediction",
            fill=TEXT_DIM,
            font=tkfont.Font(family="Courier", size=9),
            justify="center",
        )

    def _on_drop_enter(self, _):
        self._drop_zone.configure(highlightbackground=ACCENT)

    def _on_drop_leave(self, _):
        self._drop_zone.configure(highlightbackground=BORDER)

    def _on_click_drop(self, _):
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Select fundus image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self.image_path = path
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            canvas = Image.new(
                "RGB",
                THUMB_SIZE,
                tuple(int(PANEL.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)),
            )
            offset = (
                (THUMB_SIZE[0] - img.width) // 2,
                (THUMB_SIZE[1] - img.height) // 2,
            )
            canvas.paste(img, offset)

            self._photo = ImageTk.PhotoImage(canvas)
            self._drop_zone.delete("all")
            self._drop_zone.create_image(0, 0, anchor="nw", image=self._photo)

            self._fname_lbl.config(text=Path(path).name, fg=TEXT_SEC)
            self._log_write(f"Loaded: {path}\n", "accent")

            self._pred_lbl.config(text="—", fg=TEXT_DIM)
            self._conf_lbl.config(text="")
            self._bar_lbl.config(text="")
            self._bar_canvas.delete("all")
            self._draw_cam_placeholder()
            self._cam_lbl.config(text="")

        except Exception as e:
            self._log_write(f"[ERROR] Could not open image: {e}\n", "warn")

    def _on_predict(self):
        if self._busy:
            return
        if not self.image_path:
            self._log_write("[WARN] No image loaded.\n", "warn")
            return
        if not _ml_ready.is_set():
            self._log_write("[WARN] ML backend still loading…\n", "warn")
            return
        if _ml_error:
            self._log_write(f"[ERROR] ML backend failed: {_ml_error}\n", "warn")
            return
        if not Path(self.model_path).exists():
            self._log_write(f"[ERROR] Model not found: {self.model_path}\n", "warn")
            return

        self._busy = True
        self._run_btn.config(state="disabled", text="…")
        self._set_status("Running inference…", ACCENT)

        use_tta = self._use_tta.get()
        path = self.image_path
        mpath = self.model_path

        def _worker():
            try:
                fn = _run_tta_inference if use_tta else _run_inference
                result = fn(path, mpath)
                self.after(0, lambda: self._show_result(result, use_tta))
            except Exception as e:
                self.after(0, lambda e=e: self._show_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_result(self, result: dict, used_tta: bool):
        prob = result["prob"]
        pred_cls = result["pred_cls"]
        conf = result["confidence"]
        color = GON_POS if pred_cls == "GON+" else GON_NEG

        self._pred_lbl.config(text=pred_cls, fg=color)
        self._conf_lbl.config(
            text=f"Confidence  {conf * 100:.1f}%   |   P(GON+) = {prob:.4f}",
            fg=TEXT_SEC,
        )
        self._draw_bar(prob)
        self._show_gradcam(result["gradcam_overlay"], pred_cls, prob)

        sep = "─" * 42
        self._log_write(f"\n{sep}\n", "dim")
        self._log_write(f"  Image   : {Path(self.image_path).name}\n")
        self._log_write(
            f"  Mode    : {'TTA (5-pass)' if used_tta else 'Single pass'}\n"
        )
        self._log_write(f"  P(GON+) : {prob:.6f}\n")
        self._log_write(f"  P(GON-) : {1 - prob:.6f}\n")
        self._log_write("  Pred    : ")
        self._log_write(f"{pred_cls}\n", "pos" if pred_cls == "GON+" else "neg")
        self._log_write(f"  Conf    : {conf * 100:.2f}%\n")

        if used_tta and "tta_probs" in result:
            names = ["orig", "flip-V", "flip-H", "rot90", "rot270"]
            self._log_write("  TTA breakdown:\n", "dim")
            for n, p in zip(names, result["tta_probs"]):
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                self._log_write(f"    {n:<8} {bar} {p:.4f}\n", "dim")

        self._log_write(f"{sep}\n", "dim")
        self._set_status("Done.", GON_NEG)
        self._run_btn.config(state="normal", text="▶  PREDICT")
        self._busy = False

    def _show_gradcam(self, overlay: np.ndarray, pred_cls: str, prob: float):
        img = Image.fromarray(overlay.astype(np.uint8)).resize(CAM_SIZE, Image.LANCZOS)
        self._cam_photo = ImageTk.PhotoImage(img)
        self._cam_zone.delete("all")
        self._cam_zone.create_image(0, 0, anchor="nw", image=self._cam_photo)

        color = GON_POS if pred_cls == "GON+" else GON_NEG
        self._cam_zone.create_rectangle(
            0,
            CAM_SIZE[1] - 22,
            CAM_SIZE[0],
            CAM_SIZE[1],
            fill="#000000",
            outline="",
            stipple="gray50",
        )
        self._cam_zone.create_text(
            CAM_SIZE[0] // 2,
            CAM_SIZE[1] - 11,
            text=f"{pred_cls}  p={prob:.3f}",
            fill=color,
            font=tkfont.Font(family="Courier", size=9, weight="bold"),
        )
        self._cam_lbl.config(text="↑ attention regions", fg=TEXT_DIM)

    def _show_error(self, msg: str):
        self._log_write(f"[ERROR] {msg}\n", "warn")
        self._set_status("Error.", GON_POS)
        self._run_btn.config(state="normal", text="▶  PREDICT")
        self._busy = False

    def _draw_bar(self, prob: float):
        self._bar_canvas.update_idletasks()
        w = self._bar_canvas.winfo_width() or 340
        h = 22
        self._bar_canvas.delete("all")
        self._bar_canvas.create_rectangle(0, 0, w, h, fill=BORDER, outline="")
        pos_w = int(w * prob)
        if pos_w > 0:
            self._bar_canvas.create_rectangle(0, 0, pos_w, h, fill=GON_POS, outline="")
        if w - pos_w > 0:
            self._bar_canvas.create_rectangle(pos_w, 0, w, h, fill=GON_NEG, outline="")
        self._bar_canvas.create_line(w // 2, 0, w // 2, h, fill=BG, width=2)
        self._bar_lbl.config(
            text=f"GON+  {prob * 100:.1f}%  ◂▸  {(1 - prob) * 100:.1f}%  GON−"
        )

    def _log_write(self, text: str, tag: str = None):
        self._log.config(state="normal")
        if tag:
            self._log.insert("end", text, tag)
        else:
            self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _set_status(self, text: str, color: str = TEXT_SEC):
        self._status_lbl.config(text=text, fg=color)

    def _wait_for_ml(self):
        if not _ml_ready.is_set():
            self.after(200, self._wait_for_ml)
            return
        if _ml_error:
            self._set_status(f"ML ERROR: {_ml_error}", GON_POS)
            self._log_write(
                f"[ERROR] ML backend failed to load:\n{_ml_error}\n", "warn"
            )
        else:
            self._set_status("Ready.", GON_NEG)
            self._log_write(
                "ML backend loaded. Drop or click an image to begin.\n", "dim"
            )
            model_p = Path(self.model_path)
            if model_p.exists():
                self._log_write(f"Model  : {self.model_path}\n", "dim")
            else:
                self._log_write(
                    f"[WARN] Model not found at: {self.model_path}\n"
                    "       Pass --model <path> or train first.\n",
                    "warn",
                )


# ======================================= Entry point =======================================
def main():
    parser = argparse.ArgumentParser(description="HYGD Glaucoma Predictor GUI")
    parser.add_argument(
        "--model",
        default="outputs/final/final_model.pt",
        help="Path to trained model .pt file",
    )
    args = parser.parse_args()

    try:
        from tkinterdnd2 import TkinterDnD

        base = TkinterDnD.Tk
    except ImportError:
        base = tk.Tk

    class _App(GlaucomaApp, base):
        pass

    _App(args.model).mainloop()


if __name__ == "__main__":
    main()
