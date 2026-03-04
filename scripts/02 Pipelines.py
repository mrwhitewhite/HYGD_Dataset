import cv2
from pyprojroot import here
import polars as pl
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# params
path_raw_img = here() / "data" / "raw" / "Images"
path_processed = here() / "data" / "processed"
path_processed_crop = here() / "data" / "processed" / "od"


# Functions
class OD:
    def __init__(self, img):
        self.img = img
        # 1. Generate specialized maps
        self.fov_mask = self.get_fov_mask(self.img)
        self.texture_map = self.get_texture_map(self.img)
        self.vessel_map = self.get_vessel_map(self.img)

        # 2. Perform fusion and peak detection
        self.od_loc, self.heatmap = self.find_od(
            [self.fov_mask, self.texture_map, self.vessel_map]
        )

    @property
    def get_od_loc(self):
        return self.od_loc

    # ===== MAPPING FUNCTIONS =====
    def get_fov_mask(self, img_ori):
        """Generates a mask that suppresses the periphery (Halo Sieve)."""
        h, w = img_ori.shape[:2]
        green = img_ori[:, :, 1]
        blurred_fov = cv2.medianBlur(green, 31)

        # Hough Circle Detection for FOV
        circles = cv2.HoughCircles(
            blurred_fov,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=1000,
            param1=50,
            param2=30,
            minRadius=int(h * 0.3),
            maxRadius=int(h * 0.6),
        )

        fov_mask = np.zeros_like(green)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0, 0]
            cv2.circle(fov_mask, (i[0], i[1]), i[2], 255, -1)
        else:
            _, fov_mask = cv2.threshold(blurred_fov, 10, 255, cv2.THRESH_BINARY)

        # Large erosion to kill crescent halos at the border
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (201, 201))
        halo_sieve = cv2.erode(fov_mask, kernel_erode, iterations=1)
        return halo_sieve.astype(np.float32) / 255.0

    def get_texture_map(self, img_ori):
        """Uses the Red channel to find high-density gradient hubs."""
        h, w = img_ori.shape[:2]
        red = img_ori[:, :, 2]  # Better contrast for OD rim
        red = cv2.medianBlur(red, 15)

        grad_red = cv2.morphologyEx(
            red,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )

        blur_tex = int(w * 0.15) | 1
        tex_hub = cv2.GaussianBlur(grad_red, (blur_tex, blur_tex), 0)

        return cv2.normalize(tex_hub, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    def get_vessel_map(self, img_bgr):
        """Detects the major vascular hub via sparse vessel extraction."""
        w = img_bgr.shape[1]
        k_size = int(w * 0.02) | 1
        green = img_bgr[:, :, 1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        vessels = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

        # Threshold to keep only major vessels, removing background noise
        _, vessels_clean = cv2.threshold(vessels, 15, 255, cv2.THRESH_BINARY)
        vessel_hub = cv2.GaussianBlur(vessels_clean.astype(np.float32), (251, 251), 0)

        return cv2.normalize(vessel_hub, None, 0, 1, cv2.NORM_MINMAX)

    # ===== FUSION LOGIC =====

    def find_od(self, map_ls):
        """Fuses maps using logical multiplication and squares the result for peak clarity."""
        combined_prob = None
        for m in map_ls:
            if combined_prob is None:
                combined_prob = m
            else:
                combined_prob *= m

        # Squaring the map pushes 'weak' spots toward zero and 'sharpens' the peak
        final_map = combined_prob**2

        final_map_8u = cv2.normalize(final_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, _, _, max_loc = cv2.minMaxLoc(final_map_8u)

        return max_loc, final_map_8u

    # ===== VISUALIZATION =====

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Final OD Probability Map")
        plt.imshow(self.heatmap, cmap="inferno")
        plt.subplot(1, 2, 2)
        plt.title("Detected OD Location")
        plt.imshow(cv2.cvtColor(self.mark(), cv2.COLOR_BGR2RGB))
        plt.show()

    def mark(self, crop_ratio=0.3):
        out = self.img.copy()
        h, w = out.shape[:2]
        cv2.drawMarker(out, self.od_loc, (0, 0, 255), cv2.MARKER_CROSS, 100, 15)

        # Optional: Add a rectangle around the detection
        rw, rh = int(w * crop_ratio), int(h * crop_ratio)
        x1, y1 = max(self.od_loc[0] - rw // 2, 0), max(self.od_loc[1] - rh // 2, 0)
        x2, y2 = min(self.od_loc[0] + rw // 2, w), min(self.od_loc[1] + rh // 2, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 10)
        return out

    def crop_od(self, crop_ratio=0.3, size=512):
        h, w = self.img.shape[:2]
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)

        x1, y1 = max(self.od_loc[0] - cw // 2, 0), max(self.od_loc[1] - ch // 2, 0)
        x2, y2 = min(self.od_loc[0] + cw // 2, w), min(self.od_loc[1] + ch // 2, h)

        cropped = self.img[y1:y2, x1:x2]
        return cv2.resize(cropped, (size, size))


df = pl.read_csv(path_processed / "labels.csv")
img_ls = df.select("Image Name").to_numpy().flatten().tolist()

# create dir, if exist then ignore the error
try:
    os.mkdir(path_processed_crop)
except FileExistsError:
    pass

# START PROCESSING
loc_ls = []
for i in tqdm(img_ls):
    current_img_path = path_raw_img / i
    current_img = cv2.imread(current_img_path)
    od = OD(current_img)
    cropped_od = od.crop_od()
    loc_ls.append(str(od.get_od_loc))
    cv2.imwrite(str(path_processed_crop / i), cropped_od)


df = df.with_columns(pl.Series("OD_loc", loc_ls))
df.write_csv(path_processed / "labels_od.csv")
