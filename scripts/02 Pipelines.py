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
        self.fov_mask = self.get_fov_mask(img)
        self.texture_map = self.get_texture_map(img)
        self.vessel_map = self.get_vessel_map(img)
        self.spatial_prior = self.get_spatial_prior(self.fov_mask)
        self.symmetry_map = self.get_symmetry_map(img)
        self.symmetry_map2 = self.get_symmetry_map2(img)
        self.vessel_map2 = self.get_vessel_map2(img)

        self.od_loc, self.heatmap = self.find_od(
            [
                self.texture_map,
                self.vessel_map,
                self.vessel_map2,
                self.symmetry_map,
                self.symmetry_map2,
                self.spatial_prior,
            ]
        )

    @property
    def get_od_loc(self):
        return self.od_loc

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Final OD Probability Map")
        plt.imshow(self.heatmap, cmap="inferno")
        plt.subplot(1, 2, 2)
        img_marked = self.mark()
        plt.title("Detected OD Location")
        plt.imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))

    def mark(self, crop_ratio=0.3):
        out = self.img.copy()
        h, w = out.shape[:2]

        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)

        x1 = max(self.od_loc[0] - crop_w // 2, 0)
        y1 = max(self.od_loc[1] - crop_h // 2, 0)
        x2 = min(self.od_loc[0] + crop_w // 2, w)
        y2 = min(self.od_loc[1] + crop_h // 2, h)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 10)
        cv2.drawMarker(out, self.od_loc, (0, 0, 255), cv2.MARKER_CROSS, 100, 15)

        return out

    def crop_od(self, crop_ratio=0.3):
        img_ori = self.img.copy()

        crop_resize = 512
        sample_height = img_ori.shape[0]
        sample_width = img_ori.shape[1]

        # cropping
        crop_width = int(sample_width * crop_ratio)
        crop_height = int(sample_height * crop_ratio)

        x1 = max(self.od_loc[0] - crop_width // 2, 0)
        y1 = max(self.od_loc[1] - crop_height // 2, 0)
        x2 = min(self.od_loc[0] + crop_width // 2, sample_width)
        y2 = min(self.od_loc[1] + crop_height // 2, sample_height)

        cropped_img = img_ori[y1:y2, x1:x2]
        cropped_resize_img = cv2.resize(cropped_img, (crop_resize, crop_resize))

        return cropped_resize_img

    # ===== GET MASK FUNCTIONS =====
    def get_fov_mask(self, img_input):
        gray = img_input[:, :, 1]  # Green channel (cleanest boundary)
        blurred = cv2.medianBlur(gray, 31)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=1000,  # Assume only ONE big circle exists
            param1=50,
            param2=30,
            minRadius=int(img_input.shape[0] * 0.3),  # At least 30% of image height
            maxRadius=int(img_input.shape[0] * 0.6),
        )
        mask = np.zeros_like(gray)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # x, y
                radius = i[2]
                # Draw the detected circle into a binary mask
                mask = cv2.circle(mask, center, radius, 255, -1)

        # erode fov mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        mask_eroded = cv2.erode(mask, kernel, iterations=3)
        return mask_eroded

    def get_texture_map(self, img_bgr):
        red = img_bgr[:, :, 1]
        red = cv2.medianBlur(red, 15)  # Remove speckle noise

        # Structural Gradient (Dilation - Erosion)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grad = cv2.morphologyEx(red, cv2.MORPH_GRADIENT, kernel)

        # Create a "Hotspot" of density
        blur_size = int(img_bgr.shape[1] * 0.15)  # 15% of width
        if blur_size % 2 == 0:
            blur_size += 1
        texture_density = cv2.GaussianBlur(grad, (blur_size, blur_size), 0)

        return cv2.normalize(
            texture_density, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )

    def get_vessel_map(self, img):
        green = img[:, :, 1]
        vessels = cv2.morphologyEx(
            green,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        vessel_density = cv2.GaussianBlur(vessels, (201, 201), 0)
        vessel_map = cv2.normalize(
            vessel_density, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        return vessel_map

    def get_spatial_prior(self, fov_mask):
        h, w = fov_mask.shape
        M = cv2.moments(fov_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = w // 2, h // 2

        x = np.linspace(0, w, w)
        y = np.linspace(0, h, h)
        X, Y = np.meshgrid(x, y)
        sigma = w * 0.15
        central_valley = 1 - np.exp(-((X - cX) ** 2 + (Y - cY) ** 2) / (2 * sigma**2))
        prior = central_valley * (fov_mask.astype(np.float32) / 255.0)

        return cv2.normalize(prior, None, 0, 1, cv2.NORM_MINMAX)

    def get_symmetry_map(self, img_bgr):
        green = img_bgr[:, :, 1]
        green = cv2.GaussianBlur(green, (7, 7), 0)

        gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Accumulate radial symmetry (simplified for performance)
        cos_map = cv2.GaussianBlur(np.cos(ang) * mag, (101, 101), 0)
        sin_map = cv2.GaussianBlur(np.sin(ang) * mag, (101, 101), 0)
        hub_evidence = cv2.GaussianBlur(mag, (101, 101), 0) - cv2.magnitude(
            cos_map, sin_map
        )

        return cv2.normalize(hub_evidence, None, 0, 1, cv2.NORM_MINMAX)

    def get_symmetry_map2(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        texture_expert = cv2.GaussianBlur(grad, (201, 201), 0)
        texture_expert = cv2.normalize(
            texture_expert.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX
        )
        return texture_expert

    def get_vessel_map2(self, img):
        green = img[:, :, 1]
        # Highlight vessels using Black-Hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        vessels = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

        # Calculate symmetry (Directional clash)
        gx = cv2.Sobel(vessels, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(vessels, cv2.CV_32F, 0, 1, ksize=5)
        mag, ang = cv2.cartToPolar(gx, gy)

        # Vectors cancel out at the hub
        cos_blur = cv2.GaussianBlur(np.cos(ang) * mag, (201, 201), 0)
        sin_blur = cv2.GaussianBlur(np.sin(ang) * mag, (201, 201), 0)
        mag_blur = cv2.GaussianBlur(mag, (201, 201), 0)
        vessel_expert = cv2.normalize(
            mag_blur - cv2.magnitude(cos_blur, sin_blur), None, 0, 1, cv2.NORM_MINMAX
        )
        return vessel_expert

    def find_od(self, map_ls: list):
        # FIND OVERLAPPING MAP
        combined_prob = None
        for map in map_ls:
            if combined_prob is None:
                combined_prob = map
            else:
                combined_prob *= map

        final_map = combined_prob

        # LOCATE THE PEAK
        final_map_8u = cv2.normalize(final_map, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        _, _, _, max_loc = cv2.minMaxLoc(final_map_8u)

        return max_loc, final_map_8u


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
