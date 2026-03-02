import cv2
from pyprojroot import here
import polars as pl
import os
from tqdm import tqdm
import numpy as np
from skimage.filters import frangi


# params
path_raw_img = here() / "data" / "raw" / "Images"
path_processed = here() / "data" / "processed"
path_crop_bright = here() / "data" / "processed" / "od_bright"
path_crop_sobel = here() / "data" / "processed" / "od_sobel"
path_crop_vessel = here() / "data" / "processed" / "od_vessel"


# Functions
def crop_od(img, max_loc, crop_ratio=0.3):
    img_ori = img.copy()

    crop_resize = 512
    sample_height = img_ori.shape[0]
    sample_width = img_ori.shape[1]

    # cropping
    crop_width = int(sample_width * crop_ratio)
    crop_height = int(sample_height * crop_ratio)

    x1 = max(max_loc[0] - crop_width // 2, 0)
    y1 = max(max_loc[1] - crop_height // 2, 0)
    x2 = min(max_loc[0] + crop_width // 2, sample_width)
    y2 = min(max_loc[1] + crop_height // 2, sample_height)

    cropped_img = img_ori[y1:y2, x1:x2]
    cropped_resize_img = cv2.resize(cropped_img, (crop_resize, crop_resize))

    return cropped_resize_img


def od_bright(img):
    img_ori = img.copy()

    kernel_clean_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )  # remove random pixel
    kernel_erode_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (51, 51)
    )  # erode fov region to avoid halo
    erode_iter = 10
    fov_thres_val = 20
    gaussian_blur_kernel: tuple = (71, 71)

    # get fov mask
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    _, fov_mask = cv2.threshold(img_gray, fov_thres_val, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_ERODE, kernel_clean_fov_mask)
    fov_mask = cv2.erode(fov_mask, kernel_erode_fov_mask, iterations=erode_iter)

    # find brightest spot
    img_red = img_ori[:, :, 2]
    img_blur = cv2.GaussianBlur(img_red, gaussian_blur_kernel, 0)
    masked_img = cv2.bitwise_and(img_blur, img_blur, mask=fov_mask)
    _, _, _, max_loc = cv2.minMaxLoc(masked_img)
    return max_loc


def od_sobel(img):
    img_ori = img.copy()

    kernel_clean_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )  # remove random pixel
    kernel_erode_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (51, 51)
    )  # erode fov region to avoid halo
    erode_iter = 10
    fov_thres_val = 20

    sobel_ksize = 7
    gaussian_blur_kernel: tuple = (51, 51)

    # get fov mask
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    _, fov_mask = cv2.threshold(img_gray, fov_thres_val, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_ERODE, kernel_clean_fov_mask)
    fov_mask = cv2.erode(fov_mask, kernel_erode_fov_mask, iterations=erode_iter)

    # sobel detection
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize),
        cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize),
    )
    grad_blur = cv2.GaussianBlur(np.abs(grad), gaussian_blur_kernel, 0)
    _, _, _, loc_texture = cv2.minMaxLoc(
        cv2.bitwise_and(np.abs(grad_blur), np.abs(grad_blur), mask=fov_mask)
    )
    return loc_texture


def od_vessel(img):
    img_ori = img.copy()
    kernel_clean_fov_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_erode_fov_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    erode_iter = 1
    fov_thres_val = 20

    gaussian_blur_kernel: tuple = (51, 51)
    green = img_ori[:, :, 1]

    vessels = frangi(
        green,
        sigmas=range(2, 30, 4),
        alpha=0.7,
        beta=0.5,
        gamma=0.05,
        black_ridges=True,
    )

    # get fov mask
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    _, fov_mask = cv2.threshold(img_gray, fov_thres_val, 255, cv2.THRESH_BINARY)

    # clean fov mask
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_ERODE, kernel_clean_fov_mask)
    fov_mask = cv2.erode(fov_mask, kernel_erode_fov_mask, iterations=erode_iter)
    vessels_img = cv2.normalize(vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vessel_density = cv2.GaussianBlur(vessels_img, gaussian_blur_kernel, 0)
    masked_vessels = cv2.bitwise_and(vessel_density, vessel_density, mask=fov_mask)
    _, _, _, loc_vessels = cv2.minMaxLoc(masked_vessels)

    return loc_vessels


# params
path_raw_img = here() / "data" / "raw" / "Images"
path_processed = here() / "data" / "processed"
path_crop_bright = here() / "data" / "processed" / "od_bright"
path_crop_sobel = here() / "data" / "processed" / "od_sobel"
path_crop_vessel = here() / "data" / "processed" / "od_vessel"

df = pl.read_csv(path_processed / "labels.csv")
img_ls = df.select("Image Name").to_numpy().flatten().tolist()

# create dir, if exist then ignore the error
try:
    os.mkdir(path_crop_bright)
except FileExistsError:
    pass
try:
    os.mkdir(path_crop_sobel)
except FileExistsError:
    pass
try:
    os.mkdir(path_crop_vessel)
except FileExistsError:
    pass


loc_bright_ls = []
loc_sobel_ls = []
loc_vessel_ls = []

for i in tqdm(img_ls):
    current_img_path = path_raw_img / i
    current_img = cv2.imread(current_img_path)

    # find OD loc
    loc_bright = od_bright(current_img)
    loc_sobel = od_sobel(current_img)
    loc_vessel = od_vessel(current_img)

    # crop
    cropped_bright = crop_od(current_img, loc_bright)
    cropped_sobel = crop_od(current_img, loc_sobel)
    cropped_vessel = crop_od(current_img, loc_vessel)

    # save
    cv2.imwrite(str(path_crop_bright / i), cropped_bright)
    cv2.imwrite(str(path_crop_sobel / i), cropped_sobel)
    cv2.imwrite(str(path_crop_vessel / i), cropped_vessel)

    # append to list for df
    loc_bright_ls.append(str(loc_bright))
    loc_sobel_ls.append(str(loc_sobel))
    loc_vessel_ls.append(str(loc_vessel))

# df
df = df.with_columns(
    pl.Series("OD_loc_bright", loc_bright_ls),
    pl.Series("OD_loc_sobel", loc_sobel_ls),
    pl.Series("OD_loc_vessel", loc_vessel_ls),
)
df.write_csv(path_processed / "labels_od.csv")
