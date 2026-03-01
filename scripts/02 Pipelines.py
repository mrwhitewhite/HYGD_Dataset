import cv2
from pyprojroot import here
import polars as pl
import os
from tqdm import tqdm

# params
path_data = here() / "data"
path_processed = path_data / "processed"
path_raw = path_data / "raw"
path_crop_out = path_processed / "od_images"
path_raw_img = path_raw / "Images"


def crop_od(path: str):
    # load
    img_ori = cv2.imread(path)

    # params
    kernel_clean_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )  # remove random pixel
    kernel_erode_fov_mask = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (51, 51)
    )  # erode fov region to avoid halo
    erode_iter = 10
    fov_thres_val = 20
    gaussian_blur_kernel: tuple = (71, 71)
    crop_ratio = 0.3
    crop_resize = 512
    sample_height = img_ori.shape[0]
    sample_width = img_ori.shape[1]

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


# cropping
df = pl.read_csv(path_processed / "labels.csv")
img_ls = df.select("Image Name").to_numpy().flatten().tolist()
try:
    os.mkdir(path_crop_out)
except FileExistsError:
    pass

for i in tqdm(img_ls):
    img_path = path_raw_img / i
    cropped_img = crop_od(img_path)
    save_path = path_crop_out / i
    cv2.imwrite(str(save_path), cropped_img)
