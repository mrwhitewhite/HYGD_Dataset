from pyprojroot import here
import polars as pl
from sklearn.model_selection import train_test_split
import cv2

# params
path_raw = here() / "data" / "raw"
path_img = path_raw / "Images"
path_processed = here() / "data" / "processed"
test_size = 0.2
seed = 123

# load
df = pl.read_csv(path_raw / "Labels.csv").drop("")

# train test split without data leakage
patient_unique_id = df.select(pl.col("Patient").unique()).to_numpy().flatten()
train_id, test_id = train_test_split(
    patient_unique_id, test_size=test_size, random_state=seed
)
df = df.with_columns(
    pl.when(pl.col("Patient").is_in(test_id))
    .then(pl.lit("TEST"))
    .otherwise(pl.lit("TRAIN"))
    .alias("SPLIT")
)

# img resolution feature
img_name_ls = df.select("Image Name").to_numpy().flatten().tolist()
res_ls = [cv2.imread(path_img / i).shape for i in img_name_ls]
height_ls = [i[0] for i in res_ls]
width_ls = [i[1] for i in res_ls]
df = df.with_columns(
    pl.Series("Height", height_ls),
    pl.Series("Width", width_ls),
)

# save
df.write_csv(path_processed / "labels.csv")
print("01 Data preparation: Success!")
