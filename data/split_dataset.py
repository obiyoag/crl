import os
import json
import pandas as pd
from pathlib import Path

data_dir = Path("/workspace/Datasets/fitzpatrick17k")
data_df = pd.read_csv(data_dir / "meta_data/fitzpatrick17k.csv")
concept_df = pd.read_csv(data_dir / "meta_data/skincon.csv", index_col=0)

# filter out the samples which can not be downloaded
image_names = os.listdir(data_dir / "raw_data")
data_df["md5hash"] = data_df["md5hash"] + ".jpg"
data_df = data_df.rename(columns={"md5hash": "ImageID"})
data_df = data_df[data_df["ImageID"].isin(image_names)]
# filter out non-neoplastic samples
data_df = data_df[data_df["three_partition_label"] != "non-neoplastic"]
data_df["label"] = (
    data_df["three_partition_label"].map({"benign": 0, "malignant": 1}).astype(int)
)

# filter out images with poor quality
concept_df = concept_df[concept_df["Do not consider this image"] != 1]
concept_df = concept_df.drop("Do not consider this image", axis=1)

# merge task label in the concept dataframe
df = pd.merge(data_df[["ImageID", "label"]], concept_df, on="ImageID", how="inner")
df = df.iloc[:, :50]
concept_list = df.columns[2:]


labels = df["label"].unique()
cv_splits = {f"fold_{i}": {"train": [], "valid": []} for i in range(5)}
label_groups = {label: df[df["label"] == label] for label in labels}
for label, group in label_groups.items():
    fold_idx = 0
    for _, row in group.iterrows():
        sample_info = {
            "id": row["ImageID"],
            "label": row["label"],
            "concepts": [row[concept] for concept in concept_list],  # 存为 list
        }
        cv_splits[f"fold_{fold_idx}"]["valid"].append(sample_info)
        fold_idx = (fold_idx + 1) % 5

all_samples = {
    row["ImageID"]: {
        "id": row["ImageID"],
        "label": row["label"],
        "concepts": [row[c] for c in concept_list],
    }
    for _, row in df.iterrows()
}

for fold in range(5):
    val_ids = {sample["id"] for sample in cv_splits[f"fold_{fold}"]["valid"]}
    cv_splits[f"fold_{fold}"]["train"] = [
        all_samples[i] for i in all_samples if i not in val_ids
    ]

with open(data_dir / "meta_data/5_fold_splits.json", "w") as f:
    json.dump(cv_splits, f, indent=4)
