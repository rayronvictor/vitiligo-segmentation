from pathlib import Path
import pandas as pd
from collections import Counter
import random
import yaml
from sklearn.model_selection import KFold
import datetime
import shutil
from tqdm import tqdm


def split_dataset(dataset_path, ksplit=5):
    # Proceed to retrieve all label files for your dataset.
    # dataset_path = Path("datasets/vitiligo-poc") # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

    # Now, read the contents of the dataset YAML file and extract the indices of the class labels.
    # yaml_file = "datasets/vitiligo-poc/vitiligo-poc.yaml" # your data YAML with data directories and names dictionary
    yaml_file = dataset_path / "dataset.yaml"  # your data YAML with data directories and names dictionary
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    # Initialize an empty pandas DataFrame.
    index = [label.stem for label in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)


    # Count the instances of each class-label present in the annotation files.
    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

    #K-Fold Dataset Split

    # Now we will use the KFold class from sklearn.model_selection to generate k splits of the dataset.
    random.seed(0)  # for reproducibility
    # ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    # The dataset has now been split into k folds, each having a list of train and val indices. We will construct a DataFrame to display these results more clearly.

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"

    # Now we will calculate the distribution of class labels for each fold as a ratio of the classes present in val to those present in train.

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    # Next, we create the directories and dataset YAML files for each split.

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    # Create the necessary directories and dataset YAML files
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    return ds_yamls