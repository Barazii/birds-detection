import cv2
import pandas as pd
import os
from pathlib import Path


SAMPLE_CLASSES = [17, 36, 47, 68, 73]
IM2REC_SSD_COLS = [
    "header_cols",
    "label_width",
    "zero_based_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "image_file_name",
]


def split_to_train_test(df, label_column, train_frac=0.8):
    # stratified split
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)
    return train_df, test_df


def processing(pc_base_dir):
    # find the images sizes
    directory = pc_base_dir / "dataset" / "images.txt"
    images_df = pd.read_csv(
        directory, sep=" ", names=["id", "image_file_name"], header=None
    ).dropna(axis=0)
    images_sizes = []
    for i, ifn in zip(images_df["id"], images_df["image_file_name"]):
        img = cv2.imread("dataset/images" + i)
        height, width, _ = img.shape
        image_size = {"id": id, "width": width, "height": height}
        images_sizes.append(image_size)
    sizes_df = pd.DataFrame(images_sizes)

    # create the lst files
    directory = pc_base_dir / "dataset" / "bounding_boxes.txt"
    bboxes_df = pd.read_csv(
        directory,
        sep=" ",
        names=["id", "x_abs", "y_abs", "bbox_width", "bbox_height"],
        header=None,
    )
    directory = pc_base_dir / "dataset" / "train_test_split.txt"
    split_df = pd.read_csv(
        directory,
        sep=" ",
        names=["id", "is_training_image"],
        header=None,
    )
    directory = pc_base_dir / "dataset" / "image_class_labels.txt"
    image_class_labels_df = pd.read_csv(
        directory, sep=" ", names=["id", "class_id"], header=None
    )

    # merge all the metadata into one dataframe
    images_df.reset_index(inplace=True)
    full_df = pd.merge(images_df, image_class_labels_df, on="id")
    full_df = pd.merge(full_df, sizes_df, on="id")
    full_df = pd.merge(full_df, bboxes_df, on="id")
    full_df = pd.merge(full_df, split_df, on="id")
    full_df.sort_values(by=["index"], inplace=True)

    # Define the bounding boxes in the format required by SageMaker's built in Object Detection algorithm.
    # the xmin/ymin/xmax/ymax parameters are specified as ratios to the total image pixel size
    full_df["header_cols"] = (
        2  # one col for the number of header cols, one for the label width
    )
    full_df["label_width"] = (
        5  # number of cols for each label: class, xmin, ymin, xmax, ymax
    )
    full_df["xmin"] = full_df["x_abs"] / full_df["width"]
    full_df["xmax"] = (full_df["x_abs"] + full_df["bbox_width"]) / full_df["width"]
    full_df["ymin"] = full_df["y_abs"] / full_df["height"]
    full_df["ymax"] = (full_df["y_abs"] + full_df["bbox_height"]) / full_df["height"]

    if bool(os.environ["SAMPLE_ONLY"]):
        # small subset of species to reduce resources consumption
        criteria = full_df["class_id"].isin(SAMPLE_CLASSES)
        full_df = full_df[criteria]

    # object detection class id's must be zero based (1 is 0, and 200 is 199).
    unique_classes = full_df["class_id"].drop_duplicates()
    sorted_unique_classes = sorted(unique_classes)
    id_to_zero = {c: i for i, c in enumerate(sorted_unique_classes)}

    full_df["zero_based_id"] = full_df["class_id"].map(id_to_zero)
    full_df.reset_index(inplace=True)

    # use 4 decimal places, as it seems to be required by the Object Detection algorithm
    pd.set_option("display.precision", 4)

    train_dir = pc_base_dir / "train" / "birds_ssd_train.lst"
    val_dir = pc_base_dir / "validation" / "birds_ssd_val.lst"

    if bool(os.environ["RANDOM_SPLIT"]):
        # split into training and validation sets
        train_df, val_df = split_to_train_test(
            full_df, "class_id", float(os.environ["TRAIN_RATIO"])
        )
        train_df[IM2REC_SSD_COLS].to_csv(
            train_dir, sep="\t", float_format="%.4f", header=None
        )
        val_df[IM2REC_SSD_COLS].to_csv(
            val_dir, sep="\t", float_format="%.4f", header=None
        )

    else:
        train_df = full_df[(full_df.is_training_image == 1)]
        train_df[IM2REC_SSD_COLS].to_csv(
            train_dir, sep="\t", float_format="%.4f", header=None
        )
        val_df = full_df[(full_df.is_training_image == 0)]
        val_df[IM2REC_SSD_COLS].to_csv(
            val_dir, sep="\t", float_format="%.4f", header=None
        )


if __name__ == "__main__":
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    processing(pc_base_dir)
