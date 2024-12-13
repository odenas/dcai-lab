
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from collections import Counter
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, InitVar

from typing import Optional, Tuple, Set

log = logging.getLogger()
ads_id = "62340e30573f02a7303dd8b6"

keep_cols = [
    'angle_to_row',  # DONE
    'cloud_cover',   # DONE
    'crop_damage',
    'crop_health',
    'crop_height',   # DONE
    'crop_name',     # DONE
    'crop_residue',  # DONE
    'crop_residue_type',
    'row_spacing',   # DONE
    # 'soil_color',  # Bad results/data
    'soil_moisture',
    'tillage_practice',  # DONE
    # 'weed_pressure',
    'weeds',
]


def _get_box(img_width, img_height, width, height):
    """
    Calculate the bounding box of the center square of size (width, height) of
    an image of size (img_width, img_height).

    Args:
        img_width: The width of the input image.
        img_height: The height of the input image.
        width: The width of the output square.
        height: The height of the output square.

    Returns:
        A tuple (left, top, right, bottom) representing the coordinates of the
        bounding box.
    """
    left = (img_width - width) / 2
    top = (img_height - height) / 2
    right = (img_width + width) / 2
    bottom = (img_height + height) / 2
    return left, top, right, bottom


def _save_cropped_image(in_fname, out_fname, size):
    img = Image.open(in_fname)
    img_width, img_height = img.size
    cropped_img = img.crop(_get_box(img_width, img_height, size, size))
    cropped_img.save(out_fname)


def _touch(in_fname, out_fname, size):
    with open(out_fname, 'w') as f:
        f.write("hi")


def _make_paths(in_fname, out_fname, size, dry):
    if not out_fname.exists():
        if dry:
            _touch(in_fname, out_fname, size)
        else:
            _save_cropped_image(in_fname, out_fname, size)

    if not Path(in_fname).is_file() or not Path(out_fname).is_file():
        raise ValueError(f"{in_fname} -> {out_fname}")


def load_annotations_csv(alth_dset_path):
    cols = keep_cols + ["artifact_nrg_0_save_path"]
    df = pd.read_csv(alth_dset_path).set_index("id")[cols].drop_duplicates()
    return df


def filter_by_class_names(
        keep_class_names: Set[str],
        df: pd.DataFrame,
        colname: str) -> pd.DataFrame:

    results = []
    for class_name, class_df in df.groupby(colname):
        if class_name in keep_class_names:
            results.append(class_df)
    return pd.concat(results)


@dataclass
class ImageFolderDataset:
    class_name: str
    sample: int
    train_size: int
    raw_df: InitVar[pd.DataFrame]
    ifd_base_dir: Path
    artifact_prefix_path: str = "./"
    keep_cols: Tuple = tuple(keep_cols)
    crop_size: int = 500
    keep_levels: Optional[Tuple[str]] = None
    artifact_save_path_col: str = "artifact_nrg_0_save_path"
    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    def __post_init__(self, raw_df):
        raw_df = raw_df[raw_df[self.class_name].notna()]
        log.info("Kept %d rows after removing nans from %s",
                 len(raw_df), self.class_name)
        if self.keep_levels is not None:
            raw_df = filter_by_class_names(set(self.keep_levels),
                                           raw_df, self.class_name)
            classes_in_raw_df = set(raw_df[self.class_name].values)
            log.info("Data has %s classes after class filtering",
                     str(classes_in_raw_df))
            if not classes_in_raw_df <= set(self.keep_levels):
                raise ValueError(f"Expected {self.keep_levels} but got "
                                 f"{classes_in_raw_df}")
            log.info("Kept %d rows with levels %s.",
                     len(raw_df), str(self.keep_levels))

        df, _ = train_test_split(raw_df,
                                 train_size=self.sample,
                                 stratify=raw_df[self.class_name],
                                 random_state=3211)

        # split it
        self.train_df, self.test_df = train_test_split(
            df, train_size=self.train_size, stratify=df[self.class_name]
        )

    def make_ifd_(self, dry=False):
        self._make_split_ifd(self.train_df, "train", dry)
        self._make_split_ifd(self.test_df, "test", dry)

    def _make_split_ifd(self, df, split, dry):
        md = []
        split_dir = Path(self.ifd_base_dir, split)
        split_dir.mkdir(exist_ok=True, parents=True)

        pbar = tqdm(range(df.shape[0]), desc=f"Processing split: {split}")
        for class_name, class_df in df.groupby(self.class_name):
            class_base_dir = split_dir / class_name
            class_base_dir.mkdir(exist_ok=True, parents=True)

            for image_id, row in class_df.iterrows():
                in_fname = Path(self.artifact_prefix_path,
                                row[self.artifact_save_path_col])
                out_fname = Path(class_base_dir,
                                 in_fname.name.replace("artifact_nrg_0_", ""))
                if image_id != out_fname.stem:
                    raise ValueError(f"{image_id} != {in_fname.stem}")
                _make_paths(in_fname, out_fname, self.crop_size, dry)
                md.append(
                    (image_id, out_fname.relative_to(split_dir))
                )
                pbar.update(1)

        (pd.merge(pd.DataFrame(md, columns=["id", "file_name"]),
                  df[keep_cols].reset_index(), on="id")
         .rename(columns={self.class_name: 'labels'})
         .to_csv(split_dir / "metadata.csv", index=False))

    def strata_count(self, split):
        df = {"train": self.train_df, "test": self.test_df}[split]
        return Counter(df[self.class_name])
