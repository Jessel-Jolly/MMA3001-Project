"""
Dataset inspection and class instance counting for packaging defect detection.
"""

import glob
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Standard class definitions for pork rasher packaging defects
CLASS_NAMES_DEFAULT = [
    "Loose Meat",
    "Packaging Error",
    "Twisted Meat",
    "Unsealed",
    "Wrinkle",
]
NUMBERS_TO_CHECK_DEFAULT = [0, 1, 2, 3, 4]
SPLITS_TO_CHECK_DEFAULT = ["train", "valid", "test", "*"]


def get_instances(
    numbers: List[int] = NUMBERS_TO_CHECK_DEFAULT,
    file_split: str = "train",
    dataset_dir: Union[str, Path] = "Pork Rasher Error-Packaging-.v4i.yolov11",
    verbose: bool = True
) -> Dict[str, Union[str, int, np.ndarray, List[str]]]:
    """
    Counts defect instances and affected image counts in a given dataset split.

    Parameters
    ----------
    numbers : List[int], default=[0, 1, 2, 3, 4]
        List of integer class IDs to check.
    file_split : str, default="train"
        Subdirectory split to inspect ('train', 'valid', 'test', or '*' for all).
    dataset_dir : Union[str, Path], default="Pork Rasher Error-Packaging-.v4i.yolov11"
        Root folder of the YOLO-formatted dataset.
    verbose : bool, default=True
        Whether to print the formatted summary table to stdout.

    Returns
    -------
    Dict[str, Union[str, int, np.ndarray, List[str]]]
        Dictionary containing:
        - 'split': Name of split inspected
        - 'total_images': Total number of label files found
        - 'instance_counts': Array of total defect instances per class ID (+ undamaged)
        - 'image_counts': Array of number of images containing each class ID (+ undamaged)
        - 'labels': List of readable label names
    """
    labels = [
        "Loose Meat:\t\t",
        "Packaging Error:\t",
        "Twisted Meat:\t\t",
        "Unsealed:\t\t",
        "Wrinkle:\t\t",
        "Undamaged:\t\t",
    ]

    search_path = os.path.join(str(dataset_dir), file_split, "labels", "*")
    names = sorted(glob.glob(search_path))

    count = np.zeros(len(numbers) + 1, dtype=int)
    images = np.zeros(len(numbers) + 1, dtype=int)
    numbers_str = np.array(numbers, dtype=str)

    for txt in names:
        if not os.path.isfile(txt):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                data = np.loadtxt(txt, dtype=str).flatten()
            except Exception:
                data = np.array([], dtype=str)

            if data.size == 0:
                count[-1] += 1
                images[-1] += 1
            else:
                for index, number in enumerate(numbers_str):
                    add = 1
                    for i in data:
                        if i == number:
                            count[index] += 1
                            images[index] += add
                            add = 0

    if verbose:
        print("\n================================================================")
        print(f"\t\t\tFile: {file_split}")
        print(f"\t\t\tTotal Images: {len(names)}\n")
        print("   Type\t\t\t   Total Instances\t  Number of Images")
        for index, number in enumerate(numbers_str):
            idx = int(number)
            label_text = labels[idx] if idx < len(labels) else f"Class {idx}:\t\t"
            print(f"{label_text}\t{count[index]}\t\t\t{images[index]}")
        print(f"{labels[-1]}\t{count[-1]}\t\t\t{images[-1]}")
        print("================================================================\n")

    return {
        "split": file_split,
        "total_images": len(names),
        "instance_counts": count,
        "image_counts": images,
        "labels": labels,
    }


def summarize_dataset(
    dataset_dir: Union[str, Path] = "Pork Rasher Error-Packaging-.v4i.yolov11",
    splits: Optional[List[str]] = None,
    numbers: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Runs instance and image counts across multiple dataset splits.

    Parameters
    ----------
    dataset_dir : Union[str, Path], default="Pork Rasher Error-Packaging-.v4i.yolov11"
        Path to the root dataset directory.
    splits : Optional[List[str]], default=None
        List of splits to check. Defaults to ['train', 'valid', 'test', '*'].
    numbers : Optional[List[int]], default=None
        Class IDs to tally. Defaults to [0, 1, 2, 3, 4].
    verbose : bool, default=True
        Whether to print summary tables.

    Returns
    -------
    Dict[str, Dict]
        Mapping from split name to its detailed count summary dictionary.
    """
    if splits is None:
        splits = SPLITS_TO_CHECK_DEFAULT
    if numbers is None:
        numbers = NUMBERS_TO_CHECK_DEFAULT

    summaries = {}
    for f in splits:
        summaries[f] = get_instances(numbers=numbers, file_split=f, dataset_dir=dataset_dir, verbose=verbose)
    return summaries


if __name__ == "__main__":
    summarize_dataset()
