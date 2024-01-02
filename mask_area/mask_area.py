"""
mask_area.py

Description:

    This script generates CSV files, and accompanying plots, of data extracted from .jpg masks.
    It assumes the following folder structure (note: Any name can be used for any directory, but the .jpg name has information and needs to be named like that):
        - mask_area.py
        - Data/
            - Experiment 1/
                - Before/
                    - a1_mask1.jpg                  <- naming is {grid}_mask{mask_number}.jpg
                    - b2_mask1.jpg
                - After/
                - experiment1_after_full_sort.csv   <- It will produce a 2 CSV files per subdirectory in each Experiment
                - experiment1_after.csv             <- ("Before" and "After" in this example)
                - experiment1_before_full_sort.csv  <- one sorted by "Grid" and "Area"
                -experiment1_before.csv             <- and one sorted for just "Area" (full_sort)
            - Experiment 2/

Usage:
    python mask_area.py [base_directory] [--force_update]

    - base_directory (optional): The base directory containing experiment subdirectories.
                                 Default is "Data/".
    - --force_update (optional): If provided, force the generation of CSV files even if they already exist.

Examples:
    1. Generate CSV files in "Data":
       python mask_area.py

    2. Generate CSV files in a specific directory with force update:
       python mask_area.py /path/to/experiment_data --force_update

Author:
    Jacob Stoker

Date:
    December 18, 2023

Disclaimer:
    I'm writing all this just as practice of documenting a script properly, but I can't imagine this is going to be useful for anyone else.
    If it turns out it is useful to you and you'd just need some tweaks let me know and I'll happily update it
"""


import cv2
from tqdm import tqdm
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse


def get_mask_area(path_to_image: Path) -> float:
    """Return the area of the white pixels in a mask in microns"""
    image = cv2.imread(str(path_to_image), 0)
    num_of_pixels = cv2.countNonZero(image)
    area_of_pixel = 4.92843e-05  # mm^2
    mask_area = area_of_pixel * num_of_pixels  # mm^2
    mask_area_microns = mask_area * 1000
    return mask_area_microns


def get_list_of_subdirectories(base_directory: Path) -> list:
    """Return a list containing the full paths to all subdirectories in a given directory"""
    return [
        subdirectory
        for subdirectory in base_directory.iterdir()
        if subdirectory.is_dir()
    ]


def get_diameter_from_area(area: float) -> float:
    """Return the equivalent diameter to an input area assuming it is a circle"""
    radius = math.sqrt(area / math.pi)
    diameter = radius * 2
    return diameter


def update_and_save_dataframe(
    images_df: pd.DataFrame, sort_by: list, output_csv: Path, calculate_percent=False
):
    """
    Sort an input DataFrame based on specified sorting criteria, then calculate and add additional columns before saving
    as a CSV.

    Parameters:
    - images_df (pd.DataFrame): Input DataFrame containing image data.
    - sort_by (list): List of column names to use for sorting the DataFrame.
    - output_csv (Path): Path to the CSV file where the sorted DataFrame will be saved.
    - calculate_percent (bool, optional): If True, calculate the percentage of particles and add it to the DataFrame.
                                          Default is False.
    """
    sorted_df = images_df.sort_values(by=sort_by)
    sorted_df["Diameter"] = sorted_df["Area"].apply(get_diameter_from_area)
    sorted_df["Freq"] = sorted_df["Diameter"].map(sorted_df["Diameter"].value_counts())
    sorted_df["Cumulative Freq"] = sorted_df["Freq"].cumsum()
    if calculate_percent:
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df["Percentage of Particle"] = (
            (sorted_df.index + 1) / len(sorted_df)
        ) * 100
    sorted_df.to_csv(output_csv, index=False)


def create_csvs(base_directory: Path, force_update=False):
    """
    Generate CSV files based on image data in subdirectories.

    Parameters:
    - base_directory (Path): The base directory containing experiment subdirectories.
    - force_update (bool, optional): If True, force the generation of CSV files even if they already exist.
                                     Default is False.

    Notes:
    - Expects the following folder structure:
        - mask_area.py
        - base_directory/
            - Experiment 1/
                - Before/
                    - a1_mask1.jpg
                    - b2_mask1.jpg
                - After/
            - Experiment 2/
    - The function loops through experiment directories within the base directory.
    - For each experiment directory, it processes subdirectories (e.g., "Before" and "After").
    - Image data is extracted from images in the subdirectory, and CSVs are created based on the data.
    - For each subdirectory, it generates two CSV files: one sorted by 'Grid' and 'Area',
      and another fully sorted by 'Area'.
    - Unless fore_update is set, it will skip regenerating any existing CSVs
    """
    experiment_directories = get_list_of_subdirectories(base_directory)
    logfile = open("skipped_folders.log", "w")
    logfile.write(f"Timestamp: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")

    for directory in tqdm(experiment_directories, desc="Experiments"):
        for subdirectory in tqdm(
            get_list_of_subdirectories(directory), desc="Subdirectories", leave=False
        ):
            csv_basename = f"{directory.name}_{subdirectory.name}".lower().replace(
                " ", ""
            )
            grid_sorted_csv_name = f"{csv_basename}.csv"
            grid_sorted_csv_path = directory / grid_sorted_csv_name
            grid_csv_exists = grid_sorted_csv_path.is_file() and not force_update

            fully_sorted_csv_name = f"{csv_basename}_full_sort.csv"
            fully_sorted_csv_path = directory / fully_sorted_csv_name
            full_sort_csv_exists = fully_sorted_csv_path.is_file() and not force_update

            if grid_csv_exists and full_sort_csv_exists:
                logfile.write(
                    f"NOTE: Skipping '{subdirectory}' because the CSVs already exist\n"
                )
                continue
            elif grid_csv_exists:
                logfile.write(
                    f"NOTE: Skipping '{grid_sorted_csv_path.name}' because it already exists\n"
                )
            elif full_sort_csv_exists:
                logfile.write(
                    f"NOTE: Skipping '{fully_sorted_csv_path.name}' because it already exists\n"
                )

            image_paths = list(subdirectory.rglob("*.jpg"))
            if image_paths:
                data = []
                for image in tqdm(image_paths, desc="Images", leave=False):
                    pattern = r"(.+)_mask([0-9]+)"
                    matches = re.findall(pattern, image.name)
                    grid, mask = matches[0]
                    mask = int(mask)
                    if mask not in [0, 1]:
                        area = get_mask_area(image)
                        image_data = {"Grid": grid, "Mask": mask, "Area": area}
                        data.append(image_data)

                images_df = pd.DataFrame(data)
                if not grid_csv_exists:
                    update_and_save_dataframe(
                        images_df=images_df,
                        sort_by=["Grid", "Area"],
                        output_csv=grid_sorted_csv_path,
                    )
                if not full_sort_csv_exists:
                    update_and_save_dataframe(
                        images_df=images_df,
                        sort_by=["Area"],
                        output_csv=fully_sorted_csv_path,
                        calculate_percent=True,
                    )
    logfile.close()


def plot_data(base_directory: Path):
    """Plot a seaborn scatter graph for each *_full_sort.csv file in a base directory"""
    sns.set_style("darkgrid")
    experiment_directories = get_list_of_subdirectories(base_directory)
    for directory in experiment_directories:
        csvs = [
            csv
            for csv in directory.iterdir()
            if csv.suffix == ".csv" and "full_sort" in csv.stem
        ]
        _, ax = plt.subplots()
        for csv in csvs:
            df = pd.read_csv(csv)
            label = csv.stem.split("_")[1]
            sns.scatterplot(
                x="Diameter", y="Percentage of Particle", data=df, label=label
            )

        ax.set_xlabel("Diameter (microns)")
        ax.set_ylabel("Percentage of Particle (%)")
        ax.set_title(f"Graph for {directory.name}")
        ax.legend()
        plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate CSV files based on image data in subdirectories."
    )
    parser.add_argument(
        "base_directory",
        type=Path,
        nargs="?",
        default=Path("Data"),
        help="The base directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--force_update",
        action="store_true",
        help="Force the generation of CSV files even if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    create_csvs(base_directory=args.base_directory, force_update=args.force_update)
    plot_data(args.base_directory)
