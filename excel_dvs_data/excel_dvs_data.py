import re
import pandas as pd
import logging

"""Script to add required columns to an Excel Spreadsheet of DVS data"""

SOURCE_SPREADSHEET = "DVS Heptane data.xlsx"
DESTINATION_SPREADSHEET = "DVS Heptane data Updated.xlsx"

logging.basicConfig(level=logging.INFO)


def solvent_a_or_b(df: pd.DataFrame) -> str:
    """Return the column name in the DataFrame that exists, either Solvent A or B (or a useful string for the error message if neither)"""
    for solvent in ["A", "B"]:
        pressure_solvent_column = f"Target Partial Pressure (Solvent {solvent}) [%]"
        if pressure_solvent_column in df.columns:
            return pressure_solvent_column
    return "Target Partial Pressure (Solvent A or B)"


def check_for_needed_columns(df: pd.DataFrame, pressure_solvent_column: str) -> bool:
    """Return True if all the necessary columns for the script are present and not full of NaN values"""
    needed_columns = [
        "Time [minutes]",
        "Moisture content %",
        "RH Direction",
        pressure_solvent_column,
    ]

    for column in needed_columns:
        if column in df.columns:
            if not df[column].notna().any():
                logging.error(
                    f"ERROR: Column '{column}' exists but is full of NaN values."
                )
                return False
        else:
            logging.error(f"ERROR: '{column}' column does not exist")
            return False
    return True


def add_normalised_column(
    df: pd.DataFrame, grouping_column: str, target_column: str, new_column_name: str
) -> pd.DataFrame:
    """Add a column normalising target_column by changes in grouping_column"""
    df[f"{target_column}_at_change"] = df.groupby(grouping_column)[
        target_column
    ].transform("first")
    df[f"{new_column_name} NEW"] = df[target_column] - df[f"{target_column}_at_change"]
    df.drop(columns=[f"{target_column}_at_change"], inplace=True)
    return df


def add_change_column(
    df: pd.DataFrame,
    target_column: str,
    new_column_name: str,
) -> pd.DataFrame:
    """
    Add a column to the DataFrame indicating a change in target_column, so you can properly group by the target_column without worrying about repeated values
    eg. target_column = A, B, B, A -> new_column_name = 1, 2, 2, 3
    """
    df[new_column_name] = (df[target_column] != df[target_column].shift()).cumsum()
    return df


def add_pressure_change_columns(
    df: pd.DataFrame, pressure_solvent_column: str
) -> pd.DataFrame:
    """Add the columns based on pressure change-related to the DataFrame"""
    df = add_change_column(df, pressure_solvent_column, "Pressure Change")
    df = add_normalised_column(
        df, "Pressure Change", "Time [minutes]", "Normalised time (Individual RH Steps)"
    )
    df = add_normalised_column(
        df,
        "Pressure Change",
        "Moisture content %",
        "Normalised moisture content per RH step",
    )

    df.drop(columns=["Pressure Change"], inplace=True)
    return df


def add_direction_change_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns based on direction change to the DataFrame"""
    df = add_change_column(df, "RH Direction", "Direction Change")
    df = add_normalised_column(
        df,
        "Direction Change",
        "Time [minutes]",
        "Normalised time (up vs down RH ramps)",
    )
    df.drop(columns=["Direction Change"], inplace=True)
    return df


def calculate_and_add_columns(
    df: pd.DataFrame, pressure_solvent_column: str
) -> pd.DataFrame:
    """Calculate and add all necessary columns to the DataFrame"""
    df = add_pressure_change_columns(df, pressure_solvent_column)
    df = add_direction_change_columns(df)
    return df


def process_spreadsheet(src_spreadsheet: str, dst_spreadsheet: str):
    """Read source spreadsheet, add all the new columns and save it as destination spreadsheet"""
    src_workbook = pd.read_excel(src_spreadsheet, sheet_name=None)
    dst_workbook = pd.ExcelWriter(dst_spreadsheet)

    for sheet in src_workbook:
        # The sheets we want to process will be named, anything called eg Sheet1 should be ignored
        if re.match(r"^Sheet\d+$", sheet):
            logging.info(f"Skipping {sheet}\n")
            continue
        else:
            logging.info(f"Processing {sheet}")

        df = src_workbook[sheet]
        df.columns = df.columns.str.strip()
        pressure_solvent_column = solvent_a_or_b(df)

        if not check_for_needed_columns(df, pressure_solvent_column):
            logging.info(f"Skipping {sheet} because of missing column\n")
            continue

        df = df.apply(pd.to_numeric, errors="ignore")
        df = calculate_and_add_columns(df, pressure_solvent_column)
        df.to_excel(dst_workbook, sheet_name=sheet, index=False)

    if dst_workbook.sheets:
        dst_workbook.close()


if __name__ == "__main__":
    process_spreadsheet(SOURCE_SPREADSHEET, DESTINATION_SPREADSHEET)
