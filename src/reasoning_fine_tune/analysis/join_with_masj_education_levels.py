from pathlib import Path

import pandas as pd


def join_with_masj_education_levels(df, rating_threshold=9):
    masj_data_path = Path(__file__).parent.joinpath("../../../data/out/masj/mmlu_masj_education_levels.tsv")
    masj_edu_levels_df = pd.read_csv(
        masj_data_path,
        sep="\t",
        header=0,
    )

    masj_edu_levels_df = masj_edu_levels_df[masj_edu_levels_df["masj_rating"] >= rating_threshold]

    return pd.merge(df, masj_edu_levels_df[["question_id", "masj_complexity"]], how="inner", on="question_id")
