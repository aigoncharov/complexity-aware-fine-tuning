import csv
from pathlib import Path

import pandas as pd


def join_with_masj_reasoning_score(df, rating_threshold=9):
    masj_data_path = Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem_reasoning_score.tsv")
    masj_reasoning_score_df = pd.read_csv(
        masj_data_path,
        sep="\t",
        header=0,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar="\\",
    )

    masj_reasoning_score_df.dropna(subset="masj_num_reasoning_steps", inplace=True)

    return pd.merge(
        df, masj_reasoning_score_df[["question_id", "masj_num_reasoning_steps"]], how="inner", on="question_id"
    )
