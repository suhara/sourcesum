from typing import List
import pandas as pd


def rouge_scores2df(scores_list: List) -> pd.DataFrame:
    eval_df = pd.DataFrame(scores_list)
    eval_df["ROUGE-1_P"] = eval_df["rouge1"].apply(lambda x: x[0])
    eval_df["ROUGE-1_R"] = eval_df["rouge1"].apply(lambda x: x[1])
    eval_df["ROUGE-1_F"] = eval_df["rouge1"].apply(lambda x: x[2])
    eval_df["ROUGE-2_P"] = eval_df["rouge2"].apply(lambda x: x[0])
    eval_df["ROUGE-2_R"] = eval_df["rouge2"].apply(lambda x: x[1])
    eval_df["ROUGE-2_F"] = eval_df["rouge2"].apply(lambda x: x[2])
    eval_df["ROUGE-L_P"] = eval_df["rougeL"].apply(lambda x: x[0])
    eval_df["ROUGE-L_R"] = eval_df["rougeL"].apply(lambda x: x[1])
    eval_df["ROUGE-L_F"] = eval_df["rougeL"].apply(lambda x: x[2])
    eval_df.drop(columns=["rouge1", "rouge2", "rougeL"], inplace=True)
    return eval_df

