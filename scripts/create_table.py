import argparse
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    df_list = []
    for setting in ["xsum_pegasus", "xsum_reference", "cnn_pegasus", "cnn_reference"]:
        df = pd.read_csv(os.path.join(args.input_dir, "{}__report.csv".format(setting)), header=[0, 1])
        cur_df = df.mean().unstack(1)[["ndcg", "ap"]]
        cur_df.columns = pd.MultiIndex.from_tuples([(setting, x) for x in cur_df.columns.tolist()])
        df_list.append(cur_df)

    table_df = pd.concat(df_list, axis=1)
    table_df = table_df.loc[["lexrank", "bertscore", "rouge1", "simcse", "gptpmi", "text-davinci-003", "attention", "perplexity"]]
    table_df.to_csv(args.output)