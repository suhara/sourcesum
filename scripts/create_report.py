import argparse
from glob import glob
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs")
    parser.add_argument("--input", type=str, required=True, choices=["xsum_pegasus", "xsum_reference", "cnn_pegasus", "cnn_reference"])
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("outputs", "{}__report.csv".format(args.input))

    df_list = []
    for filepath in glob(os.path.join(args.input_dir, "{}__*__eval.csv".format(args.input))):
        df = pd.read_csv(filepath)
        method = os.path.basename(filepath).split("__")[1]
        df.columns = pd.MultiIndex.from_product([[method], df.columns])
        df_list.append(df)
    eval_df = pd.concat(df_list, axis=1)
    print(eval_df.mean())
    eval_df.to_csv(args.output, index=False)


