import argparse
import sys

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_header", type=int, default=1, choices=[1, 2])
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    if args.num_header == 1:
        df = pd.read_csv(args.input, index_col=0)
    elif args.num_header == 2:
        df = pd.read_csv(args.input, index_col=0, header=[0, 1])
    else:
        raise ValueError("Invalid num_header: {}".format(args.num_header))
    print(df.to_latex(float_format="%.4f"))
