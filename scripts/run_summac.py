import argparse
from summac.model_summac import SummaCZS, SummaCConv

import pandas as pd
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Follow-up Rewrite")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.input)

    summac_zs = SummaCZS(granularity="sentence", model_name="vitc",
                         device="cuda" if torch.cuda.is_available() else "cpu")
                         # If you have a GPU: switch to: device="cuda"
    summac_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e",
                             device="cuda" if torch.cuda.is_available() else "cpu",
                             start_file="default", agg="mean")

    summac_zs_scores = []
    summac_conv_scores = []
    for index, row in tqdm(df.iterrows()):
        gen_summ = row["gen"]
        input_doc = row["document"]
        summac_zs_scores.append(summac_zs.score([input_doc], [gen_summ])["scores"][0])
        summac_conv_scores.append(summac_conv.score([input_doc], [gen_summ])["scores"][0])

    df["summac_zs"] = summac_zs_scores
    df["summac_conv"] = summac_conv_scores

    df.to_csv(args.output, index=False)
