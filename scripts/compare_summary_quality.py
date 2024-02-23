from collections import defaultdict
import json
from typing import List

from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm


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


def dataset2dict(dataset,
                 input_fieldname,
                 output_fieldname):
    id_data_dict = {}
    for split in ["train", "validation", "test"]:
        for _, row in tqdm(dataset[split].to_pandas().iterrows()):
            id_data_dict[row["id"]] = {
                "split": split,
                "document": row[input_fieldname],
                "summary": row[output_fieldname]
            }
    return id_data_dict


if __name__ == "__main__":
    """
    Compare generated summary qualities of reconstructable summaries
    """
    xsum_dataset = load_dataset("xsum")
    xsum_id_data_dict = dataset2dict(xsum_dataset, "document", "summary")

    xsum_reference_all_data = json.load(open("data/xsum_pegasus_all.json"))
    #xsum_reference_filtered_data = list(filter(lambda x: x["split"] in ["validation", "test"] , xsum_reference_all_data))

    id2label = {}
    #for example in xsum_reference_filtered_data:
    for example in xsum_reference_all_data:
        score = np.array(example["summary_label"]).mean()
        if score > 0.5:
            label = "reconstructable"
        elif score < 0.5:
            label = "unreconstructable"
        else:
            label = "unsure"
        id2label[example["original_id"]] = label

    xsum_pegasus_all_data = json.load(open("data/xsum_pegasus_all.json"))
    #xsum_pegasus_filtered_data = list(filter(lambda x: x["split"] in ["validation", "test"] , xsum_pegasus_all_data))
    xsum_pegasus_filtered_data = list(filter(lambda x: x["split"] in ["test"] , xsum_pegasus_all_data))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    tagged_examples = []
    for example in xsum_pegasus_filtered_data:
        ref_sum = xsum_id_data_dict[example["original_id"]]["summary"]
        gen_sum = example["gen_summary"]
        scores = scorer.score(ref_sum, gen_sum)
        example["scores"] = scores
        tagged_examples.append((example["original_id"],
                                xsum_id_data_dict[example["original_id"]]["document"],
                                id2label[example["original_id"]],
                                ref_sum,
                                gen_sum))
    tagged_df = pd.DataFrame(tagged_examples, columns=["original_id", "document", "label", "ref", "gen"])
    tagged_df.to_csv("outputs/xsum__tagged_examples.csv", index=False)

    xsum_example_dict = defaultdict(list)
    for example in xsum_pegasus_filtered_data:
        # * Reconstructability is juged by the reference summary
        label = id2label[example["original_id"]]
        xsum_example_dict[label].append(example["scores"])

    rouge_df_dict = {}
    for name, scores_list in xsum_example_dict.items():
        rouge_df_dict[name] = rouge_scores2df(scores_list)

    for rouge in ["ROUGE-1_F", "ROUGE-2_F", "ROUGE-L_F"]:
        # The length is different but NaN will be filled
        rouge_df = pd.DataFrame({"reconstructable": rouge_df_dict["reconstructable"][rouge],
                                 "unsure": rouge_df_dict["unsure"][rouge],
                                 "unreconstructable": rouge_df_dict["unreconstructable"][rouge]})
        rouge_df.boxplot()
        plt.savefig("figures/xsum__reconstructablility_rouge__{}.png".format(rouge))
        plt.close()

