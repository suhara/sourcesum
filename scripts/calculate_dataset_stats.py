import json

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [" ".join(grams) for grams in n_grams]


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    dict_list = []
    for name in ["cnn_pegasus", "cnn_reference", "xsum_pegasus", "xsum_reference"]:
        filepath = "data/{}.json".format(name)
        data = json.load(open(filepath))

        novel_ngram_dict = {}
        for n in range(1, 5):
            ratio_list = []
            for x in data:
                doc = " ".join([ann["sentence"] for ann in x["annotations"]]).lower()
                summ = x["gen_summary"].lower()
                doc_ngram_set = set(get_ngrams(doc, n))
                summ_ngram_set = set(get_ngrams(summ, n))
                ratio_list.append(len(summ_ngram_set - doc_ngram_set) / len(summ_ngram_set))
            novel_ngram_dict[n] = np.array(ratio_list).mean()

        dict_list.append(
            {"name": name,
             "num_pairs": len(data),
             "avg_num_sent": np.array([len(x["annotations"]) for x in data]).mean(),
             "avg_rel_sent": np.array([len(x["rel_sent_positions"]) for x in data]).mean(),
             "avg_input_len": np.array([len(tokenizer.encode(" ".join([ann["sentence"] for ann in x["annotations"]]), add_special_tokens=False)) for x in data]).mean(),
             "avg_summ_len": np.array([len(tokenizer.encode(x["gen_summary"], add_special_tokens=False)) for x in data]).mean(),
             "novel_ngram_1": novel_ngram_dict[1],
             "novel_ngram_2": novel_ngram_dict[2],
             "novel_ngram_3": novel_ngram_dict[3],
             "novel_ngram_4": novel_ngram_dict[4]})

    df = pd.DataFrame(dict_list)
    df.to_csv("outputs/dataset_stats.csv", index=False)
    with open("outputs/dataset_stats.tex", "w") as fout:
        fout.write(df.to_latex())

