import itertools
import json
import sys

from matplotlib import pyplot as plt
import pandas as pd


if __name__ == "__main__":
    with open("data/xsum_reference.json") as fin:
        re_xsum_ref_examples = json.load(fin)
    with open("data/xsum_pegasus.json") as fin:
        re_xsum_peg_examples = json.load(fin)
    with open("data/cnn_reference.json") as fin:
        re_cnn_ref_examples = json.load(fin)
    with open("data/cnn_pegasus.json") as fin:
        re_cnn_peg_examples = json.load(fin)

    color_codes = ["#0072b2", "#f0e442"]

    # * Diff of relevant sentence positions
    xsum_diff_rel_pos_df = pd.DataFrame({
        "Reference": pd.Series(list(itertools.chain(*[[j - i for i, j in zip(x["rel_sent_positions"][0:], x["rel_sent_positions"][1:])] for x in re_xsum_ref_examples]))).value_counts(),
        "Pegasus": pd.Series(list(itertools.chain(*[[j - i for i, j in zip(x["rel_sent_positions"][0:], x["rel_sent_positions"][1:])] for x in re_xsum_peg_examples]))).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    xsum_diff_rel_pos_df.div(xsum_diff_rel_pos_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                            color=color_codes)
    plt.ylabel("%")
    plt.xlabel("Sentence gap between adjacent relevant sentences")
    plt.tight_layout()
    plt.savefig("figures/xsum_diff_rel_pos_dist.pdf")
    plt.close()

    # TODO
    cnn_diff_rel_pos_df = pd.DataFrame({
        "Reference": pd.Series(list(itertools.chain(*[[j - i for i, j in zip(x["rel_sent_positions"][0:], x["rel_sent_positions"][1:])] for x in re_cnn_ref_examples]))).value_counts(),
        "Pegasus": pd.Series(list(itertools.chain(*[[j - i for i, j in zip(x["rel_sent_positions"][0:], x["rel_sent_positions"][1:])] for x in re_cnn_peg_examples]))).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    cnn_diff_rel_pos_df.div(cnn_diff_rel_pos_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                          color=color_codes)
    plt.ylabel("%")
    plt.xlabel("Sentence gap between adjacent relevant sentences")
    plt.tight_layout()
    plt.savefig("figures/cnn_diff_rel_pos_dist.pdf")
    plt.close()

    # * # of relevant sentences
    xsum_rel_num_df = pd.DataFrame({
        "Reference": pd.Series([len(x["rel_sent_positions"]) for x in re_xsum_ref_examples]).value_counts(),
        "Pegasus": pd.Series([len(x["rel_sent_positions"]) for x in re_xsum_peg_examples]).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    xsum_rel_num_df.div(xsum_rel_num_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                  color=color_codes)
    plt.ylabel("%")
    plt.xlabel("# of relevant sentences")
    plt.tight_layout()
    plt.savefig("figures/xsum_num_rel_sent_dist.pdf")
    plt.close()

    # TODO: How to consider different summary sentences for the same document
    cnn_rel_num_df = pd.DataFrame({
        "Reference": pd.Series([len(x["rel_sent_positions"]) for x in re_cnn_ref_examples]).value_counts(),
        "Pegasus": pd.Series([len(x["rel_sent_positions"]) for x in re_cnn_peg_examples]).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    cnn_rel_num_df.div(cnn_rel_num_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                color=color_codes)
    plt.ylabel("%")
    plt.xlabel("# of relevant sentences")
    plt.tight_layout()
    plt.savefig("figures/cnn_num_rel_sent_dist.pdf")
    plt.close()


    # * Relevant sentence positions
    xsum_rel_pos_df = pd.DataFrame({
        "Reference": pd.Series(list(itertools.chain(*[x["rel_sent_positions"] for x in re_xsum_ref_examples]))).value_counts(),
        "Pegasus": pd.Series(list(itertools.chain(*[x["rel_sent_positions"] for x in re_xsum_peg_examples]))).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    xsum_rel_pos_df.div(xsum_rel_pos_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                  color=color_codes)
    plt.ylabel("%")
    plt.xlabel("Relevant sentence position")
    plt.tight_layout()
    plt.savefig("figures/xsum_rel_sent_pos.pdf")
    plt.close()

    cnn_rel_pos_df = pd.DataFrame({
        "Reference": pd.Series(list(itertools.chain(*[x["rel_sent_positions"] for x in re_cnn_ref_examples]))).value_counts(),
        "Pegasus": pd.Series(list(itertools.chain(*[x["rel_sent_positions"] for x in re_cnn_peg_examples]))).value_counts()
    })

    plt.rcParams.update({"font.size": 14})
    cnn_rel_pos_df.div(cnn_rel_pos_df.sum(axis=0), axis=1).plot(kind="bar", rot=0,
                                                                yticks=[0, 0.05, 0.1, 0.15, 0.2],
                                                                color=color_codes)
    plt.ylabel("%")
    plt.xlabel("Relevant sentence position")
    plt.tight_layout()
    plt.savefig("figures/cnn_rel_sent_pos.pdf")
    plt.close()
