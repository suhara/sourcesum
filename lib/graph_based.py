from typing import List

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import numpy as np


class LexRankScorer:
    def __init__(self,
                 documents: List[str]):
        self.lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    def calc_out_in_scores(self,
                           output_sents: List[str],
                           input_sents: List[str],
                           score_name: str = None) -> np.ndarray:
        """Return out_in_scores."""
        scores = self.lxr.rank_sentences(input_sents, threshold=None, fast_power_method=False)
        out_in_scores = np.tile(np.array(scores), [len(output_sents) , 1])

        return out_in_scores
