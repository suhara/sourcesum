import itertools
from typing import Tuple, Dict, List, Any

import torch
from transformers import (
    PreTrainedTokenizer,
)


class Preprocessor:
    """Tokenize and assign sentence ids"""
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 add_cls_token: bool = False,
                 add_eos_token: bool = False,
                 device: torch.device = None):

        self.tokenizer = tokenizer
        self.add_cls_token = add_cls_token
        self.add_eos_token = add_eos_token
        self.device = device

    def process_input_sentences(self,
                                sentences: List[str]) -> Tuple[Dict, List]:
        """One text at one time"""
        input_ids = self.tokenizer(sentences,
                                   add_special_tokens=False).input_ids
        sent_ids = []
        for i, ids in enumerate(input_ids):
            sent_ids.append([i + 1] * len(ids))

        input_ids_list = list(itertools.chain(*input_ids))
        sent_ids_list = list(itertools.chain(*sent_ids))
        if self.add_cls_token:
            input_ids_list = [self.tokenizer.cls_token_id] + input_ids_list
            sent_ids_list = [0] + sent_ids_list
        if self.add_eos_token:
            input_ids_list.append(self.tokenizer.eos_token_id)
            sent_ids_list.append(0)

        attention_mask_list = [1] * len(input_ids_list)
        if self.device is not None:
            inputs = {"input_ids": torch.LongTensor([input_ids_list]).to(self.device),
                      "attention_mask": torch.LongTensor([attention_mask_list]).to(self.device)}
        else:
            inputs = {"input_ids": torch.LongTensor([input_ids_list]),
                      "attention_mask": torch.LongTensor([attention_mask_list])}

        return inputs, sent_ids_list


    def process_output_sentences(self,
                                 sentences):
        output_ids = self.tokenizer(sentences,
                                    add_special_tokens=False).input_ids
        sent_ids = []
        for i, ids in enumerate(output_ids):
            sent_ids.append([i + 1] * len(ids))

        output_ids_list = list(itertools.chain(*output_ids))
        sent_ids_list = list(itertools.chain(*sent_ids))

        # Add <pad> at the beginning
        output_ids_list = [self.tokenizer.pad_token_id] + output_ids_list
        sent_ids_list = [0] + sent_ids_list

        # Add </s> at the end
        output_ids_list.append(self.tokenizer.eos_token_id)
        sent_ids_list.append(0)

        # attention_mask_list = [1] * len(input_ids_list)
        # if self.device is not None:
        #     inputs = {"input_ids": torch.LongTensor([input_ids_list]).to(self.device),
        #               "attention_mask": torch.LongTensor([attention_mask_list]).to(self.device)}
        # else:
        #     inputs = {"input_ids": torch.LongTensor([input_ids_list]),
        #               "attention_mask": torch.LongTensor([attention_mask_list])}

        return torch.LongTensor([output_ids_list]).to(self.device), sent_ids_list
