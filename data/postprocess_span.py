
import argparse
import os
import json, _jsonnet, jsonlines

from typing import List, Dict, Any
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import spacy

import numpy as np

has_match = 0
gold_has_match = 0
pred_seq_is_in_patterns = 0
total = 0
EXCLUDE = ["logits", "masks", "loss", "mask", "sent_loss", "seq_tag_probs", "sent_probs", "seq_label_loss"]

def flatten_list(l):
    # fail lol
    res = []
    if not isinstance(l, list):
        return l
    else:
        if len(l) < 1:
            return l
        if isinstance(l[0], list):
            for it in l:
                res.extend(flatten_list(list(it)))
            return res
        else:
            return l

def get_shapes(list_of_dict_batches):
    #for batch in list_of_dict_batches:
    batch = list_of_dict_batches[0]
    for k, examples in batch.items():
        if k=="paragraph":
            if isinstance(data[0]['paragraph'], dict):
                print(k, data[0]['paragraph']['tokens'].keys())
                print(np.asarray(examples['tokens']['token_ids']).shape)
                continue
            elif isinstance(data[0]['paragraph'], list):
                print(np.asarray(examples).shape)
        if k in ["tags", "pred_tags"]:
            print(k, f"len: {len(examples)}", [np.asarray(ex).shape for ex in examples])
        elif isinstance(examples, list):
            print(k, np.asarray(examples).shape)
            # top_k_tags => examples[0] = List[{pred_tags: List[str], score: float}]
        else:
            print(k, type(examples))
    return


def merge_batches(list_of_dict_batches):
    agg_examples = defaultdict(lambda: list())
    for batch in list_of_dict_batches:
        for k, examples in batch.items():
            if k in EXCLUDE:
                continue
            if k=="paragraph":
                if isinstance(examples, dict): # token_ids is stored in a nested dict ; 
                    # token_ids.shape = batch x 1(=max_parag_len) x max_seq_len
                    squeezed_parag = np.squeeze(np.asarray(examples['tokens']['token_ids']), axis=1)
                elif isinstance(examples, list): # token_ids is stored directly under paragraph
                    agg_examples[k].append(examples) # token_ids.shape = batch x max_seq_len
                    continue
            elif k in ["tags", "pred_tags", "top_k_tags"]: # batch x [arbitrary len]
                agg_examples[k].append(examples)
                continue
            elif k in ["sent_labels", "seq_tags", "orig_tag_indices"]: # batch x 1
                squeezed_parag = np.squeeze(np.asarray(examples), axis=1)
            else:
                if k=="metadata":
                    continue
                squeezed_parag = np.squeeze(np.asarray(examples), axis=0)
            agg_examples[k].append(squeezed_parag.tolist())
            #print(np.concatenate(examples, dim=0).shape) # cannot concat because each batch has different max sent len!
    return dict(agg_examples)

def convert_to_easy_read(data, idx_to_tag_file, idx_to_label_file, batched):
    global has_match, gold_has_match, total
    with open(idx_to_tag_file) as f:
        idx_to_tag = {i: line.strip() for i, line in enumerate(f.readlines())}
    with open(idx_to_label_file) as f:
        idx_to_label = {i: line.strip() for i, line in enumerate(f.readlines())}

    if batched:
        batch_tokens = data["merged_tokens"]
        for batch_idx, batch in enumerate(tqdm(batch_tokens)):
            for i, sent in enumerate(batch):
                sent_pred_has_match = False
                total +=1 
                
                if 'seq_tags' in data.keys():
                    if not all([(tag in [0, 1]) for tag in data['seq_tags'][batch_idx][i]]):
                        gold_has_match += 1
                if not all([tag=='O' for tag in data['pred_tags'][batch_idx][i]]):
                    sent_pred_has_match = True
                    has_match += 1

                if sent_pred_has_match:
                    if isinstance(sent, str):
                        sent = sent.split()
                
                # easy_read_sent = bio_to_easy_read(sent, data["realigned_pred_seq_tags"][batch_idx][i])
                out_dict = {
                    "merged_tokens": agg_data["merged_tokens"][batch_idx][i],
                    # "easy_read": easy_read_sent,
                    **{key: val[batch_idx][i] for key, val in data.items() if key.startswith("realigned_pred_tags")},
                    **{key: val[batch_idx][i] for key, val in data.items() if key.startswith("score")}
                }
                if 'seq_tags' in data.keys():
                    out_dict['seq_tags'] = [idx_to_tag[it] for it in data['seq_tags'][batch_idx][i]]
                if 'sent_labels' in data.keys():
                    out_dict['sent_labels'] = [idx_to_label[it] for it in [data['sent_labels'][batch_idx][i]]]
                if 'realigned_gold_tags' in data.keys():
                    out_dict['realigned_gold_tags'] = data['realigned_gold_tags'][batch_idx][i]
                # realigned_seq_tags = {key: val for key, val in data.items() if key.startswith("realigned_pred_seq_tags")}
                # scores = {key: val for key, val in data.items() if key.startswith("score")}

                # out_dict.update(realigned_seq_tags)
                # out_dict.update(scores)
                
                if "pred_sent_labels" in data.keys():
                    out_dict["pred_sent_labels"] = data['pred_sent_labels'][batch_idx][i]

                if "pred_seq_tags" in data.keys():
                    out_dict["pred_seq_tags"] = data['pred_seq_tags'][batch_idx][i]
                elif "pred_tags" in data.keys():
                    out_dict["pred_tags"] = data['pred_tags'][batch_idx][i]
                
                yield out_dict
    else: # output non-batched data
        tokens = data["merged_tokens"]
        for i, sent in enumerate(tokens):
            sent_pred_has_match = False
            total +=1 
            if 'seq_tags' in data.keys():
                if not all([(tag in [0, 1]) for tag in data['seq_tags'][i]]):
                    gold_has_match += 1
            if not all([tag=='O' for tag in data['pred_tags'][i]]):
                sent_pred_has_match = True
                has_match += 1

            if sent_pred_has_match:
                if isinstance(sent, str):
                    sent = sent.split()
            # easy_read_sent = bio_to_easy_read(sent, data["realigned_pred_seq_tags"][batch_idx][i])
            out_dict = {
                "merged_tokens": agg_data["merged_tokens"][i],
                # "easy_read": easy_read_sent,
                **{key: val[i] for key, val in data.items() if key.startswith("realigned_pred_tags")},
                **{key: val[i] for key, val in data.items() if key.startswith("score")}
            }
            if 'seq_tags' in data.keys():
                out_dict["seq_tags"] = idx_to_tag[data['seq_tags'][i]]
            if 'sent_labels' in data.keys():
                out_dict['sent_labels'] = idx_to_label[data['sent_labels'][i]]
            if 'realigned_gold_tags' in data.keys():
                out_dict['realigned_gold_tags'] = data['realigned_gold_tags'][i]
            # realigned_seq_tags = {key: val for key, val in data.items() if key.startswith("realigned_pred_seq_tags")}
            # scores = {key: val for key, val in data.items() if key.startswith("score")}

            # out_dict.update(realigned_seq_tags)
            # out_dict.update(scores)
            
            if "pred_sent_labels" in data.keys():
                out_dict["pred_sent_labels"] = data['pred_sent_labels'][i]

            if "pred_seq_tags" in data.keys():
                out_dict["pred_seq_tags"] = data['pred_seq_tags'][i]
            elif "pred_tags" in data.keys():
                out_dict["pred_tags"] = data['pred_tags'][i]
            
            yield out_dict


def bio_to_easy_read(tokens: List[str], tags: List[str]):
    assert len(tokens) == len(tags)
    easy_read_tokens = []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            easy_read_tokens.append(tok)
        else:
            easy_read_tokens.append(tok+"/"+tag)
    return " ".join(easy_read_tokens)
    
class TextHandler(object):
    def __init__(self, tokenizer_model_type, idx_to_tag_file, extra_tokens):
        """extra_tokens: List[str]
        """
        self.tokenizer_model_type = tokenizer_model_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_type,
            additional_special_tokens=extra_tokens
            )
        with open(idx_to_tag_file) as f:
            self.idx_to_tag = {i: line.strip() for i, line in enumerate(f.readlines())}

    def is_special_token(self, token):
        return token in self.tokenizer.all_special_tokens

    def convert_subwords_to_word(self, tokens: List[str]) -> str:
        for i, tok in enumerate(tokens):
            if tok.startswith("##"):
                tokens[i] = tok.split("##")[1]
        return "".join(tokens)

    @staticmethod
    def clean_up_tokenization(self, out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3544

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string
        
    def realign_tags_to_tokens(
            self,
            tokens: List[str],
            gold_tags: List[str] = None, # TODO: reconstruct orig gold tags from expanded gold tags, or just use OG gold tags??
            # probably safer to reconstruct but check it's the same as OG after reconstruction??
            top_1_tags: List[str] = None,
            top_k_tags: List[Dict[str,Any]] = None,
            force_non_o: bool = False,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            ):
        """
        tokens: list of strings from `AutoTokenizer.convert_ids_to_tokens`
        vocab: dict containing words in tokeniser's vocab as keys. Can be obtained by `tokenizer.get_added_vocab()`
        force_non_o: force use 2nd best pred tag path if 1st best path is all {O, PADDING}
        pred_tags: The top-1 viterbi tags, output when top_k_tags=None in crf_tagger.py.
        top_k_tags: The top-k viterbi tags, output when top_k_tags=k in crf_tagger.py
            List[Dcit[str,str]], each Dict is {'pred_tags': List[str], 'score': float}

        ref: https://github.com/huggingface/transformers/blob/v4.29.0/src/transformers/tokenization_utils.py
        """
        sub_texts = []
        indices_of_head_of_sub_text = []
        current_sub_text = []
        sub_text_head = -1
        for idx, token in enumerate(tokens):
            if skip_special_tokens and self.is_special_token(token):
                if token not in ["[ IMAGE ]", "CREF", "CITATION", "EQN"]: 
                    continue
            if token is None:
                continue
            if not token.startswith("##"):
                if current_sub_text: # we've collected every subword for current_sub_text 
                    # => clean up buffer (current_sub_text) before adding current token
                    sub_texts.append(self.convert_subwords_to_word(current_sub_text))
                    indices_of_head_of_sub_text.append(sub_text_head)
                    current_sub_text, sub_text_head = [], -1
                
                is_subtext_head = False
                if idx < len(tokens)-1:
                    if tokens[idx+1] is not None:
                        if tokens[idx+1].startswith("##"):
                            current_sub_text.append(token)
                            sub_text_head = idx
                            is_subtext_head = True
                
                if not is_subtext_head:
                    sub_texts.append(token)
                    indices_of_head_of_sub_text.append(idx)

            else: # is subword
                current_sub_text.append(token)

        if current_sub_text:
            sub_texts.append(self.convert_subwords_to_word(current_sub_text))
            indices_of_head_of_sub_text.append(sub_text_head)

        assert len(indices_of_head_of_sub_text)==len(sub_texts)
        
        res = {}
        if force_non_o and (top_k_tags is not None):
            for k, tag_dict in enumerate(top_k_tags):
                # tags_to_use = tag_dict['pred_tags'] \
                #     if all([tag in ['O', "@@PADDING@@"] for tag in top_1_tags[1:-1]] ) else top_1_tags
                tags_to_use = tag_dict['pred_tags']
                res[f"realigned_pred_tags_{k}"] = [tags_to_use[i] for i in indices_of_head_of_sub_text]
                res[f"score_{k}"] = tag_dict['score']
            # tags_to_use = top_k_tags[1]['pred_tags'] \
            #     if all([tag in ['O', "@@PADDING@@"] for tag in top_1_tags[1:-1]] ) else top_1_tags
            # `tags` should be the same as top_k_tags[0]['pred_tags'] 
        else:
            assert len(indices_of_head_of_sub_text) <= len(top_1_tags),\
                f"pred_tags has len = {len(top_1_tags)} but indices_of_head_of_sub_text is len {len(indices_of_head_of_sub_text)}"
            tags_to_use = top_1_tags
            res[f"realigned_pred_tags_{0}"] = [tags_to_use[i] for i in indices_of_head_of_sub_text]
            res["score_0"] = 1

        if clean_up_tokenization_spaces:
            text = " ".join(sub_texts)
            out_text = self.clean_up_tokenization(self, text)
        else:
            out_text = sub_texts
        res["text"] = out_text
        if gold_tags is not None:
            res["gold_tags"] = [self.idx_to_tag[gold_tags[i]] for i in indices_of_head_of_sub_text]

        return res
        # {
        #     "text": out_text,
        #     "pred_tags": [tags_to_use[i] for i in indices_of_head_of_sub_text],
        #     "gold_tags": [self.idx_to_tag[gold_tags[i]] for i in indices_of_head_of_sub_text]
        #     }
        
    def batch_realign_tags_to_tokens(
        self,
        token_batch: List[List[str]],
        gold_tags_batch: List[List[str]] = None,
        top_1_tags_batch: List[List[str]] = None,
        top_k_tags: List[List[Dict[str, Any]]] = None,
        force_non_o: bool = False,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        ):
        #batch_merged_text, batch_merged_pred_tags, batch_merged_gold_tags = [], [], []
        out_dict = defaultdict(lambda: list())
        if force_non_o:
            assert top_k_tags is not None
        for i, tokens in enumerate(token_batch):
            merged_dict = self.realign_tags_to_tokens(
                tokens,
                gold_tags=gold_tags_batch[i] if gold_tags_batch is not None else None,
                top_1_tags=top_1_tags_batch[i] if top_1_tags_batch is not None else None,
                top_k_tags=top_k_tags[i] if top_k_tags is not None else None,
                force_non_o=force_non_o,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            
            for k, res in merged_dict.items():
                out_dict[k].append(res)
        return dict(out_dict)
    #{"text": batch_merged_text, "pred_tags": batch_merged_pred_tags, "gold_tags": batch_merged_gold_tags}

    def get_tokenized_batches(self, agg_data):
        """Gets tokenised text (joined as a single string) and (as split tokens), 
        and add them into `agg_data` in-place
        """
        for sents in agg_data["paragraph"]:
            # sent: batch x seq_len
            #batch_sents = self.tokenizer.batch_decode(sents, clean_up_tokenization_spaces=False)
            batch_tokens = [self.tokenizer.convert_ids_to_tokens(batch) for batch in sents]
            agg_data["decoded_subwords"].append(batch_tokens) # a list

        return

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--idx_to_tag_file", type=str, default="move_tags.txt")
    parser.add_argument("--idx_to_label_file", type=str, default="class_labels.txt")
    parser.add_argument("--lm", type=str, default="allenai/scibert_scivocab_uncased") # bert-base-cased
    parser.add_argument("--configs", type=str, default="config.jsonnet") # bert-base-cased
    parser.add_argument("--force_non_o", action="store_true")
    parser.add_argument("--batched", action="store_true")
    return parser.parse_args()

if __name__=="__main__":
    args = build_args()
    ALL_PATTERNS = None
    nlp = None
    
    data = list(jsonlines.open(args.in_file))
    get_shapes(data)
    
    agg_data = merge_batches(data)
    configs = json.loads(_jsonnet.evaluate_file(args.configs))
    extra_tokens = configs['dataset_reader']['tokenizer']['tokenizer_kwargs']['additional_special_tokens']
    token_handler = TextHandler(args.lm, args.idx_to_tag_file, extra_tokens)
    out_data = defaultdict(lambda: list())
    print(agg_data.keys())
    
    agg_data["decoded_subwords"] = []
    token_handler.get_tokenized_batches(agg_data)
     
    agg_data["merged_tokens"] = []
    #agg_data["realigned_pred_seq_tags"] = []
    agg_data["realigned_gold_tags"] = []
    top_1_tags_key_name = "pred_seq_tags" if "pred_seq_tags" in agg_data.keys() else "pred_tags"
    if args.batched:
        for i, batch in enumerate(agg_data["decoded_subwords"]):
            batch_realigned_data_dict = token_handler.batch_realign_tags_to_tokens(
                token_batch=batch,
                gold_tags_batch=agg_data['seq_tags'][i] if 'seq_tags' in agg_data.keys() else None,
                top_1_tags_batch=agg_data[top_1_tags_key_name][i],
                top_k_tags=agg_data["top_k_tags"][i] if args.force_non_o else None,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                force_non_o=args.force_non_o
            )
            
            agg_data["merged_tokens"].append(batch_realigned_data_dict['text'])
            if 'gold_tags' in batch_realigned_data_dict.keys():
                agg_data["realigned_gold_tags"].append(batch_realigned_data_dict['gold_tags'])
            
            # store scores and realigned pred seq tags
            for key, val in batch_realigned_data_dict.items():
                if key not in agg_data.keys():
                    agg_data[key] = []
                agg_data[key].append(val)
            # agg_data["realigned_pred_seq_tags"].append(batch_realigned_data_dict['pred_tags'])
    else:
        for i, sent in enumerate(agg_data["decoded_subwords"]):
            realigned_data_dict = token_handler.realign_tags_to_tokens(
                sent,
                top_1_tags=agg_data[top_1_tags_key_name][i],
                top_k_tags=agg_data["top_k_tags"][i] if args.force_non_o else None,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                force_non_o=args.force_non_o
                )
            agg_data["merged_tokens"].append(realigned_data_dict['text'])
            if 'gold_tags' in realigned_data_dict.keys():
                agg_data["realigned_gold_tags"].append(realigned_data_dict['gold_tags'])
            
            for key, val in realigned_data_dict.items():
                if key not in agg_data.keys():
                    agg_data[key] = []
                agg_data[key].append(val)

    # TODO: move convert_to_easy_read and bio_to_easy_read to crf_human_evals.py
    out_name = os.path.basename(args.in_file).split("_")[0]+"_postproc"
    with jsonlines.open(os.path.join(args.out_dir, out_name+".jsonl"), "w") as f:
        # open(os.path.join(args.out_dir, "test_postproc_easy_read.txt"), "w") as ef:
        for d in convert_to_easy_read(agg_data, args.idx_to_tag_file, args.idx_to_label_file, args.batched):
            f.write(d)
            # ef.write("\t".join([d["sent_labels"], d["easy_read"]+"\n"]))

        print(f"Results saved to {os.path.join(args.out_dir, out_name+'.jsonl')}")
    
    # if args.check_mem:
    #     print(f"gold_has_match: {gold_has_match}")
    #     print(f"has_match: {has_match}")
    #     print(f"pred_seq_is_in_patterns: {pred_seq_is_in_patterns} / {has_match}")
    #     print(f"total: {total}")
