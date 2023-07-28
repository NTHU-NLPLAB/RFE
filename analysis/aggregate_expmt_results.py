import os
import json, _jsonnet
import pandas as pd

#from ..classifier.metrics.normalize_span_f1 import flatten_dict

from collections import defaultdict

EXPMTS=[
    # "az-no-seq-tag-lr-5e-4",
    # "az-no-seq-tag-lr-5e-5",
    # "az-simple-seq-tag-no-crf-lr-5e-4",
    # "az-simple-seq-tag-no-crf-lr-5e-5",
    # "az_class_matched_bio-lr-5e-4",
    # "az_class_matched_bio_strict-lr-5e-4",
    ## "az-clf-balanced-over-lr-5e-4-labsm-0.3-uncased", # bert, multitask
    ## "az-no-seq-tag-balanced-labsm-0.3-uncased", # bert, single-task
    # "az-clf-balanced-over+sub",
    # "az-clf-balanced-over-lr-1e-3",
    # "az+s2-clf",
    ## "az-clf-balanced-scibert-frozen-labsmooth-0.3", # scibert, multitask
    ## "az-no-seq-tag-balanced-scibert-labsm-0.3", # scibert, single-task
    #"az-clf-no-seq-tag-merged-balanced-scibert", # scibert-single-task-merged
    # "crf-tagger",
    # "crf-tagger-balanced",
    # "crf-tagger-balanced-o",
    # "crf-tagger-class-bio",
    # "crf-tagger-class-bio-strict",
    # "crf-masked-final-token",
    # "crf-tagger-scibert",
    ## "crf-tagger-scibert-balanced",
    # "crf-tagger-class-bio-strict-scibert",
    # "crf-tagger-class-bio-strict-scibert-balanced",
    ## "crf-tagger-merged-balanced",
    ## "crf-tagger-merged-balanced-scibert" # scibert-crf-merged
    "crf-tagger-balanced-scibert-data-10062023",
    ] # folder name
results = defaultdict(lambda: dict())
EXCLUDE_ROWS=[
    "dataset_reader.do_seq_labelling",
    "model.text_field_embedder.token_embedders.tokens.type",
    "model.text_field_embedder.token_embedders.tokens.tokenizer_kwargs.return_offsets_mapping",
    "model.text_field_embedder.token_embedders.tokens.tokenizer_kwargs.additional_special_tokens",
    "model.text_field_embedder.token_embedders.tokens.model_name",
    "dataset_reader.tokenizer.tokenizer_kwargs.return_offsets_mapping",
    "validation_data_path",
    "train_data_path",
    "model.verbose_metrics",
    "model.seq2seq_encoder.bidirectional",
    "model.embedder.token_embedders.tokens.tokenizer_kwargs.return_offsets_mapping",
    "model.embedder.token_embedders.tokens.tokenizer_kwargs.additional_special_tokens",
    "model.embedder.token_embedders.tokens.train_parameters",
    "model.embedder.token_embedders.tokens.model_name",
    "model.embedder.token_embedders.tokens.type",
    "dataset_reader.type",
    "dataset_reader.tokenizer.type",
    "dataset_reader.tokenizer.tokenizer_kwargs.additional_special_tokens",
    "dataset_reader.text_token_indexers.tokens.type",
    "dataset_reader.text_token_indexers.tokens.tokenizer_kwargs.return_offsets_mapping",
    "dataset_reader.text_token_indexers.tokens.tokenizer_kwargs.additional_special_tokens",
    "dataset_reader.text_token_indexers.tokens.model_name",
    "model.text_field_embedder.token_embedders.tokens.train_parameters",
    "paragraph accuracy",
    "sequence tagging accuracy",
    "sequence tagging accuracy no O",
    "model.checkpoint",
    "model.dataset",
    "model.init_clf_weights",
    "data_loader.shuffle"
    ]


def flatten_dict(d, _keys=()):
    if not isinstance(d, dict):
        return d
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_children = flatten_dict(v, _keys=_keys+(k,))
            res.update(flat_children)
        else:
            res[_keys+(k,)] = v
    return res

def join_keys(d):
    new_d = {}
    for k, v in d.items():
        new_k = ".".join(k)
        new_d[new_k] = v
    return new_d

for expmt in EXPMTS:
    with open(os.path.join("..", "experiments", expmt, "test_eval_metrics.json")) as f:
        res = json.load(f)
        for metric, val in res.items():
            if metric not in EXCLUDE_ROWS:
                results[expmt][metric] = val
    
    config_path = os.path.join("..", "experiments", expmt, "config.jsonnet")
    configs = json.loads(_jsonnet.evaluate_file(config_path))
    configs = join_keys(flatten_dict(configs))
    
    for k, v in configs.items():
        if k == "model.text_field_embedder.token_embedders.tokens.train_parameters":
            results[expmt]["token_embedders.train_parameters"] = v
        elif k not in EXCLUDE_ROWS:
            results[expmt][k] = v

        


df = pd.DataFrame.from_dict(results)
print(df.to_string())
df.to_csv("analysis.csv")

