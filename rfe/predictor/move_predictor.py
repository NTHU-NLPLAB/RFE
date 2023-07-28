import torch
from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from typing import List, Dict, Iterable, cast
from scipy.stats import norm
from overrides import overrides

from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, FBetaMeasure, F1Measure
from sklearn.metrics import recall_score, precision_score, f1_score
from ..dataset_reader import ExampleReader, TagAlignError

from rfe.utils import LABEL_TO_INDEX


@Predictor.register('move_predictor')
class MovePredictor(Predictor):
    def __init__(
            self,
            *args,
            use_obj_store: bool = False,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """Outputs the predictions to file (non-batch version)
        """
        
        output_dict = self.predict_probs(inputs)
        return {**inputs,
                "sent_probs": output_dict['sent_probs'],
                "pred_sent_labels": output_dict['pred_sent_labels'],
                "seq_tag_probs": output_dict['seq_tag_probs'],
                "pred_seq_tags": output_dict["pred_seq_tags"]
                }

    def predict_probs(self, inputs: JsonDict):
        """Gets prediction from model (non-batch version)
        Ref: https://github.com/allenai/allennlp/blob/main/allennlp/predictors/predictor.py
        """
        # called after `predict_batch_json`
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        # predict_instance(): applies token indexer to instance, calls forward() on instance
        return output_dict


    @overrides
    def predict_batch_json(self, inputs: List) -> List[JsonDict]:
        """Takes `_batch_json_to_instances` returned outputs and feeds them into
        model, gets model outputs, returns sent_probs, seq_tag_probs, & text
        """
        # https://github.com/allenai/allennlp/blob/main/allennlp/predictors/predictor.py#L300
        # inputs: batch x (num max sents) x hidden
        instances = self._batch_json_to_instances(inputs)
        # _batch_json_to_instances() calls _json_to_instance for each item in the batch
        
        output_dicts = self.predict_batch_instance(instances)
        # predict_batch_instance(): apply_token_indexer() (which is just `pass` in the source code) to each instance,
        # then calls forward(), and then returns output dicts: {'probs': [...], 'loss': ..}

        results = []
        for inp, output_dict in zip(inputs, output_dicts):
            sent_probs = output_dict['sent_probs']
            seq_tag_probs = output_dict['seq_tag_probs']
            pred_sent_labels = output_dict['pred_sent_labels']
            pred_seq_tags = output_dict['pred_seq_tags']
            results.append({**inp,
                            'sent_probs': sent_probs,
                            'seq_tag_probs': seq_tag_probs,
                            "pred_sent_labels": pred_sent_labels,
                            "pred_seq_tags": pred_seq_tags})
        return results

    #@overrides
    def _json_to_instance(self, json_dict: Iterable[Dict]) -> Instance:
        """Reads test examples from file and turns into AllenNLP instance
        """
        # json_dict is actually not a json dict, just a python dict but @overrides require the parameter name
        # of the submethod be the same as the name in the overridden method so there you go.
        if isinstance(json_dict, dict):
            texts = [json_dict["text"]]
            seq_labels = json_dict.get("seq_tag", None) # for classifying weak supervision data
            if seq_labels is not None: 
                seq_labels = [seq_labels]
        elif isinstance(json_dict, Iterable):
            texts = [sent['text'] for sent in json_dict] # a list of sentences
        return self._dataset_reader.example_to_instance(texts=texts, sent_labels=None, seq_labels=seq_labels)

    #@overrides
    def _batch_json_to_instances(self, json_dicts:  List[Iterable[Dict]]) -> List[Instance]:
        """Batch version of `_json_to_instance`
        """
        # `json_dicts` is a batch (i.e. list) of paragraphs
        # using the dumb name because `@overrides` requires it to be the same as the overridden method
        instances = []
        for paragraph in json_dicts:
            try:
                instance = self._json_to_instance(paragraph)
                instances.append(instance)
            except TagAlignError as e:
                continue
        return instances
