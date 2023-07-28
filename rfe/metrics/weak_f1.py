# Borrowed from https://github.com/DFKI-NLP/sam/blob/main/sam/metrics/weak_span_based_f1_measure.py
from typing import List, Optional, Callable, Tuple, Set

import torch

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    bmes_tags_to_spans,
    TypedStringSpan,
)


TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


def get_overlap_len(indices_1: Tuple[int,int], indices_2: Tuple[int,int]) -> int:
    if indices_1[0] > indices_2[0]:
        tmp = indices_1
        indices_1 = indices_2
        indices_2 = tmp
    if indices_1[1] <= indices_2[0]:
        return 0
    return min(indices_1[1] - indices_2[0], indices_2[1] - indices_2[0])


def has_weak_overlap(indices_1: Tuple[int,int], indices_2: Tuple[int,int]) -> bool:
    # checks if overlap in span is at least half of the length of the shorter span
    min_len = min(indices_1[1]-indices_1[0], indices_2[1]-indices_2[0])
    overlap_len = get_overlap_len(indices_1, indices_2)
    return 2*overlap_len >= min_len


def increase_span_end_index(span: Tuple[str, Tuple[int, int]], offset: int) -> Tuple[str, Tuple[int, int]]:
    # format of span is (label,(start,end))
    return span[0], (span[1][0], span[1][1]+offset)


def get_weak_match(
        span: Tuple[str, Tuple[int, int]],
        gold_spans: List[Tuple[str, Tuple[int, int]]],
        inclusive_end_index: bool = False
) -> Tuple[str, Tuple[int, int]]:
    """
    This method checks if the predicted span is weakly matched with any of the gold spans. If predicted type and gold
    type matches then we check if their respective indices are weakly overlapping or not. Weak overlap between gold and
    predicted span is defined in Lauscher et al. (2018) as overlap which should be at least half of the length of
    shorter span. If they are weakly overlapping as well then we return the matched span. 
    
    In addition to this, we use `inclusive_end_index` boolean which, if set, adds an offset to the end index of each span 
    in the gold spans list and to the predicted span. Once a match is found, we revert back changes to the end index of the 
    matched span. 
    The reason why we add an offset to the end index is that an AllenNLP span containing a single token has length 0 (because 
    its start and end indices are the same)

    :param span: Predicted span instance as a tuple with span label and indices(start and end) of span.
    :param gold_spans: List of gold span instances as tuple with span label and indices(start and end) of span.
    :param inclusive_end_index: if set adds an offset to the end index of each span in gold spans list and also to
    predicted span. Once a match is found we revert back changes to end index of matched span.
    :return: gold span instance if matched with predicted span instance else None
    """
    if inclusive_end_index:
        span = increase_span_end_index(span, offset=1)
        gold_spans = [increase_span_end_index(gold_span, offset=1) for gold_span in gold_spans]

    match_found = None
    predicted_type, predicted_indices = span
    for gold_type, gold_indices in gold_spans:
        if predicted_type == gold_type and has_weak_overlap(predicted_indices, gold_indices):
            match_found = gold_type, gold_indices
            if inclusive_end_index:
                match_found = increase_span_end_index(match_found, offset=-1)
            break
    return match_found


@Metric.register("span_f1_weak")
class SpanBasedF1WeakMeasure(SpanBasedF1Measure):
    """
    This class provides same functionality as SpanBasedF1Measure in the case of weak=False. If weak is true (default)
    then matches are calculated in relaxed manner: spans have to match only in type and have a certain overlap (by at
    least half of the length of the shorter of them).
    """

    def __init__(self, vocabulary: Vocabulary, weak: bool = True, **kwargs) -> None:
        super().__init__(vocabulary=vocabulary, **kwargs)
        self._weak = weak

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        prediction_map: Optional[torch.Tensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        prediction_map : `torch.Tensor`, optional (default = `None`).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            `vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))`,
            rather than simply `vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))`.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(gold_labels).bool()

        predictions, gold_labels, mask, prediction_map = self.detach_tensors(
            predictions, gold_labels, mask, prediction_map
        )

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to SpanBasedF1Measure contains an "
                "id >= {}, the number of classes.".format(num_classes)
            )

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue

            predicted_string_labels = [
                self._label_vocabulary[label_id]
                for label_id in sequence_prediction[:length].tolist()
            ]
            gold_string_labels = [
                self._label_vocabulary[label_id]
                for label_id in sequence_gold_label[:length].tolist()
            ]

            tags_to_spans_function: TAGS_TO_SPANS_FUNCTION_TYPE
            # `label_encoding` is empty and `tags_to_spans_function` is provided.
            if self._label_encoding is None and self._tags_to_spans_function:
                tags_to_spans_function = self._tags_to_spans_function
            # Search by `label_encoding`.
            elif self._label_encoding == "BIO":
                tags_to_spans_function = bio_tags_to_spans
            elif self._label_encoding == "IOB1":
                tags_to_spans_function = iob1_tags_to_spans
            elif self._label_encoding == "BIOUL":
                tags_to_spans_function = bioul_tags_to_spans
            elif self._label_encoding == "BMES":
                tags_to_spans_function = bmes_tags_to_spans
            else:
                raise ValueError(f"Unexpected label encoding scheme '{self._label_encoding}'")

            predicted_spans = tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            gold_spans = tags_to_spans_function(gold_string_labels, self._ignore_classes)

            predicted_spans = self._handle_continued_spans(predicted_spans)
            gold_spans = self._handle_continued_spans(gold_spans)

            # Sorting spans so that it is deterministic all the time (handle_continued_spans may not maintain the order)
            predicted_spans = sorted(predicted_spans)
            gold_spans = sorted(gold_spans)

            for span in predicted_spans:
                span_original = span
                if self._weak:
                    span = get_weak_match(span, gold_spans, inclusive_end_index=True)
                if (not self._weak and span in gold_spans) or (self._weak and span):
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    if self._weak:
                        span = span_original
                    self._false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`
            - tp (true positives) : 'int'
            - fp (false positives) : 'int'
            - fn (false negatives) : 'int'
            Additionally, an `overall` key is included, which provides the precision,
            recall and f1-measure for all spans.
        """
        if is_distributed():
            raise RuntimeError(
                "Distributed aggregation for SpanBasedF1Measure is currently not supported."
            )
        all_tags: Set[str] = set() # a set() variable; the ":Set[str]" is a variable annotation to increase readability; see https://peps.python.org/pep-0526/
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            # In order to get counts of tp,fn and fp in output metric we create tp_key, fp_key, fn_key and then add it
            # to all_metrics
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            tp_key = "tp" + "-" + tag
            fp_key = "fp" + "-" + tag
            fn_key = "fn" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure
            all_metrics[tp_key] = self._true_positives[tag]
            all_metrics[fp_key] = self._false_positives[tag]
            all_metrics[fn_key] = self._false_negatives[tag]

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics
