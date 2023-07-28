# Modified from
# https://github.com/DFKI-NLP/sam/blob/main/sam/models/crf_tagger_with_f1.py

from typing import Dict, Optional, List, Any, cast
import logging

#from overrides import overrides
import wandb
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import ListField
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, FBetaMeasure, F1Measure

from ..metrics.weak_f1 import SpanBasedF1WeakMeasure
from ..metrics._span_based_f1_measure import MaskedSpanBasedF1Measure
from ..metrics.span_utils import bio_tags_to_spans
from ..metrics.normalize_span_f1 import normalize_span_f1_result, flatten_dict
from .move_tag_model import DEFAULT_TAG_NAMESPACE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
wandb.login()

@Model.register("crf_tagger")
class CrfTaggerWithF1(Model):
    """
    The `CrfTagger` encodes a sequence of text with a `Seq2SeqEncoder`,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    Registered as a `Model` with name "crf_tagger".
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the tokens `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : `str`, optional (default=`labels`)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : `FeedForward`, optional, (default = `None`).
        An optional feedforward layer to apply after the encoder.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` or `constrain_crf_decoding` is true.
    include_start_end_transitions : `bool`, optional (default=`True`)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : `bool`, optional (default=`None`)
        If `True`, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    dropout:  `float`, optional (default=`None`)
        Dropout probability.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    top_k : `int`, optional (default=`1`)
        If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
        Top k parses are returned as a list of dicts, where each dictionary is of the form:
        {"tags": List, "score": float}.
        The ("tags)" value for the first dict in the list for each data_item will be the top
        choice, and will equal the corresponding item in output_dict['tags']
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        label_namespace: str = DEFAULT_TAG_NAMESPACE,
        feedforward: Optional[FeedForward] = None,
        label_encoding: Optional[str] = None,
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = None,
        calculate_span_f1: bool = None,
        calculate_weak_span_f1: bool = True,
        dropout: Optional[float] = None,
        verbose_metrics: bool = False,
        exclude_none_labels_from_token_f1_metrics: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        top_k: int = 1,
        mask_sep_token: bool = None,
        ignore_loss_on_o_tags: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = seq2seq_encoder
        self.top_k = top_k
        self.ignore_loss_on_o_tags = ignore_loss_on_o_tags
        self.mask_sep_token = mask_sep_token
        self._verbose_metrics = verbose_metrics
        self.exclude_none_labels_from_token_f1_metrics = exclude_none_labels_from_token_f1_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))
        
        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )

        wandb.init(
            # Set the project where this run will be logged
            project="crf-tagger",
            # Track hyperparameters and run metadata
            config={
                "num_tags": self.num_tags,
                "encoder": self.encoder,
                "dropout": dropout,
                }
            )

        self.metrics = {
            "token/accuracy": CategoricalAccuracy(),
            "token/accuracy3": CategoricalAccuracy(top_k=3),
            "token/overall-micro": FBetaMeasure(
                beta=1, average='micro',
                # exclude NONE labels
                labels=[k for k, v in vocab.get_index_to_token_vocabulary(namespace=self.label_namespace).items()
                        if v != "O" or not exclude_none_labels_from_token_f1_metrics
                        ]
            ),
            "token/overall-macro": FBetaMeasure(
                beta=1, average='macro',
                # exclude NONE labels
                labels=[k for k, v in vocab.get_index_to_token_vocabulary(namespace=self.label_namespace).items()
                        if v != "O" or not exclude_none_labels_from_token_f1_metrics
                        ]
            )
        }

        for k, v in vocab.get_index_to_token_vocabulary(namespace=self.label_namespace).items():
            self.metrics[f'token/{v}'] = F1Measure(k)
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab,
                tag_namespace=label_namespace,
                #label_encoding=label_encoding
                label_encoding=None,
                tags_to_spans_function=bio_tags_to_spans
            )
            if self.mask_sep_token:
                self._f1_metric = MaskedSpanBasedF1Measure(
                    vocab,
                    tag_namespace=label_namespace,
                    #label_encoding=label_encoding
                    label_encoding=None,
                    tags_to_spans_function=bio_tags_to_spans
                )
        self.calculate_weak_span_f1 = calculate_weak_span_f1
        if self.calculate_weak_span_f1:
            self._weak_f1_metric = SpanBasedF1WeakMeasure(
                vocab,
                tag_namespace=label_namespace,
                #label_encoding=label_encoding,
                label_encoding=None,
                tags_to_spans_function=bio_tags_to_spans,
                weak=True
            )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            seq2seq_encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        if feedforward is not None:
            check_dimensions_match(
                seq2seq_encoder.get_output_dim(),
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )
        initializer(self)

    def forward(
        self,  # type: ignore
        paragraph: ListField, # takes Instance output from example_reader
        sent_labels: torch.Tensor = None,
        seq_tags: torch.Tensor = None,
        orig_tag_indices: torch.Tensor = None,
        ignore_loss_on_o_tags: bool = False,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        seq_tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        ignore_loss_on_o_tags : `bool`, optional (default = `False`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.
        
        # Returns
        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.BoolTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """
        ignore_loss_on_o_tags = self.ignore_loss_on_o_tags
            
        assert 'tokens' in paragraph.keys()
        text_field = paragraph['tokens']
        masks = text_field['mask'].transpose(1, 0) # batch x max_parag_len x sent_len => max_parag_len x batch x sent_len
        token_ids = text_field['token_ids'].transpose(1, 0)
        type_ids = text_field['type_ids'].transpose(1, 0)
        
        # iterate through each sent in a parag
        # We're no longer using multi-sentence parags, so we'll just
        # take the 1st element
        tokens = token_ids[0]
        mask = masks[0]

        type_id = type_ids[0]
        if seq_tags is not None:
            seq_tags = seq_tags.transpose(1, 0)[0] # batch x parag_len x seq_len => parag_len x batch x seq_len => batch x seq_len

        sent_inst = {
                "tokens": {
                    "mask": mask,
                    "token_ids": tokens,
                    "type_ids": type_id,
                    }
                }
        output = {}
        if seq_tags is None:
            # When predicting, the predictor (https://github.com/allenai/allennlp/blob/main/allennlp/predictors/predictor.py#L300)
            # calls model.forward_on_instance()
            # https://github.com/allenai/allennlp/blob/v2.9.3/allennlp/models/model.py#L193
            # which checks len and shape of output_dict. If len or shape != batch, the dict will be discarded.
            # This means output['paragraph'] won't go through the predictor and out into the predicted outputs.
            # We however need the token_ids to be able to realign them to the tags.
            # Solution: put token_ids into paragraph keys and skip the rest
            # (we still use 'paragraph' as key name to stay compatible with postprocess_span)
            output['paragraph'] = tokens

        embedded_text_input = self.text_field_embedder(sent_inst) # batch x hidden

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input) # sci-bert

        encoded_text = self.encoder(embedded_text_input, mask) # the lstm

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text) # batch x seq_len x num_tags (128, 80, 16)
        # mask shape = (batch x seq_len)
        last_tag_index = mask.sum(1).long() - 1 # sum along sequence
        SEP_token_idx = self.vocab.get_token_index("[SEP]", namespace="tags")
        logger.debug("token_ids", tokens.shape)
        logger.debug("last_tag_index", last_tag_index.shape)

        last_tokens = tokens.gather(1, last_tag_index.unsqueeze(1)) # unsqueezed last_tag_index: batch x 1
        if (self.mask_sep_token) and (sum(last_tokens) == mask.size(0) * SEP_token_idx): # we get batch_size x 103
            # last_tag_index = torch.sum(mask, dim=1) # sent len of each element in batch
            mask[range(mask.shape[0]), last_tag_index] = False # mask the [SEP] token
            logger.debug("Set final token (the SEP token) of mask to False")
            mask[:,0] = False # mask the [CLS] token too ; mask: batch x seq_len 

        best_paths = self.crf.viterbi_tags(logits, mask, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        output["logits"] = logits
        output["mask"] = mask
        output["pred_tags"] = predicted_tags
        if seq_tags is None:
            assert "paragraph" in output.keys()
        
        if self.top_k > 1:
            output["top_k_tags"] = best_paths

        if seq_tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                crf_mask = mask & (seq_tags != o_tag_index)
            else:
                crf_mask = mask
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, seq_tags, crf_mask)

            output["loss"] = -log_likelihood
            wandb.log({"loss": output["loss"]})

            ####################################################################################
            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, seq_tags, mask)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, seq_tags, mask)
            if self.calculate_weak_span_f1:
                self._weak_f1_metric(class_probabilities, seq_tags, mask)
            wandb.log(self.get_metrics(reset=False))
        return output
        ####################################################################################

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"pred_tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]

        output_dict["pred_tags"] = [decode_tags(t) for t in output_dict["pred_tags"]]

        if "top_k_tags" in output_dict:
            output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            metrics_to_return['span'] = normalize_span_f1_result(f1_dict)
        if self.calculate_weak_span_f1:
            f1_dict = self._weak_f1_metric.get_metric(reset=reset)
            metrics_to_return['span_weak'] = normalize_span_f1_result(f1_dict)
        metrics_to_return = flatten_dict(metrics_to_return)
        metrics_to_return = {'/'.join(x): y for x, y in metrics_to_return.items()}
        if not self._verbose_metrics:
            metrics_to_return = {x: y for x, y in metrics_to_return.items() if "overall" in x or 'acc' in x}

        return metrics_to_return
