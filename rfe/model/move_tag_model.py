from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import ListField
from allennlp.nn import util
from allennlp.common import FromParams, Params
from allennlp.models import Model
from allennlp.data.fields import TextField
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed
from allennlp.training.metrics import SequenceAccuracy, FBetaMeasure, F1Measure, SpanBasedF1Measure
from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.common.checks import check_dimensions_match
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

import json
import logging
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .move_classifier import MoveClassifier
from ..metrics.weak_f1 import SpanBasedF1WeakMeasure
from ..metrics.span_utils import bio_tags_to_spans
from ..metrics.normalize_span_f1 import normalize_span_f1_result, flatten_dict
from typing import Dict, Optional
from ..dataset_reader.example_reader import DEFAULT_LABEL_NAMESPACE, DEFAULT_TAG_NAMESPACE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
wandb.login()


@Model.register('move_tag_classifier')
class MoveTagClassifier(MoveClassifier):
    """
    Difference to MoveClassifierBase:
    1. Added sequence tags for each sent, learned as an torch.nn.Embedding and concat'ed w text embs
    2. Removed positional encoding & bert booling for sent embs
    2.1. Added LSTM to learn relation btwn sent tokens and seq tags instead of pos enc.
    2.2. Pooling changed from bert CLS token to CNN

    ## Parameters
    vocab: Vocabulary,
    embedder: `TextFieldEmbedder` for turning input text into TextField embeddings
    pooler: the [CLS] token of (BERT) transformers to pool encoder sentence outputs 
        from max_seq_len x hidden into size of 1 x hidden
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text. The LSTM for the concat'ed tags+text embs.
    pooler : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. 
        If `seq2seq_encoder` is provided, this encoder will pool its output. (the CNN)
        Otherwise, this encoder will operate directly on the output of the 
        `text_field_embedder`.
    _checkpoint: str, path to trained weights
    dataset: 'az', 'pubmed'
    init_clf_weights: str, {None, 'zeros', 'rand'}, ways to init the Linear classifier layer
        zeros: init with 0s
        rand: init with rands (TODO: specify rand range (in normal dist?))
        copy: copy from loaded weights, discard extras if input num_labels is larger than current;
                fill in 0s if fewer
        (See `load_weights()` for details)
    """
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 seq2seq_encoder: Seq2SeqEncoder = None, # the LSTM
                 pooler: Seq2VecEncoder = None, # the pooler after LSTM
                 tag_embedding_size: Optional[int] = None, # hidden size
                 do_seq_labelling: bool = None,
                 final_dropout: float = None,
                 checkpoint: str = None,
                 dataset: str = 'az',
                 class_weighting: bool = None,
                 label_smoothing: float = None,
                 init_clf_weights: str = None,
                 verbose_metrics: bool = True,
                 joint_loss_lambda: float = None, # sum classifier loss with seq_tagging loss or not; if not, use classifier loss only
                 ):
        super().__init__(vocab, embedder, pooler, checkpoint, dataset, init_clf_weights)
        self.num_tags = len(vocab.get_token_to_index_vocabulary(namespace=DEFAULT_TAG_NAMESPACE)) if self.dataset == "az" else None
        self.num_labels = len(vocab.get_token_to_index_vocabulary(namespace=DEFAULT_LABEL_NAMESPACE))
        self.do_seq_labelling = do_seq_labelling
        self.tag_embedding_size = tag_embedding_size # the vocab size of seq tags. For creating tag embs.
        self.seq2seq_encoder = seq2seq_encoder # the lstm after cat(tag_emb, sent_emb)
        self.pooler = pooler # overrides MoveClassifier pooler.
        self.clasifier = nn.Linear(self.pooler.get_output_dim(), self.num_labels) # sentence clf
        self.final_dropout = nn.Dropout(final_dropout)
        self.joint_loss_lambda = joint_loss_lambda
        self.label_smoothing = label_smoothing
        self._verbose_metrics = verbose_metrics

        self.tag_embedding, self.tag_embedding_proj = None, None
        sent_labels = list(vocab.get_token_to_index_vocabulary(namespace=DEFAULT_LABEL_NAMESPACE))
        logger.info(f'{sent_labels})')

        self.class_weighting = class_weighting
        if class_weighting is not None:
            with open('class_counts.json') as f:
                class_counts = json.load(f)
                total = sum(class_counts.values())
                class_weights = [total / (class_counts[k] * len(sent_labels)) for i, k in enumerate(sent_labels)]
                self.class_weights = torch.Tensor(class_weights).float().to(torch.cuda.current_device())
                logger.info(f"Using class weights {self.class_weights}")
            
        if self.do_seq_labelling:
            if self.tag_embedding_size is None:
                self.tag_embedding_size = 128
            self.tag_embedding = nn.Embedding(num_embeddings=self.num_tags, 
                                                embedding_dim=self.tag_embedding_size) # emb_dim = hidden size.
            # torch Embedding takes input of whatever shape (e.g. batch x seq_len)=> outputs input_shape x embedding_dim
            logger.info(f'use tag embedding of size: {self.num_tags} x {self.tag_embedding_size} '
                            f'(tags: {list(vocab.get_token_to_index_vocabulary(namespace=DEFAULT_TAG_NAMESPACE))})')
            
            # nn.Linear takes shape (*, in_feats), outputs (*, out_feats)
            # takes LSTM outputs;
            # reshape emb (Linear expects an 2D input) => Linear => reshape back into batch x seq_len x num_labels
            self.tag_embedding_proj = TimeDistributed(nn.Linear( # projection / fully-connected layer for tag clf
                in_features=self.seq2seq_encoder.get_output_dim(),
                out_features=self.num_tags
                )
            )

        if self.checkpoint:
            self.load_weights(self.checkpoint)

        #################### metrics #####################
        if self.do_seq_labelling:
            self.strict_tag_f1 = SpanBasedF1Measure(
                vocab,
                tag_namespace=DEFAULT_TAG_NAMESPACE,
                #label_encoding='BIO',
                label_encoding=None,
                tags_to_spans_function=bio_tags_to_spans # import bio_tags_to_spans from ..metrics.span_utils if needed
            )
            self.weak_tag_f1 = SpanBasedF1WeakMeasure(
                vocab,
                tag_namespace=DEFAULT_TAG_NAMESPACE,
                #label_encoding='BIO',
                label_encoding=None,
                tags_to_spans_function=bio_tags_to_spans,
                weak=True,
            )

        self.seq_tag_accuracy = SequenceAccuracy()
        self.sent_f1s = {
            self.vocab.get_token_from_index(idx, DEFAULT_LABEL_NAMESPACE): F1Measure(positive_label=idx) \
                         for idx in range(self.num_labels)
                         }
        
        self.sent_f1_overall = FBetaMeasure(beta=1.0, average='weighted', labels=list(range(self.num_labels)))
        
        # WandB monitoring
        wandb.init(
            # Set the project where this run will be logged
            project="move_tag_classifier",
            # Track hyperparameters and run metadata
            config={
                "num_tags": self.num_tags,
                "num_labels": self.num_labels,
                "do_seq_labelling": self.do_seq_labelling,
                "tag_embedding_size": self.tag_embedding_size,
                "seq2seq_encoder": self.seq2seq_encoder,
                "pooler": self.pooler,
                "joint_loss_lambda": self.joint_loss_lambda,
                }
            )

        
    def forward(self,
                paragraph: ListField, # takes Instance output from example_reader
                sent_labels: torch.Tensor = None,
                seq_tags: torch.Tensor = None,
                orig_tag_indices: torch.Tensor = None,
                metadata = None,
                ) -> Dict[str, torch.Tensor]:
        """
        paragraph: {'tokens': {'token_ids': [...], 'mask': [...], 'type_ids': [...]},
        values of token_ids, mask, type_ids have shape (max num sents, max sent length) x batch
        
        Note for dev:
        https://guide.allennlp.org/building-your-model#1
        The input/output spec of Model.forward() is somewhat more strictly defined than that of PyTorch modules. 
        Its parameters need to match field names in your data code exactly.
        https://guide.allennlp.org/reading-data#1
        In the dataset_reader, the fields names are important—because the resulting dictionary of tensors 
        is passed by name to the model, they have to match the model's forward() arguments exactly.

        Also, whatever's passed into `forward()` will get saved into output_dict 
        (so here output_dict["paragraph"] = parag instance as pytorch tensor; output_dict["sent_labels"] = sent_labels etc.), 
        we don't have to do it manually ourselves. We can thus also save information needed in postprocessing this way.
        """
        # print(len(paragraph)) # 1 (corresponds to using 1 token indexer, using namespace 'tokens')
        # print(sent_labels.shape) # batch x (max_parag_sent)
        
        assert len(paragraph.keys()) == 1 and 'tokens' in paragraph.keys()
        text_field = paragraph['tokens']
        masks = text_field['mask'].transpose(1, 0) # batch x max_parag_len x sent_len => max_parag_len x batch x sent_len
        token_ids = text_field['token_ids'].transpose(1, 0)
        type_ids = text_field['type_ids'].transpose(1, 0)
        if self.do_seq_labelling and seq_tags is not None:
            seq_tags = seq_tags.transpose(1, 0) # batch x parag_len x seq_len => parag_len x batch x seq_len
        
        # iterate through each sent in a parag
        output = {}
        parag_sent_probs = []
        parag_sent_preds = []
        parag_sent_tag_probs = []
        parag_sent_tag_preds = []
        seq_losses = []

        for sent_idx, sent in enumerate(token_ids): # token_ids = 1 parag
            token_id = sent # batch x seq_len
            mask = masks[sent_idx] # same as util.get_text_field_mask(sent_inst)
            type_id = type_ids[sent_idx]
            sent_inst = {
                    "tokens": {
                        "mask": mask,
                        "token_ids": token_id,
                        "type_ids": type_id,
                        }
                    }
            token_embeds = self.embedder(sent_inst) # batch x hidden
            sent_embedding = [token_embeds] # batch x hidden x 1
            
            if self.do_seq_labelling and seq_tags is not None:
                seq_tag_encoding = self.tag_embedding(seq_tags[sent_idx]) # batch x sent_len (x hidden??)
                sent_embedding.append(seq_tag_encoding) # batch x hidden x 2
            sent_embedding = torch.cat(sent_embedding, dim=-1) # embedding of one single sentence!
            logger.debug(f"Initial sent emb shape: {sent_embedding.shape}")
            output = {}
            # seq2seq_encoder accepts (batch_size, seq_length, input_dim),
            # and outputs (batch_size, seq_length, output_dim); output_dim = lstm hidden size
            if self.seq2seq_encoder:
                sent_embedding = self.seq2seq_encoder(sent_embedding, mask=mask) # hopefully learns relation betwn tokens and seq tags
                logger.info(f"After LSTM emb shape: {sent_embedding.shape}")
            
            #################### seq tagging evals ####################
            if self.do_seq_labelling:
                tag_emb_proj = self.tag_embedding_proj(sent_embedding) # Project sent's seq_tag_emb into batch x seq_len x num_labels => take preds
                tag_preds = torch.argmax(tag_emb_proj, dim=-1) # batch x seq_len

                if seq_tags is not None: # is training
                    tag_emb_proj = tag_emb_proj.transpose(1, 2) # cross entropy takes  (batch x num_classes (x other dims))
                    seq_tags = seq_tags[sent_idx] # batch x seq_len

                    tag_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, namespace=DEFAULT_TAG_NAMESPACE)
                    loss = F.cross_entropy(tag_emb_proj, seq_tags, ignore_index=tag_pad_index)
                    seq_losses.append(loss) # batch x seq_len

                    # SequenceAccuracy: predictions(batch, num_classes, seq_len) ; gold_labels: (batch, seq_len).
                    # mask: batch x seq_len
                    mask_padded = mask & (seq_tags != tag_pad_index)

                    self.seq_tag_accuracy(tag_emb_proj, seq_tags, mask_padded)

                    # SpanBasedF1Measure: predictions (batch, seq_len, num_classes); gold_labels: (batch, seq_len).
                    tag_emb_proj = tag_emb_proj.transpose(1, 2)
                    
                    # TODO: how to make allennlp's spanBasedF1Measure only look at non-padded tokens? 
                    # Since SpanBasedF1Measure checks the gold tags (if beginning w anything other than BIO it raises errors)
                    # https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py#L108
                    # (the mask passed into spanBasedF1Measure is only used to calculate a superficial sequence length, not helpful!)
                    # cf source code: https://github.com/allenai/allennlp/blob/main/allennlp/training/metrics/span_based_f1_measure.py#L158

                    # one way to do it: do the mask-filtering myself before sending it in SpanBasedF1Measure
                    # another way: evaluate with every tensor[1:]
                    # another way is to break the source code and make it not check & raise the error...which woudl not be the optimal way

                    self.strict_tag_f1(tag_emb_proj, seq_tags, mask) #mask_padded)
                    self.weak_tag_f1(tag_emb_proj, seq_tags, mask) #mask_padded)
                # reshape & save
                tag_probs = F.softmax(tag_emb_proj, dim=-1) # tag_emb_proj: batch, seq_len, num_classes
                
                parag_sent_tag_probs.append(tag_probs) 
                parag_sent_tag_preds.append(tag_preds)
            ################## seq tagging evals ends ##################        
            
            # pooler accepts (batch_size, seq_len, input_dim),
            # outputs (batch_size, output_dim), here output_dim == hidden
            sent_embedding = self.pooler(sent_embedding, mask=mask)
            logger.debug(f"Pooled emb shape: {sent_embedding.shape}") # should be batch x hidden

            if self.final_dropout:
                sent_embedding = self.final_dropout(sent_embedding)

            # calculate loss & probs the sent
            logits = self.classifier(sent_embedding) # in: batch x hidden; out: batch x num_labels
            logger.debug(f"Logits shape: {logits.shape}")
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1) # TODO: check
            logger.debug(f"Sent pred label shape = {pred_label.shape}")
            parag_sent_probs.append(probs) # logits for this parag
            parag_sent_preds.append(pred_label)

        if sent_labels is not None: #i.e. during training
            losses = []
            # parag_sent_probs = List[ (batch x num_labels) * max_parag_len]
            # sent_labels: (batch x parag_len)
            sent_labels = sent_labels.transpose(0, 1)
            for parag_logits, label in zip(parag_sent_probs, sent_labels): # label that correspond to each sentence
                logger.debug(f"logits shape: {parag_logits.shape}")
                logger.debug(f"sent_labels shape: {label.shape}")
                assert not torch.any(torch.isnan(parag_logits))
                assert not torch.any(torch.isinf(parag_logits))

                pad_idx = self.vocab.get_token_index('[PAD]', namespace='tokens')
                if self.class_weighting is not None:
                    loss = F.cross_entropy(
                        parag_logits,
                        label,
                        label_smoothing=self.label_smoothing,
                        weight=self.class_weights.to(parag_logits.device)
                        )
                else:           
                    loss = F.cross_entropy(
                        parag_logits,
                        label,
                        label_smoothing=self.label_smoothing
                        )
                    
                losses.append(loss)

                label_mask = label.ge(0) # True for tensor elements greater than or equal to 0
                self.sent_accuracy(parag_logits, label, label_mask)
                self.sent_f1_overall(parag_logits, label, label_mask)
                for label_name, f1 in self.sent_f1s.items():
                    f1(parag_logits, label, mask=label_mask)
            
            output['sent_loss'] = sum(losses)/len(losses) # using avg loss for each parag, because backprop uses the 'loss' key to access loss (scalar)
            output['loss'] = output['sent_loss']

            if (self.joint_loss_lambda is not None) and (self.do_seq_labelling):
                # seq_losses = max_parag_sent x batch x seq_len
                seq_losses = torch.tensor(seq_losses)
                seq_n_elements = torch.numel(seq_losses)
                summed_seq_losses = torch.sum(seq_losses)
                avg_seq_loss = summed_seq_losses / seq_n_elements
                wandb.log({"seq_loss": avg_seq_loss})
                output['loss'] = (1-self.joint_loss_lambda ) * output['loss'] \
                                    + self.joint_loss_lambda * avg_seq_loss
                output['seq_label_loss'] = seq_losses
        output['sent_probs'] = torch.stack(parag_sent_probs).transpose(0, 1) # b4 transpose: max_parag_len x batch x num_tags
        output['pred_sent_labels'] = torch.stack(parag_sent_preds).transpose(0, 1) # b4 transpose: max_parag_len x batch x num_tags
        if self.do_seq_labelling:
            output['seq_tag_probs'] = torch.stack(parag_sent_tag_probs).transpose(0, 1) # b4 transpose: max_parag_len x batch x seq_len x num_tags
            output['pred_seq_tags'] = torch.stack(parag_sent_tag_preds).transpose(0, 1) # b4 transpose: max_parag_len x batch x seq_len

        # if the sent is masked, its 0th token will be False
        #parag_mask = masks[:, :, 0]
        #parag_mask = torch.transpose(parag_mask, dim0=0, dim1=1) # FOR DOING PARAG ACC.
        #print(f"\nparag_preds: {parag_preds.shape}, sent_labels: {sent_labels.shape}, parag_masks: {parag_mask.shape}")
        # parag_preds = torch.stack(parag_sent_preds)
        # parag_preds = torch.unsqueeze(parag_preds, 1) # FOR CALCULATING PARAG ACC.
        
        if sent_labels is not None:
            if torch.isnan(output['loss']):
                losses = []
                #sent_labels = sent_labels.transpose(0, 1)
                logger.debug(f"label: {label}")
                logger.debug("="*20 + "="*20)
                logger.debug("Try adding 1e-8 to logit")
                for parag_logits, label in zip(parag_sent_probs, sent_labels):
                    new_logits = parag_logits+1e-8
                    logger.debug(f"new_logits: {new_logits}")
                    loss = F.cross_entropy(
                        new_logits,
                        label,
                        label_smoothing=self.label_smoothing,
                        weight=None if not self.class_weighting else self.class_weights.to(new_logits.device)
                        )
                    losses.append(loss)
                output['sent_loss'] = sum(losses)/len(losses) # using avg loss for each parag, because backprop uses the 'loss' key to access loss (scalar)
                output['loss'] = output['sent_loss']
                if self.do_seq_labelling:
                    output['loss'] = (1-self.joint_loss_lambda ) * output['sent_loss'] \
                                            + self.joint_loss_lambda * avg_seq_loss
                
                if torch.isnan(output['loss']):
                    losses = []
                    logger.debug("Try using rand for logit")
                    for parag_logits, label in zip(parag_sent_probs, sent_labels):
                        new_logits = torch.rand(parag_logits.size()).to(torch.cuda.current_device())
                        logger.debug(f"new_logits {new_logits}")
                        loss = F.cross_entropy(
                            new_logits,
                            label,
                            label_smoothing=self.label_smoothing,
                            weight=None if not self.class_weighting else self.class_weights.to(new_logits.device)
                            )
                        losses.append(loss)
                    output['sent_loss'] = sum(losses)/len(losses)
                    output['loss'] = output['sent_loss']
                    if self.do_seq_labelling:
                        output['loss'] = (1-self.joint_loss_lambda ) * output['sent_loss'] \
                                                + self.joint_loss_lambda * avg_seq_loss
                
            wandb.log({"loss": output['loss'],
                        "sent_loss": output['sent_loss'],})
            if self.do_seq_labelling:
                wandb.log({"joint_loss": output['loss']})

        if metadata is not None:
            output['metadata'] = metadata

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"sent accuracy": self.sent_accuracy.get_metric(reset),
                    "paragraph accuracy": self.parag_accuracy.get_metric(reset)['accuracy']}

        metrics['sent_f1_overall'] = self.sent_f1_overall.get_metric(reset)
        for label_name, f1 in self.sent_f1s.items():
            metrics["sent_"+label_name] = f1.get_metric(reset)

        if self.do_seq_labelling:
            f1_dict = self.strict_tag_f1.get_metric(reset=reset)
            metrics['span'] = normalize_span_f1_result(f1_dict)

            f1_dict = self.weak_tag_f1.get_metric(reset=reset)
            metrics['span_weak'] = normalize_span_f1_result(f1_dict)

        metrics = flatten_dict(metrics)
        metrics = {'/'.join(x): y for x, y in metrics.items()}
        wandb.log(metrics)

        if not self._verbose_metrics:
            metrics_to_return = {x: y for x, y in metrics_to_return.items() if "overall" in x or 'acc' in x}

        return metrics

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Postprocessing script for Evaluator (`allennlp evaluate`)
        Takes the result of forward and makes it human readable. 
        Most of the time, the only thing this method does is convert tokens / predicted labels 
        from tensors to strings that humans might actually understand.

        Since we've taken the softmax in `forward()` already, here we just return the output dict with labels.
        """
        human_readable_labels = []
        for parag in output_dict["pred_sent_labels"]:
            parag_labels = []
            for sent_label in parag:
                #readable_label = self.vocab.get_index_to_token_vocabulary(DEFAULT_LABEL_NAMESPACE).get(
                #     sent_label, str(sent_label)
                # )
                readable_label = self.vocab.get_token_from_index(int(sent_label), DEFAULT_LABEL_NAMESPACE)
                parag_labels.append(readable_label)
            human_readable_labels.append(parag_labels)
        output_dict["pred_sent_labels"] = human_readable_labels

        if self.do_seq_labelling:
            human_readable_tags = []
            for parag_tags in output_dict["pred_seq_tags"]:
                parag_sent_tags = []
                for sent_tags in parag_tags: # sent_tags, sent: both (batch x seq_len)
                    readable_tag_list, readable_sent = [], []
                    for tag in sent_tags:
                        readable_tag = self.vocab.get_token_from_index(int(tag), DEFAULT_TAG_NAMESPACE)
                        readable_tag_list.append(readable_tag)   
                    parag_sent_tags.append(readable_tag_list)
                human_readable_tags.append(parag_sent_tags)
            output_dict["pred_seq_tags"] = human_readable_tags

        return output_dict
