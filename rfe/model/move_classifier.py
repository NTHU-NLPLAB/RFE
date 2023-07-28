from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import ListField
from allennlp.nn import util
from allennlp.common import FromParams, Params
from allennlp.models import Model
from allennlp.data.fields import TextField
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, SequenceAccuracy, F1Measure
from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.modules.transformer import positional_encoding
from allennlp.nn import InitializerApplicator
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..utils import LABEL_TO_INDEX

@Model.register('move_classifier')
class MoveClassifier(Model):
    """
    ## Parameters
    vocab: Vocabulary,
    embedder: `TextFieldEmbedder` for turning input text into TextField embeddings
    pooler: the [CLS] token of (BERT) transformers to pool encoder sentence outputs 
        from max_seq_len x hidden into size of 1 x hidden
    checkpoint: str, path to trained weights
    dataset: 'az', 'pubmed', 's2orc-move'
    init_clf_weights: str, {None, 'zeros', 'rand'}, ways to init the Linear classifier layer
        zeros: init with 0s
        rand: init with rands (TODO: specify rand range (in normal dist?))
        copy: copy from loaded weights, discard extras if input num_labels is larger than current;
                sfill in 0s if fewer
        (See `load_weights()` for details)
    """
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 pooler: Seq2VecEncoder,
                 checkpoint: str,
                 dataset: str,
                 init_clf_weights: str,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super().__init__(vocab) # not really sure why this works.
        # more Vocab config info: https://guide.allennlp.org/using-config-files#4
        if not checkpoint:
            initializer(self)
        self.embedder = embedder
        self.pooler = pooler
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.init_clf_weights = init_clf_weights # how the Linear for the classifier will be initialised

        LABEL_TO_INDEX_local = LABEL_TO_INDEX[self.dataset]
        self.pe = positional_encoding.SinusoidalPositionalEncoding()
        self.num_labels = len(LABEL_TO_INDEX_local)
        self.classifier = nn.Linear(pooler.get_output_dim(), self.num_labels)
        self.sent_accuracy = CategoricalAccuracy()
        self.parag_accuracy = SequenceAccuracy()
        self.sent_f1s = [F1Measure(positive_label=idx) for idx in range(len(LABEL_TO_INDEX_local))]

        if self.checkpoint:
            self.load_weights(self.checkpoint)

    def forward(self,
                paragraph: ListField, # takes Instance output from example_reader
                labels: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        sent_heads = []

        # paragraph: {'tokens': {'token_ids': [...], 'mask': [...], 'type_ids': [...]},
        # values of token_ids, mask, type_ids have shape (max num sents, max sent length) x batch

        # print(len(paragraph)) # 1 (corresponds to using 1 token indexer, using namespace 'tokens')
        # print(labels.shape) # batchsize x (max num of sents in a parag)
        
        assert len(paragraph.keys()) == 1 and 'tokens' in paragraph.keys()
        text_field = paragraph['tokens']
        masks = text_field['mask'].transpose(1, 0)
        token_ids = text_field['token_ids'].transpose(1, 0)
        type_ids = text_field['type_ids'].transpose(1, 0)
        #print(f"masks: {masks.shape}, {type(masks)}")

        for sent_idx, sent in enumerate(token_ids):
            token_id = sent # batch x (max seq len)
            mask = masks[sent_idx]
            type_id = type_ids[sent_idx]
            sent_inst = {
                    "tokens": {
                        "mask": mask,
                        "token_ids": token_id,
                        "type_ids": type_id
                        }
                    }
            token_embeds = self.embedder(sent_inst)
            mask = util.get_text_field_mask(sent_inst)
            encoding = self.pooler(token_embeds, mask=mask) # batch x hidden
            sent_heads.append(encoding)
        # Add positional encoding to sents # TODO: use a better pos encoding
        sent_heads = torch.stack(sent_heads) # (num max sents) x batch x hidden
        # transposing because pos enc takes batch x timescale (i.e. num max sents) x hidden
        sent_heads = torch.transpose(sent_heads, dim0=0, dim1=1)
        sent_heads = self.pe(sent_heads)

        output = {} # batch x (num max sents) x hidden
        parag_logits = [] # List[sent logits]
        parag_probs = [] # List[probs of each sent in each class]
        parag_preds = []
        losses = []
        # put each parag into classifier, get logit & class pred for each sent
        for sent_idx, sent in enumerate(sent_heads): # iterate through each sent in the parag
            # parag: (max_parag_len) x hidden [represents a paragraph]
            logits = self.classifier(sent)
            probs = F.softmax(logits, dim=1) # (num max sents) x (num labels)
            parag_probs.append(probs)
            parag_logits.append(logits) # logits = 1 parag
            
            pred_label = torch.argmax(probs, dim=1)
            parag_preds.append(pred_label)

            if labels is not None: # labels = batch[List[labels]]; labels for each sent in this parag
                parag_labels = labels[sent_idx]
                loss = F.cross_entropy(logits, parag_labels, ignore_index=-1)
                losses.append(loss)
                label_mask = parag_labels.ge(0) # True for tensor elements greater than or equal to 0
                self.sent_accuracy(logits, parag_labels, label_mask)
                for f1 in self.sent_f1s: # don't use list comprehension!
                    # the returned val of the function is None since it only updates attr info!
                    f1(logits, parag_labels, mask=label_mask)
            output['loss'] = sum(losses)/len(losses) # using avg loss for each parag for now,
            # because this ('loss') will be used by backprop etc.
        output['probs']= parag_probs
        parag_preds = torch.stack(parag_preds)
        parag_preds = torch.unsqueeze(parag_preds, 1)
        parag_mask = masks[:, :, 0]
        parag_mask = torch.transpose(parag_mask, dim0=0, dim1=1)
        #print(f"\nparag_preds: {parag_preds.shape}, labels: {labels.shape}, parag_masks: {parag_mask.shape}")
        # TODO: Do parag acc update here if needed.

        return output # AllenNLP's forward returns a dict, not a tensor like PyTorch

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"sent accuracy": self.sent_accuracy.get_metric(reset), \
            "paragraph accuracy": self.parag_accuracy.get_metric(reset)['accuracy'], }
        for f1 in self.sent_f1s:
            metrics.update(**f1.get_metric(reset))

        return metrics

    def load_weights(self, model_path):
        ckpt_dict = torch.load(
            model_path,
            map_location=lambda storage, loc: storage,
        ) # load checkpoint in corresponding device (cuda, or cpu if no cuda)

        target_weights_shape = list(ckpt_dict["classifier.weight"].shape)
        target_weights_shape[0] = self.num_labels
        target_bias_shape = list(ckpt_dict["classifier.bias"].shape)
        target_bias_shape[0] = self.num_labels

        if self.init_clf_weights == "random":
            ckpt_dict["classifier.weight"] = torch.rand(target_weights_shape)
            ckpt_dict["classifier.bias"] = torch.rand(target_bias_shape)
            print(f"\nLoaded weights with type {self.init_clf_weights}!\n")

        if self.init_clf_weights == "zero":
            ckpt_dict["classifier.weight"] = torch.zeros(target_weights_shape)
            ckpt_dict["classifier.bias"] = torch.zeros(target_bias_shape)
            print(f"\nLoaded weights with type {self.init_clf_weights}!\n")

        target_weights_shape = list(ckpt_dict["classifier.weight"].shape)
        target_weights_shape[0] = self.num_labels
        target_bias_shape = list(ckpt_dict["classifier.bias"].shape)
        target_bias_shape[0] = self.num_labels

        if self.init_clf_weights == "random":
            ckpt_dict["classifier.weight"] = torch.rand(target_weights_shape)
            ckpt_dict["classifier.bias"] = torch.rand(target_bias_shape)
            print(f"\nLoaded weights with type {self.init_clf_weights}!\n")

        if self.init_clf_weights == "zero":
            ckpt_dict["classifier.weight"] = torch.zeros(target_weights_shape)
            ckpt_dict["classifier.bias"] = torch.zeros(target_bias_shape)
            print(f"\nLoaded weights with type {self.init_clf_weights}!\n")

        if self.init_clf_weights == "copy":
            ckpt_num_labels = ckpt_dict["classifier.weight"].shape[0]
            if self.num_labels < ckpt_num_labels:
                # remove extra weights to fit self.num_labels
                ckpt_dict["classifier.weight"] = ckpt_dict["classifier.weight"][:self.num_labels, :]
                ckpt_dict["classifier.bias"] = ckpt_dict["classifier.bias"][:self.num_labels]
                # TODO: also try throwing away weights of other labels?
            elif self.num_labels > ckpt_num_labels:
                # add dimension len to fit new num of labels
                extra_weights = torch.randn(ckpt_dict["classifier.weight"][0].shape)
                extra_weights = torch.stack([extra_weights for i in range(self.num_labels-ckpt_num_labels)], 0)
                extra_biases =  torch.cat([torch.randn(1) for i in range(self.num_labels-ckpt_num_labels)], 0)
                
                ckpt_dict["classifier.weight"] = torch.cat((ckpt_dict["classifier.weight"], extra_weights), 0)
                ckpt_dict["classifier.bias"] = torch.cat((ckpt_dict["classifier.bias"], extra_biases))
            print(f"\nLoaded weights with type q{self.init_clf_weights}q!\n")

        model_dict = self.state_dict()
        ckpt_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict.keys()} # model => ckpt_dict
        assert len(ckpt_dict), "Cannot find shareable weights"
        model_dict.update(ckpt_dict)
        self.load_state_dict(model_dict)

        print(f"weights after manipulation: {self.state_dict()['classifier.weight']}\n")

        print(f"weights after manipulation: {self.state_dict()['classifier.weight']}\n")
        print(f"\n\n====== Checkpoint loaded from {self.checkpoint} ======\n\n")

    def adjust_weight_shape(self, key_name, loaded_state_dict):
        raise NotImplementedError
        raise NotImplementedError
        cur_model = self.state_dict()
        cur_shape = dict(cur_model)[key_name].shape
        loaded_shape = loaded_state_dict.shape
        assert cur_shape == loaded_shape

        for dim_idx, dim in enumerate(cur_shape):
            if dim > loaded_shape[dim_idx]:
                a = 1
            elif dim < loaded_shape[dim_idx]:
                a = 1
        return
