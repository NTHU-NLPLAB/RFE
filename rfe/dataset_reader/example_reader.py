from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, LabelField, ListField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer, Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from typing import Iterable, Dict, Union
from typing import Iterable, Dict, Union
import jsonlines
import logging
import sys
from ..utils import LABEL_TO_INDEX, REGION_BREAK
from .sequence_field import SequenceField

logger = logging.getLogger(__name__)
DEFAULT_TAG_NAMESPACE = 'move_tags'
DEFAULT_LABEL_NAMESPACE = 'class_labels'

class TagAlignError(Exception):
    pass

@DatasetReader.register('example_reader')
class ExampleReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 text_token_indexers: Dict[str, TokenIndexer],
                 max_parag_length: int,
                 dataset: str,
                 do_seq_labelling: bool = None,
                 input_mode: str = "sent",
                 attach_header: bool = None,
                 filter_length: bool = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_length = filter_length
        self.tokenizer = tokenizer
        self.text_token_indexers = text_token_indexers
        self.max_parag_length = max_parag_length
        self.dataset = dataset
        self.do_seq_labelling = do_seq_labelling
        self.input_mode = input_mode
        self.attach_header = attach_header

    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(file_path) as f:
            num_examples = 0
            for example in f:
                if self.input_mode == 'parag': # NOTE: 'parag mode' is NOT being maintained at the moment!!
                # example = List[Dict{'text': , "label": ..., "seq_tag": ...}]
                    num_examples += 1
                    if len(example) > self.max_parag_length:
                        continue
                    text = [sent['text'] for sent in example]
                    if isinstance(text[0], list) and isinstance(text[0][0], str):
                        # sent['text'] is tokenised as a list of words
                        text = [' '.join(sent_text) for sent_text in text]
                    sent_label = [sent['label'] for sent in example]
                    assert text, f"{num_examples}th example"
                    assert sent_label, f"{num_examples}th example"
                    if self.do_seq_labelling:
                        seq_labels = [sent['seq_tag'] for sent in example] # a list of lists!
                        
                elif self.input_mode == "sent":
                    # example = Dict{'text': ..., 'label', ..., 'seq_tag': ...}
                    if isinstance(example['text'], list):
                        if self.filter_length is True and len(example['text']) < 4:
                            continue
                    num_examples += 1
                    if self.attach_header:
                        # attach at the end so it won't mess up seq-labelling tags
                        text = [example['text'] + ["topic:"+example['header']]]
                    else:
                        text = [example['text']]

                    if isinstance(text[0], list) and isinstance(text[0][0], str):
                        # sent['text'] is tokenised in the pre-processing into a list of words => stitch it back
                        text = [' '.join(sent_text) for sent_text in text]
                    sent_label = [example['label']]
                    assert text and sent_label, f"{num_examples}th example"
                    if self.do_seq_labelling:
                        if self.attach_header:
                            header_tags = ['O'] * (len(example['header'].split())+2) # +2 for 'topic', ':'
                            seq_labels = [example['seq_tag'] + header_tags]
                        else:
                            seq_labels = [example['seq_tag']]

                try:
                    if self.do_seq_labelling:
                        instance = self.example_to_instance(text, sent_label, seq_labels)
                    else:
                        instance = self.example_to_instance(text, sent_label)
                except TagAlignError as e:
                    logger.warning(f"Skipping example {num_examples}")
                    continue
                
                yield instance

    def align_tags_to_tokens(self, tokenized_line: Iterable[Token], tags: Iterable[int]) -> Iterable[int]:
        """Gets offset for each subword wrt each word they originate from, and align subwords
        to word-based sequence tags.
        Subwords that are not the beginning of a word will be assigned -100 as their tag. 
        Special tokens ([CLS], [SEP]) will also get -100.

        INPUT
        tokenized_line: List[Token]
        https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/pretrained_transformer_tokenizer.py#L283
            Each token should contain subword start and end index with respect to each word
            e.g. [CLS] This is @huggingface [SEP] => [[CLS], This, is, @, hugging, face, [SEP]]
            [CLS], [SEP]: (0, 0)
            This: (0, 4)
            is: (0, 2)
            @huggingface => [@, hugging, face]; @ = (0, 1); hugging = (1, 8); face = (8, 12)
        
        OUTPUT
        aligned_tags: List[int]
            1 tag for each subword. Should be the same length as tokenized_line.
        """
        aligned_tags = []
        orig_tag_indices = [] # how we will split the words back into subwords to align with the tags
        # if None => special toks ([CLS], [SEP]); starts with 0; subwords of the same word share the same index
        # e.g. This is @huggingface.  => This(0) is(1) @(2) ##hugging(2) ##face(2)
        
        cur_tag_idx = 0
        prev_subword_end = None
        for idx, tok in enumerate(tokenized_line):
            start, end = tok.idx, tok.idx_end
            try:
                if start == end == None: # is one word in orig sent but has no tag
                    aligned_tags.append(DEFAULT_PADDING_TOKEN)
                    orig_tag_indices.append(-1) # use -1 to distinguish from subword (-100)
                elif start != prev_subword_end: # is one word in orig sent
                    aligned_tags.append(tags[cur_tag_idx])
                    orig_tag_indices.append(cur_tag_idx)
                    cur_tag_idx += 1
                elif start == prev_subword_end: # a subword
                    aligned_tags.append(DEFAULT_PADDING_TOKEN)
                    orig_tag_indices.append(-100)
            except IndexError:
                logger.warning(f"IndexError in aligning tags to subwords. Skipping this example.")
                logger.info(f"The offending example:\n"
                            f"tokenized_line: len = {len(tokenized_line)}, {tokenized_line}\n"
                            f"tags: len = {len(tags)}, {tags}\n"
                            f"aligned tags: len = {len(aligned_tags)}, {aligned_tags}")
                # for future investigation:
                # tokenized_line = [[CLS], Here, the, re, ##comb, ##ination, process, is, neglected, because, the, system, we, consider, is, do, ##ped, [UNK], ., In, steady, state, [UNK], ↑, (, [UNK], ), /, [UNK], [SEP]]
                # aligned_tag = ['@@PADDING@@', 'O', 'O', 'O', '@@PADDING@@', '@@PADDING@@', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '@@PADDING@@', 'O', '@@PADDING@@', 'O', 'O', 'O', 'O', 'O', '@@PADDING@@', '@@PADDING@@', 'O', 'O']
                return None, None
                # looks like it could be a punctuation problem, which might mean we need to use another way to join spacy's pre-tokenised text
                # OR figure out how to add "is_split_into_words" in allennlp's transformers...
            prev_subword_end = end

        assert len(aligned_tags) == len(tokenized_line) == len(orig_tag_indices), \
            f"Expected aligned_tags to be len {len(tokenized_line)} but got {aligned_tags}"

            # Commented out below is for transformers < 4.1 probably; 
            # Newer transformer versions use different stat & end indexing.
            # if (start==0) & (end!=0):
            #     aligned_tags[idx] = tags[cur_tag_idx] # (0,0) => special tokens; (non-zero, non-zero) => not first subword
            #     cur_tag_idx += 1
            # if (start==0) & (end!=0):
            #     aligned_tags[idx] = tags[cur_tag_idx] # (0,0) => special tokens; (non-zero, non-zero) => not first subword
            #     cur_tag_idx += 1
        
        return aligned_tags, orig_tag_indices
    
    def example_to_instance(
        self,
        texts: Iterable[str],
        sent_labels: int = None,
        seq_labels: int = None
        ) -> Instance:
        """
        Turns tokenised sents into TextField datatype & index them.
        https://guide.allennlp.org/reading-data#1
        In the dataset_reader, the fields names are important, because the resulting dictionary of tensors 
        is passed by name to the model, they have to match the model's forward() arguments *exactly*.
        """
        parag_field = []
        tokenised = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text) # AllenNLP uses 'return_offsets_mapping=True' by default
            tokenised.append(tokens) # for align_tags_to_tokens if `PretrainedTransformerTokenizer` is used
            text_field = TextField(tokens, self.text_token_indexers) # where we put the tokenised and the labels
            # text_token_indexers=> {'tokens': SingleIdTokenIndexer}
            parag_field.append(text_field)
        parag_field = ListField(parag_field) # num_parag x sent_len
        instance = {'paragraph': parag_field} # paragraph, sent_labels, seq_labels
        instance['metadata'] = MetadataField({"text": texts})

        _LABEL_TO_INDEX = LABEL_TO_INDEX[self.dataset]
        all_sent_labels_field = []
        if sent_labels is not None:
            for label in sent_labels:
                label_field = LabelField(label, label_namespace=DEFAULT_LABEL_NAMESPACE)
                all_sent_labels_field.append(label_field)
            all_sent_labels_field = ListField(all_sent_labels_field)
            instance["sent_labels"] = all_sent_labels_field # [max_parag_len]

        all_seq_label_fields = []
        all_orig_tag_indices_fields = []
        if seq_labels is not None:
            for sent_idx, tag_sequence in enumerate(seq_labels):
                # pretrained transformer tokenisers split words into subwords => requires aligning tags to tokens
                aligned_tags = []
                orig_tag_indices = None
                if isinstance(self.tokenizer, PretrainedTransformerTokenizer):
                    tokenised_line = tokenised[sent_idx].copy()
                    aligned_tags, orig_tag_indices = self.align_tags_to_tokens(tokenised_line, tag_sequence)
                    # TODO: how to make sure tags are aligned with indices in text_token_indexer as well?
                    if (orig_tag_indices is None) and (aligned_tags is None): # faulty example, should be discarded
                        raise TagAlignError
                else:
                    aligned_tags = tag_sequence

                tokenised_line = SequenceField(tokenised_line)
                sent_seq_labelfield = SequenceLabelField(aligned_tags, tokenised_line, label_namespace=DEFAULT_TAG_NAMESPACE)
                orig_tag_indices_field = SequenceLabelField(orig_tag_indices, tokenised_line, label_namespace="orig_tag_seq")
                all_seq_label_fields.append(sent_seq_labelfield) # List[SequenceLabelField]
                all_orig_tag_indices_fields.append(orig_tag_indices_field)
            instance["seq_tags"]  = ListField(all_seq_label_fields) # num_parag x sent_len
            instance["orig_tag_indices"]  = ListField(all_orig_tag_indices_fields) # num_parag x sent_len
            
        instance = Instance(instance)
        # AllenNLP turns Instance objects into batches of tensors through a DataLoader.
        return instance


def main(): # run main() for debugging
    file_path = sys.argv[1]
    #max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    example_reader = ExampleReader(
        WhitespaceTokenizer(),
        {'tokens': SingleIdTokenIndexer()},
        max_parag_length=13,
        dataset="az"
    )
    cnt = 0
    print(f"{REGION_BREAK}")
    for instance in example_reader.read(file_path): # ExampleReader._read is called
        print(instance)
        cnt += 1
        if cnt == 10:
            break
    print(f"{REGION_BREAK}")

if __name__ == '__main__':
    main()
