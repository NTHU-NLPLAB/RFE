# Modified from https://github.com/allenai/allennlp/blob/v2.9.3/allennlp/data/fields/sequence_field.py
from allennlp.data.fields import TextField, SequenceField
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from typing import Union, List, Optional, Dict

class SequenceField(SequenceField):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a method onto
    `Field`: :func:`sequence_length`.  It exists so that `SequenceLabelField`, `IndexField` and other
    similar `Fields` can have a single type to require, with a consistent API, whether they are
    pointing to words in a `TextField`, items in a `ListField`, or something else.
    """

    __slots__ = ['sequence', '_token_indexers']  # type: ignore

    def __init__(
        self,
        sequence: TextField,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None
    ) -> None:
        self.sequence = sequence
        self._token_indexers = token_indexers


    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        return len(self.sequence)

    def empty_field(self) -> "SequenceField":
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        if self._token_indexers is not None:
            for indexer_name, indexer in self.token_indexers.items():
                text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field