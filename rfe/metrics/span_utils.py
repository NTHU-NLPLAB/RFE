# Modified from
# https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_utils/span_utils.py
from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token


TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)


T = TypeVar("T", str, Token)


def enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List[T]], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.

    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.

    # Parameters

    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans


def bio_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").

    # Parameters

    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns

    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or [DEFAULT_PADDING_TOKEN[2:]] # ME: replace [] with [PAD]?
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 1
    span_end = 1
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence[1:]): # ME: start from 1 because idx is the [CLS] token
        index = index+1 # ME: because we start enumerating from idx=1
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if (bio_tag not in ["B", "I", "O"]) and (string_tag != DEFAULT_PADDING_TOKEN): # ME: added 2nd condition
            # Note!! "string_tag <<is>> DEFAULT_PADDING_TOKEN" returns different results than
            # "string_tag == DEFAULT_PADDING_TOKEN" !!
            # "is" compares memory addresses of 2 vars. If Cython didn't optimise these variables by 'intern'-ing them, 
            # two vars might be stored in different addresses despite having same values.
            # "==" looks at values of 2 vars. So even if they aren't stored at the same address, 
            # you will still get True if they have same values.
            # This is why you should always use "... is (not) None", not "... == None"
            # https://realpython.com/python-is-identity-vs-equality/
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or (conll_tag in classes_to_ignore):
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and (conll_tag == active_conll_tag):
            # We're inside a span.
            span_end += 1
        elif (string_tag is DEFAULT_PADDING_TOKEN): 
            if (active_conll_tag is not None):
                # We're inside a span
                span_end += 1
            if (active_conll_tag is None): 
                # Previous is O (this active_conll_tag is None)
                continue
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                assert bio_tag == "I", f"Current tag is {string_tag}, but previously we're in I-{active_conll_tag}"
                spans.add((active_conll_tag, (span_start, span_end))) # end previous tag span
            active_conll_tag = conll_tag # use tag from current (ill-formed) span
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

