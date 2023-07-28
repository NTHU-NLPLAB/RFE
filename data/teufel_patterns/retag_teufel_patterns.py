import re
import os
import sys
import jsonlines
import logging
import json
import pickle
import spacy
import argparse

from datetime import datetime
from time import time
from tqdm import tqdm

DEBUG=False
logger = None

def split_patterns(pattern_dict):
    """pattern_dict = Dict[List[str]]
    Split the patterns into lists of things to match.
    """
    for pattern_name, pattern_list in pattern_dict.item():
        tokenized = []
        for pattern in pattern_list:
            tokenized.append(pattern.split())
        pattern_dict[pattern_name] = tokenized
    # return pattern_type

def parse_dataset(dataset, data_path=None, cache_path=None, filter_length=None):
    """dataset: List[Dict{"text": str, "label": str, "fileno": int}]
    """
    nlp = spacy.load("en_core_web_lg", exclude = ['ner', 'custom', 'textcat'])
    logger.info('Spacy loaded! Parsing dataset...')
    parsed = []
    sent_too_short, diff_labels, diff_headers = 0, 0, 0
    for parag in tqdm(dataset):
        tok_parag = []
        sent_buffer = {}
        for sent in parag:
            # the average char len of a sentence in this dataset is 114; std=60; avg # words per sent = 25
            # avg word len = 4.5 char (std=3.1)
            if filter_length:
                if len(sent['text']) < 20 and not sent_buffer:  # 15 ~= 4 words
                    sent_too_short += 1
                    sent_buffer = {**sent}
                    continue
                if sent_buffer: # previous sent is too short, merge with current sent
                    sent['text'] = sent_buffer['text'] + sent['text']
                    # for label, header, fileno, s_id: use longer sentence's; but keep track
                    if sent['label'] != sent_buffer['label']:
                        diff_labels += 1
                    if sent['header'] != sent_buffer['header']:
                        diff_headers += 1
                    sent_buffer = {}
            
            sent["text"] = nlp(sent["text"].strip())
            tok_parag.append(sent)
        parsed.append(tok_parag)
    logger.info('Done parsing!')
    logger.info(f'{sent_too_short} sentences too short (merged); of which {diff_labels} different labels, {diff_headers} different headers.')

    save_path = cache_path if cache_path is not None else data_path.split(".j")[0]+".pickle"
    with open(save_path, "wb") as f:
        pickled_doc = pickle.dumps(parsed)
        f.write(pickled_doc)
    logger.info(f"Parsed data saved in '{save_path}'")

    return parsed


def load_cached_parsed_data(cache_path, data_path, filter_length=None):
    # ad-hoc code for reading in ALG intro
    #if data_path == "../2004-ALG.txt" and not (os.path.exists(cache_path)):
    if data_path == "../alg_test.txt" and not (os.path.exists(cache_path)):
        #with open("../2004-ALG.txt") as f:
        with open("../alg_test.txt") as f:
            alg = [line.strip() for line in f.readlines() if line!="\n"]
            alg = " ".join(alg)
            # while alg[:4] != "We":
            #     alg = f.readline()
        nlp = spacy.load("en_core_web_lg")
        nlp.add_pipe("sentencizer")
        alg = nlp(alg)
        all_parags = []
        for sent in alg.sents:
            all_parags.append(
                {"text": sent.as_doc()}
            )
        logger.info(f"alg len: {len(alg)}")
        # with open(cache_path, "wb") as f:
        #     pickled_doc = pickle.dumps([all_parags])
        #     f.write(pickled_doc)
        # logger.info(f"written to '{cache_path}'")
        return all_parags
    if cache_path is not None:
        assert os.path.exists(cache_path), f"cache_path {os.path.exists(cache_path)} does not exist!"
        logger.info(f"Loading parsed dataset from '{cache_path}'...")
        with open(cache_path, "rb") as f:
            all_parags = pickle.load(f)
        logger.info(f"'{cache_path}' loaded")
    else:
        assert data_path.endswith("jsonl")
        all_parags = list(jsonlines.open(data_path))
        all_parags = parse_dataset(all_parags, data_path=data_path, cache_path=cache_path, filter_length=filter_length)
    return all_parags


def check_overlap(sent, matched_indices_list):
    '''matched_indices_list = matched_patterns['indices'] = [(start, end), (...)]
    If multiple matches exist for a sentence, check if any of the patterns overlap in 
    sent position.
    Returns list of tuples, each tuple = index of the match in matched_indices_list
    that overlaps with the other match in the tuple
    Returns empty list if no overlap
    '''
    if len(matched_indices_list) < 2:
        return []
    overlap_span_idx = []
    matched_pos = [-1 for __ in sent['text']] # fill in with idx of "which match" as we go through matched_indices_list
    has_ovlp = False
    for which_match, match_span in enumerate(matched_indices_list): 
        # match_idx: idx of the match tuple among all matches
        # match_span = (match_start, match_end)
        for has_match in matched_pos[match_span[0]:match_span[1]]: # token has match or not
            # the current match's range of start:end includes other matches!
            if has_match >= 0:
                overlap_span_idx.append((has_match, which_match))
                has_ovlp = True
                break
        # update matched_pos in sentence (`matched_pos[idx]` is the orig value in matched_pos)
        matched_pos = [matched_pos[idx] if ((idx < match_span[0]) or (idx >=match_span[1])) else which_match 
                        for idx in range((len(sent['text'])))]
        if DEBUG and has_ovlp:
            logger.debug(f"===== Has overlap in sent {sent['text']}, sent pattern idx = {sent['pattern_indices']},")
            logger.debug(f"===== matched_pos = {matched_pos}, overlap_span_idx = {overlap_span_idx}")
    return overlap_span_idx


def resolve_overlap(sent, matched_patterns, pattern_idx_list, overlap_span_idx, merge, keep_min):
    """if overlap exists, decide resolve type:
    Type 1
    merge: when there is overlap but inclusion [0, 3], [2, 5] => [0, 5]
    TODO keep_copies: copy the sentence, each copy contains different matches the had overlaps

    Type 2
    keep_min: when one is included in another [1, 5], [2, 3] => keep [2, 3]
    if keep_min=False: when one is included in another [1, 5], [2, 3] => keep [1, 5]
    """
    new_seq_tags = {"pattern_indices": [],
                    "patterns": []}
    all_overlapped_idx = set()
    for tp in overlap_span_idx:
        # tp: the positions in the "matched patterns" list that correspond to patterns that overlap with each other
        all_overlapped_idx = all_overlapped_idx.union(set(tp))
    all_overlapped_idx = list(all_overlapped_idx)

    for match_idx, match in enumerate(pattern_idx_list):
        if match_idx not in all_overlapped_idx:
            new_seq_tags["pattern_indices"].append(match)
            new_seq_tags["patterns"].append(matched_patterns[match_idx])
            continue
        for match_1_idx, match_2_idx in overlap_span_idx:
            match_1_span = pattern_idx_list[match_1_idx]
            match_2_span = pattern_idx_list[match_2_idx]
            final_span = []
            final_pattern = matched_patterns[match_1_idx] if matched_patterns[match_1_idx] == matched_patterns[match_2_idx] else ""
            if match_1_span == match_2_span:
                final_span = list(match_1_span)
                if not final_pattern:
                    final_pattern = matched_patterns[match_1_idx]+"+"+matched_patterns[match_2_idx]
            elif (((match_1_span[0] <= match_2_span[0]) and (match_1_span[1] > match_2_span[1]))
                or
                ((match_1_span[0] < match_2_span[0]) and (match_1_span[1] >= match_2_span[1]))):
                # match_1 wraps around match_2
                if keep_min:
                    final_span = list(match_2_span)
                    final_pattern = matched_patterns[match_2_idx]
                else:
                    final_span = list(match_1_span)
                    final_pattern = matched_patterns[match_1_idx]
            elif (((match_2_span[0] <= match_1_span[0]) and (match_2_span[1] > match_1_span[1]))
                or
                ((match_2_span[0] < match_1_span[0]) and (match_2_span[1] >= match_1_span[1]))):
                # match_2 wraps around match_1
                if keep_min:
                    final_span = list(match_1_span)
                    final_pattern = matched_patterns[match_1_idx]
                else:
                    final_span = list(match_2_span)
                    final_pattern = matched_patterns[match_2_idx]
            elif ((match_1_span[0] > match_2_span[0]) and (match_1_span[1] > match_2_span[1])):
                # first of 2 is at the left of first of 1
                if merge:
                    final_span = [match_2_span[0], match_1_span[1]]
                    # make new combined pattern ()
                    if not final_pattern:
                        final_pattern = matched_patterns[match_1_idx]+"+"+matched_patterns[match_2_idx]
            elif (match_1_span[0] < match_2_span[0] and (match_1_span[1] < match_2_span[1])):
                # first of 1 is at the left of first of 2, but last of 2 is outside right of 1
                if merge:
                    final_span = [match_1_span[0], match_2_span[1]]
                    # make new combined pattern
                    if not final_pattern:
                        final_pattern = matched_patterns[match_1_idx]+"+"+matched_patterns[match_2_idx]
            assert final_span, f"{(match_1_span, match_2_span)}"
            assert final_pattern
            # check for duplicates before adding                  
            if final_span not in new_seq_tags["pattern_indices"]:
                new_seq_tags["patterns"].append(final_pattern)
                new_seq_tags["pattern_indices"].append(final_span)
    if DEBUG: logger.debug(f"===== before resolve: sent = {sent} =====")
    sent.update(new_seq_tags)
    if DEBUG: logger.debug(f"==== after resolve: sent matches = {sent} =====")

    
def match_pattern(sentence, pat_head_dict, pattern_name, class_name, verbose=False):
    """pat_head = first token of pattern
    pat_list: List[List[Str]], list of patterns; value of pat_head
    (patterns with the same first token are stored under the same pat_head)
    pattern_name: pattern type (name of a patterns under pat_head); key of {pat_head: pat_list}
    sentence: spacy doc object (we will need its `.tag_`, `.text` attributes)
    """
    def senttok_eq_pattok(sent_tok, pat_tok):
        if pat_tok in {"CITATION", "SELFCITATION"}:
            if sent_tok.text in {"CITATION", "SELFCITATION"}:
                return True
            elif sent_tok.tag_ in ["NNP", "NNPS"] and sent_tok.text != "EQN": # proper noun
                return True
            else:
                return False
        elif pat_tok == sent_tok.text == 'CREF':
            return True
        elif pat_tok.startswith("#"): #match for POS
            if pat_tok in ["#NNP", "#NNPS "]: # most likely OTH tags
                if sent_tok.tag_ in ["CREF", "CITATION", "SELFCITATION"]:
                    return True
            elif str(sent_tok.tag_)[0] != pat_tok[1]:
                return False
            else: return True
        elif str(sent_tok.lemma_).lower() == pat_tok:
            return True
        return False
            
    # match token by token to enable pos matching
    # TODO: chck how equations are represented in BOTH az corpus and in patterns! 
    # => looks like eqns aren't represented in patterns at all?

    matched_patterns = {
        'pattern_indices': [],
        'patterns': []
    }
    
    for sent_offset, __ in enumerate(sentence[:len(sentence)]): 
         
         for pat_head in pat_head_dict:
            #pat_head = pat_head_dict.keys()
            if not senttok_eq_pattok(sentence[sent_offset], pat_head):
                #logger.debug(sentence[sent_offset], pat_head)
                continue # no need to check the patterns beginning with this token
            
            # assuming we found a pat_head that matches the 1st token of the sent chunk
            patters_list = pat_head_dict[pat_head]
            for pattern in patters_list:
                pat_len = len(pattern)
                if sent_offset+pat_len >= len(sentence):
                    break
                
                v_after_agent = 0
                if pattern_name.endswith("AGENT") and sentence[sent_offset+pat_len].tag_:
                    # Require that AGENT must be followed by a verb / modal verb.
                    # This rule-based approach is VERY limited!! :((
                    if sent_offset != 0:
                        if sentence[sent_offset-1].tag_ in {"IN", "TO", "WRB", "WP", "WDT"}: # IN: prepositions; CC: and, or; WXX: where, which, that
                            break # extremely simplistic but this avoids misidentifying sents like:
                            # "The prototype developed "for" this\AIM study\AIM comprised\AIM two modes of operation: ..."
                    if (sentence[sent_offset+pat_len].tag_[0] == "V") or (sentence[sent_offset+pat_len].tag_[0] == "MD"):
                        v_after_agent = 1
                    # check stricter conditions
                    if sent_offset+pat_len+1 < len(sentence) and sentence[sent_offset+pat_len+1].tag_: # check 2 words after
                        if (sentence[sent_offset+pat_len].tag_[0], sentence[sent_offset+pat_len+1].tag_[0])==("R", "V"):
                            v_after_agent = 2 # adv + v
                    if sent_offset+pat_len+2 < len(sentence) and sentence[sent_offset+pat_len+2].tag_:
                        # require an adv (word list?) in between be and VBN so the result isn't too noisy
                        if (sentence[sent_offset+pat_len].lemma_=="be") and\
                           (sentence[sent_offset+pat_len+1].tag_=="RB") and \
                           (sentence[sent_offset+pat_len+2].tag_=="VBN"):
                            v_after_agent = 3 # passive (be + past participle); here the matched NP is actually not an agent but jason wants this anyway...
                        elif (sentence[sent_offset+pat_len].lemma_=="be") and\
                           (sentence[sent_offset+pat_len+1].tag_=="VBN") and \
                           (sentence[sent_offset+pat_len+2].tag_=="IN"):
                            v_after_agent = 3
                    if not v_after_agent:
                        break
                        
                # see if this chunk of sentence matches the whole pattern
                match = [False for tok in pattern]
                for pat_idx, pat_tok in enumerate(pattern):
                    sent_idx = sent_offset+pat_idx
                    if not senttok_eq_pattok(sentence[sent_idx], pat_tok):
                        match[pat_idx] = False
                        break # move on to matching next pattern
                    else:
                        match[pat_idx] = True
                    
                    if all(match):
                        matched_patterns['pattern_indices'].append((sent_offset, sent_offset+pat_len+v_after_agent))
                        matched_patterns['patterns'].append(class_name+"+"+pattern_name)
                        if DEBUG:
                            logger.debug(f"Match! {pattern}"
                                         f"{matched_patterns['patterns']}"
                                         f"{matched_patterns['pattern_indices']}"
                                         f"{[tok.text for tok in sentence[sent_offset:sent_offset+pat_len+v_after_agent]]}"
                                         )
    return matched_patterns
        
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-i", type=str, required=True)
    parser.add_argument("--pattern_path", "-p", type=str, required=True)
    parser.add_argument("--cached_parse", "-c", type=str, default=None)
    parser.add_argument("--cached_matches", "-m", type=str, default=None)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--filter_length", "-f", action="store_true")
    parser.add_argument("--out_file", "-o", type=str, required=True)
    return parser.parse_args()


def main():
    args = build_args()
    print(f"Input args: {args}")

    global DEBUG, logger
    DEBUG = args.debug
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%d%m%Y-%H%M%S") # ddmmYY-HMS
    logging.basicConfig(
                    handlers=[
                        logging.FileHandler(f"./retag_teufel_patterns_{dt_string}.log", mode='a'),
                        logging.StreamHandler(sys.stdout)
                    ],
                    #format='[%(asctime)s.%(msecs)d %(levelname)s] :%(funcName)s: %(message)s' if DEBUG else ' %(message)s',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG if DEBUG else logging.INFO)

    logger = logging.getLogger(__name__)

    cached_matches = ""
    if DEBUG:
        logger.debug("testing with ALG's abstract")
        # all_parags = load_cached_parsed_data('../alg_parags.pickle', '../2004-ALG.txt')
        all_parags = load_cached_parsed_data("../alg_intro.pickle", "../alg_test.txt")
    else:
        data_path = args.data_path # see file2cache.txt for file-cache correspondence
        if args.cached_matches and os.path.exists(args.cached_matches):
            cached_matches = args.cached_matches
            all_parags = list(jsonlines.open(cached_matches))
        else:
            all_parags = load_cached_parsed_data(args.cached_parse, data_path, filter_length=True)

    all_patterns = None
    try:
        #assert os.path.exists("../teufel_patterns/all_patterns_with_lexicon_by_label.json")
        assert os.path.exists(args.pattern_path)
        with open(args.pattern_path) as f:
            all_patterns = json.load(f)
    except AssertionError:
        with open("../../teufel_patterns/all_patterns_with_lexicon_by_label.json") as f:
            all_patterns = json.load(f)

    if not os.path.exists(cached_matches):
        # start finding matches, takes ~17 min
        start_time = time()
        logger.info('Matching Teufel patterns...')
        if isinstance(all_parags[0], dict):
            all_parags = [[sent] for sent in all_parags]

        for parag in tqdm(all_parags):
            for sent in parag:
                matched_pats = {"pattern_indices": [], "patterns": []}
                sent_text = sent['text']
                if DEBUG:
                    logger.debug(f"\nChecking sentence: {sent_text}")
                for label in all_patterns.keys():
                    for pat_name, pat_head_dict in all_patterns[label].items():
                        # AGENT_PATTERNS are considered fomulaic patterns if theyre not subjects of sents (section 5.2.2.2). Not implemented yet. 
                        # see also: https://github.com/davidjurgens/citation-function/blob/master/code/global_functions_march16.py
                        matched = match_pattern(sent_text, pat_head_dict, pat_name, label)
                        # matched = {"patterns": [...], "pattern_indices": [...]}
                        if matched['patterns']:
                            matched_pats['patterns'].extend(matched['patterns']) # old format: uses `append()``
                            matched_pats['pattern_indices'].extend(matched['pattern_indices'])
                sent.update(matched_pats)
                sent['text'] = [tok.text for tok in sent["text"]] # convert spacy doc to list of toks
        logger.info(f"\nFinished matching! Took {time()-start_time}\n")
    # # unpack inner lists
    # for parag in tqdm(all_parags):
    #     for sent in parag:
    #         matched_pats = sent['patterns']
    #         matched_pats = [pat for pat_list in matched_pats for pat in pat_list]
    #         sent['patterns'] = matched_pats

    #         matched_spans = sent['pattern_indices']
    #         matched_spans = [span for span_list in matched_spans for span in span_list]
    #         sent['pattern_indices'] = matched_spans

    
    logger.info("Checking and resolving pattern overlaps...")
    for parag in tqdm(all_parags):
        for sent in parag:
            if "pattern_indices" not in sent.keys(): # az_parags_matched_patterns_no_resolve_overlap.jsonl
                # uses a version of code where the key "pattern_indices" was "indices"
                sent["pattern_indices"] = sent['indices']
            overlap_span_idx = check_overlap(sent, sent['pattern_indices'])
            if overlap_span_idx:
                resolve_overlap(
                    sent,
                    sent['patterns'],
                    sent['pattern_indices'],
                    overlap_span_idx,
                    merge=True,
                    keep_min=False
                    )
                #print(sent)
            if DEBUG: logger.debug(sent)
    # if DEBUG: exit()

    logger.info("writing data tagged with pattern to file...")
    with jsonlines.open(args.out_file, "w") as f:
        f.write_all(all_parags)
    logger.info(f"written to '{args.out_file}'!\n\n")


if __name__=="__main__":
    start = time()
    main()
    logger.debug(f"Took {time()-start} in total")
