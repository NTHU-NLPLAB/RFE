import os
import sys
import re
import pickle
import jsonlines
import json
#import spacy

from itertools import product
from tqdm import tqdm
from collections import defaultdict

from patterns import FORMULAIC_PATTERNS, AGENT_PATTERNS
from lexicons import ALL_CONCEPT_LEXICONS, ALL_ACTION_LEXICONS

all_patterns = [FORMULAIC_PATTERNS, AGENT_PATTERNS]
all_lexicons = ALL_CONCEPT_LEXICONS
all_lexicons.update(ALL_ACTION_LEXICONS)


all_lex_cnt = 0
for val in all_lexicons.values():
    all_lex_cnt+=len(val)
print(f"avg lex counts: {all_lex_cnt/len(all_lexicons)}")


def split_patterns(pattern_type):
    """pattern_type = FORMULAIC_PATTERNS, AGENT_PATTERNS, ALL_CONCEPT_LEXICONS, ALL_ACTION_LEXICONS
    Split the patterns into lists of things to match.
    """
    for pattern_name in list(pattern_type.keys()):
        tokenized = []
        patterns = pattern_type[pattern_name]
        for pattern in patterns:
            tokenized.append(pattern.split())
        pattern_type[pattern_name] = tokenized
    # return pattern_type
    

def tokenise_dataset(dataset, save_as_cache=False):
    """dataset: List[Dict{"text": str, "label": str, "fileno": int}]
    """
    nlp = spacy.load("en_core_web_lg")
    print('Spacy loaded! Tokenising dataset...')
    tokenised = []
    for parag in tqdm(dataset):
        tok_parag = []
        for sent in parag:
            tok_parag.append(
                {
                    "text": nlp(sent["text"].strip()),
                    "label": sent["label"],
                    "fileno": sent["fileno"]
                }
            )
        tokenised.append(tok_parag)
    print('Done tokenising!')
    if save_as_cache:
        with open("./data/az/az_papers/parag_train_tokenised_pattern.pickle", "wb") as f:
            pickled_doc = pickle.dumps(tokenised)
            f.write(pickled_doc)
        print("Tokenised AZ papers training set saved in "
        +"'./data/az/az_papers/parag_train_tokenised_pattern.pickle'")
    return tokenised



def main(in_file, out_file):
    # all_patterns_with_lexicon = defaultdict(lambda: list())
    all_patterns_with_lexicon = defaultdict(lambda: defaultdict(lambda: list()))

    total_pat_len = 0
    min_, max_ = 10000, -1
    for pattern_type in all_patterns:
        split_patterns(pattern_type)
        for pat_name, pat_list in tqdm(pattern_type.items()):
            total_pat_len += len(pat_list)
            for pattern in pat_list:
                if len(pattern) > max_:
                        max_ = len(pattern)
                        max_pattern = pattern
                if len(pattern) < min_:
                        min_ = len(pattern)
                        min_pattern = pattern
                lex_positions = {}
                for tok_idx, token in enumerate(pattern):
                    if token.startswith("@"): # is lexicon
                        lex_positions[tok_idx] = all_lexicons.get(token[1:], False)
                        if not lex_positions[tok_idx]:
                            print(f"{token[1:]} not found in lexicon!")
                            print(f"erroneous token: {token}, pattern: {pattern}")
                            return
                all_lex_combinations = product(*(lex_positions.values()))
                

                for lex_comb in all_lex_combinations:
                    filled_pattern = pattern.copy()
                    for comb_idx, lex_pos in enumerate(sorted(list(lex_positions.keys()))):
                        # comb_idx = order of lex in comb (first lex in pattern => 1st item in combination tuple)
                        # lex_pos = where each lex is in the pattern (tok idx)
                        filled_pattern[lex_pos] = lex_comb[comb_idx]
                    
                    # split pattern into list of unigrams
                    split_pattern = []
                    for tok in filled_pattern:
                        if len(tok.split()) > 1:
                            mwe = tok.split()
                            split_pattern.extend(mwe)
                        else:
                            split_pattern.append(tok)
                    all_patterns_with_lexicon[pat_name][split_pattern[0]].append(split_pattern)
    
    val_cnt = 0
    for pat_head_dict in all_patterns_with_lexicon.values(): # keys are pattern types/names
        val_cnt+=sum(len(pat_list) for pat_list in pat_head_dict.values())
    print(f"min pattern len: {min_} ({min_pattern}); max pattern len: {max_} ({max_pattern})")
    print(f"average num items in all patterns: {val_cnt/total_pat_len}")
    print(f"all_patterns_with_lexicon: {len(all_patterns_with_lexicon)} pattern types")
    print(f"total num pats: {total_pat_len}")
    
    with open(out_file, "w") as f:
    #with open("all_patterns_with_lexicon.json", "w") as outfile:
        json.dump(all_patterns_with_lexicon, f)
    print(f"saved to {out_file}")
    
    
if __name__=="__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    main(in_file, out_file)

