# source : https://gist.github.com/ngcrawford/2237170
# also ref: https://superfastpython.com/multiprocessing-pool-map-multiple-arguments/
import json, jsonlines
import sys, logging

from multiprocessing import Pool, cpu_count
#from more_itertools import chunked
#from textwrap import dedent
from itertools import zip_longest, chain, repeat
from time import time
from tqdm import tqdm

# from ..teufel_patterns.retag_teufel_patterns import match_pattern


DEBUG = False
aux_data = None
logging.basicConfig(
                handlers=[
                    logging.FileHandler("./multith_match.log", mode='a'),
                    logging.StreamHandler(sys.stdout)
                ],
                #format='[%(asctime)s.%(msecs)d %(levelname)s] :%(funcName)s: %(message)s' if DEBUG else ' %(message)s',
                format='%(message)s',
                datefmt='%H:%M:%S',
                level=logging.DEBUG)

# for more info on the pos tag defs, see
# https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf
POS_MAP = { # maps tuefel's POS tags to spacy's
    "JJ": ["JJ", "JJR", "JJS"],
    "NN": ["NN", "NNS", "NNP", "NNPS", "PRP"], 
    "RB": ["RB", "RBS", "RBR"],
    "VV": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"], # MD = modal
}
NOISY_SENTS = set(
    ["The authors have nothing to disclose .",
     "CONFLICTS OF INTEREST"
    ]
)

logger = logging.getLogger(__name__)

class SimpleParToken(object):
    def __init__(self, deppar_line) -> None:
        """INPUT: deppar_line: token with dependency info from Joanna's parsing code
        """
        tags = deppar_line.strip().split("\t")
        if len(tags)==6:
            self.text, self.lemma_, self.pos_, self.tag_, self.head, self.children = tags
        elif len(tags)==5:
            self.text, self.lemma_, self.pos_, self.tag_, self.head = tags
            self.children=None
        else:
            self.text = tags[0]
            self.lemma_ = self.pos_ = self.tag_ = self.head = self.children = None


def read_sents(filehandle):
    sent = []
    #with open(filename) as f:
    line = filehandle.readline()
    while line:
        while line != "\n":
            if line.strip():
                tok = SimpleParToken(line)
                sent.append(tok)
            line = filehandle.readline()
        # out of the loop: we've encountered a "\n"]
        #print([tok.text for tok in sent])
        if len(sent) > 5:
            yield sent
        sent = []
        line = filehandle.readline()
    
# pos tag ref:
# https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf
def match_pattern(sentence, pat_head_dict, pattern_name, class_name, verbose=False):
    """pat_head = first token of pattern
    pat_list: List[List[Str]], list of patterns; value of pat_head
    (patterns with the same first token are stored under the same pat_head)
    pattern_name: pattern type (name of a patterns under pat_head); key of {pat_head: pat_list}
    sentence: List[SimpleParToken] (we will need its `.tag_`, `.text` attributes)
    """
    def senttok_eq_pattok(sent_tok, pat_tok):
        if pat_tok in ["CREF", "CITATION", "SELFCITATION"]:
            if sent_tok.text == "CREF":
                pass
            elif sent_tok.tag_ in ["NNP", "NNPS"]: # proper noun
                pass
            else:
                return False
            
        elif pat_tok.startswith("#"): #match for POS
            if str(sent_tok.tag_)[0] != pat_tok[1]:
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
                is_agent = 0
                if pattern_name.endswith("AGENT") and sentence[sent_offset+pat_len].tag_:
                    # Require that AGENT must be followed by a verb / modal verb.
                    # This rule-based approach is VERY limited!! :((
                    if sent_offset != 0:
                        if sentence[sent_offset-1].tag_ == "IN" or sentence[sent_offset-1].tag_ == "TO":
                            break # extremely simplistic but this avoids mis-identifying sents like:
                            # "The prototype developed for this\AIM study\AIM comprised\AIM two modes of operation: ..."
                    if (sentence[sent_offset+pat_len].tag_[0] == "V") or (sentence[sent_offset+pat_len].tag_[0] == "MD"):
                        is_agent = 1
                    # check stricter conditions
                    if sent_offset+pat_len+1 < len(sentence) and sentence[sent_offset+pat_len+1].tag_: # check 2 words after
                        if (sentence[sent_offset+pat_len].tag_[0], sentence[sent_offset+pat_len+1].tag_[0])==("R", "V"):
                            is_agent = 2 # adv + v
                        if (sentence[sent_offset+pat_len].lemma_=="be") and (sentence[sent_offset+pat_len+1].tag_=="VBN"):
                            is_agent = 2 # passive (be + past participle); here the matched NP is actually not an agent but jason wants this anyway...
                    if not is_agent:
                        break
                        
                # see if this chunk of sentence matches the whole pattern
                match = [False for tok in pattern]
                for pat_idx, pat_tok in enumerate(pattern):
                    sent_idx = sent_offset+pat_idx # honestly not sure if i should -1 here...
                    if not senttok_eq_pattok(sentence[sent_idx], pat_tok):
                        break # move on to matching next pattern
                    else:
                        match[pat_idx] = True
                    
                    if all(match):
                        matched_patterns['pattern_indices'].append((sent_offset, sent_offset+pat_len+is_agent))
                        matched_patterns['patterns'].append(class_name+"+"+pattern_name)
                        if DEBUG:
                            logger.debug(f"Match! {pattern}"
                                         f"{matched_patterns['patterns']}"
                                         f"{matched_patterns['pattern_indices']}"
                                         f"{[tok.text for tok in sentence[sent_offset:sent_offset+pat_len]]}"
                                         )
    return matched_patterns

def match_all_patterns(sentence, all_patterns):
    sent_text = " ".join([tok.text for tok in sentence])
    if sent_text in NOISY_SENTS:
        return {"text": sent_text, "pattern_indices": [], "patterns": []}
    
    sent_data = {"text": sentence}
    matched_pats = {"pattern_indices": [], "patterns": []}
    for label in all_patterns.keys():
        for pat_name, pat_head_dict in all_patterns[label].items():
            # AGENT_PATTERNS are considered fomulaic patterns if theyre not subjects of sents (section 5.2.2.2). Not implemented yet. 
            # see also: https://github.com/davidjurgens/citation-function/blob/master/code/global_functions_march16.py
            matched = match_pattern(sentence, pat_head_dict, pat_name, label)
            # matched = {"patterns": [...], "pattern_indices": [...]}
            if matched['patterns']:
                matched_pats['patterns'].extend(matched['patterns']) # old format: uses `append()``
                matched_pats['pattern_indices'].extend(matched['pattern_indices'])
    
    sent_data.update(matched_pats)
    sent_data["text"] = [tok.text for tok in sentence]
    return sent_data

def init_match_all_pattern(init_data):
    """loads shared data (large)
    """
    global aux_data
    aux_data = init_data
    
def match_all_patterns_wrapper(data_iterator):
    return match_all_patterns(data_iterator, aux_data)


if __name__ == '__main__':

    in_file = "./s2orc.0.dependency.mcite.txt"
    outfile = "./s2orc_matches.jsonl"

    with open("../teufel_patterns/all_patterns_with_lexicon_by_label.json") as f:
        all_patterns = json.load(f)

    start_time = time()
    total_lines = 588394107
    with open("./s2orc.0.dependency.mcite.txt") as in_f,\
        jsonlines.open(outfile, "w") as out_f:

        parsed_sents_generator = read_sents(in_f)
        # ref: 
        # https://rvprasad.medium.com/data-and-chunk-sizes-matter-when-using-multiprocessing-pool-map-in-python-5023c96875ef
        num_proc = min(cpu_count()/2, 16)
        with Pool(num_proc, init_match_all_pattern, (all_patterns,)) as p:
            results = tqdm(p.imap(match_all_patterns_wrapper,
                       parsed_sents_generator,
                       chunksize=256,), total=total_lines
                    )
            # error:
            # Traceback (most recent call last):
            #   File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/lib/python3.8/multiprocessing/pool.py", line 848, in next
            #     item = self._items.popleft()
            # IndexError: pop from an empty deque

            # forum ref: https://bugs.python.org/issue28696
            # Looks like an `imap` issue, will try using just `map`

            # update: can't use map because we need to write out as we get the results!
            # no error like above when running on server though...
            for r in results:
                if r:
                    out_f.write(r)
            logger.info(f"\nFinished matching! Took {time()-start_time}\n")
