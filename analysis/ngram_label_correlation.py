import sys
import math
import pickle
import time
import spacy
import jsonlines
import argparse
from collections import Counter, defaultdict
import string
from tqdm import tqdm

#from ..helper_code.utils import LABEL_TO_INDEX
# TODO: refac using pandas dfs to store MI, rr, labels, word counts,
#  instead of handcrafted dictionaries...
# 1 pandas df for PMI, LMI, rr (rank), raw counts each

alphabets_exclude_a = set(string.ascii_letters)
alphabets_exclude_a.remove('a')
alphabets_exclude_a.remove('A')
alphabets_exclude_a.union(set(['\u2009', '±']))

nlp = spacy.load("en_core_web_sm")
print("Spacy loaded.")

mi_thresholds = {
    "pubmed": 100, # 100 for unigrams
    "az": 5
}

rr_thresholds = {
    "pubmed": 50, # 50 for unigrams
    "az": 3
}

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--xgram", type=int, default=1)
    parser.add_argument("--save_spacy_tokens", action="store_true")
    parser.add_argument("--load_from_cached", action="store_true")
    return parser.parse_args()

def recursive_flatten_to_dicts(in_list):
    """
    Recursively iterates through a list of lists of lists of..., whose
    bottom layer is a dict, and flatten all that into a list of dict.
    """
    out_list = []

    for it in in_list:
        if isinstance(it, list):
            out_list.extend(recursive_flatten_to_dicts(it))
        elif isinstance(it, dict): # the item is a dict => reached final layer
            out_list.append(it)
    return out_list

def print_tuple_list_rounded_values(in_tp_list, round_digit):
    for k, v in in_tp_list:
        rounded = round(v[0], round_digit)
        print(f"{k}, {rounded}, {v[1]}, ", end='')
    print()
    return

def text_is_clean(text):
    if not text:
        return False
    if text in string.punctuation:
        return False
    if any(char in string.punctuation for char in text):
        return False
    if any(char.isdigit() for char in text):
        return False
    if text.isspace():
        return False
    if len(text) == 1 and text in alphabets_exclude_a:
        return False
    if text == '/-':
        return False
    return True

class MutualInformation(object):
    def __init__(
        self,
        word,
        label,
        corpus_size,
        class_word_count,
        do_smoothing=True,
        smooth_add_k=100
        ):
        self.corpus_size = corpus_size

        assert word in class_word_count[label].keys()
        assert label in LABEL_TO_INDEX.values()

        self.count_wl = class_word_count[label][word]
        self.count_w = sum(sub_cnt_dict[word] for sub_cnt_dict in class_word_count.values())
        self.count_l = sum(class_word_count[label].values())
        assert self.count_wl <= self.count_w , \
            f"count_wl = {self.count_wl}, count_w = {self.count_w}"
        assert self.count_wl <= self.count_l, \
            f"count_wl = {self.count_wl}, count_l = {self.count_l}"
        
        self.do_smoothing = do_smoothing
        self.smooth_add_k = smooth_add_k
        self.pmi = 0
        self.lmi = 0

        if self.do_smoothing:
            assert smooth_add_k
            self.count_wl += smooth_add_k
            self.count_l += smooth_add_k
            self.count_w += smooth_add_k
        else:
            assert self.count_l * self.count_w * self.count_wl != 0

    def get_pmi(self):
        """point-wise mutual information
        """
        p_wl = self.count_wl / self.corpus_size
        p_w = self.count_w / self.corpus_size
        p_l = self.count_l / self.corpus_size
        return math.log10(p_wl / p_w / p_l)
    
    def get_lmi(self):
        """local mutual information
        
        Implements Schuster et al. (2019)'s LMI description
        (https://aclanthology.org/N18-2017.pdf)
        """
        p_l_given_w = self.count_wl / self.count_w
        p_wl = self.count_wl / self.corpus_size
        p_l = self.count_l / self.corpus_size
        return p_wl * math.log10(p_l_given_w / p_l)

def get_all_MI(class_word_count, label, corpus_size):
    """
    returns pmi, lmi for all words in a given class (label)
    """
    lmi_dict = {}
    pmi_dict = {}
    for word in class_word_count[label]:
        if class_word_count[label][word] < int(mi_thresholds[args.dataset] / args.xgram):
            continue

        mi_obj = MutualInformation(
            word,label, corpus_size, class_word_count
            )
        pmi_dict[word] = mi_obj.get_pmi()
        lmi_dict[word] = mi_obj.get_lmi()
    return {"pmi": pmi_dict, "lmi": lmi_dict}

def get_rank(counter_dict):
    rank = 1
    ranked_dict = {}

    sorted_counter_dict = counter_dict.most_common()
    for word, _ in sorted_counter_dict:
        ranked_dict[word] = rank
        rank += 1
    assert rank > 0

    return ranked_dict

def get_rank_ratio(class_word_count, label):
    """
    rank in entire corpus / rank in class
    => higher ratio means more prominent in class compared to corpus
    class_word_count: {label: {word: count}, label: {...}, ...}

    Output:
    [{word: ratio}, {word: ratio}, ...]
    """
    corpus_word_count = defaultdict(lambda: 0)
    for sub_cnt_dict in class_word_count.values():
        for w, cnt in sub_cnt_dict.items():
            corpus_word_count[w] += cnt
    corpus_word_count = Counter(dict(corpus_word_count))
    _class_word_count = class_word_count[label].copy()

    corpus_ranks = get_rank(corpus_word_count)
    class_ranks = get_rank(_class_word_count)

    class_corpus_rr = {}
    for word, class_rank in class_ranks.items():
        if corpus_word_count[word] < int(rr_thresholds[args.dataset] / args.xgram):
            continue
        class_corpus_rr[word] = corpus_ranks[word] / class_rank

    class_corpus_rr = Counter(class_corpus_rr).most_common() # sort by rank ratio
    # add class freq
    class_corpus_rr = [(w, (rr, _class_word_count[w])) for w, rr in class_corpus_rr]

    return class_corpus_rr # is sorted

def main(args):
    #args = build_args()
    labels = range(len(LABEL_TO_INDEX))
    if args.load_from_cached:
        t_start = time.time()
        with open(f"outputs/{args.dataset}.pickle", "rb") as f:
            corpus_by_class = pickle.load(f)
        print(f"Loaded tokenized corpus from outputs/{args.dataset}.pickle, took {time.time()-t_start}")
    else:
        corpus = list(jsonlines.open(args.data_file))
        print(f"Loaded text data from {args.data_file}")

        print("Flattening data to list of dicts...")
        corpus = recursive_flatten_to_dicts(corpus) # a list of dicts
        # The above is mainly for the 2 types of input data: sents and parags
        
        corpus_by_class = {k: [] for k in labels}
        print("Tokenising...")
        for line in tqdm(corpus):
            # append tokenised sentence
            corpus_by_class[LABEL_TO_INDEX[line['label']]].append(nlp(line['text']))

        # TODO: save as cache?
        if args.save_spacy_tokens:
            pickled_doc = pickle.dumps(corpus_by_class)
            with open(f"outputs/{args.dataset}.pickle", "wb") as f:
                f.write(pickled_doc)
            print(f"Saved tokenized corpus to outputs/{args.dataset}.pickle.")

    print("Cleaning data...")
    class_word_count = {k: None for k in labels}
    for label in tqdm(corpus_by_class):
        all_toks = []
        if args.xgram == 1:
            all_toks = [tok.text for line in corpus_by_class[label] for tok in line if text_is_clean(tok.text)]
        else:
            assert args.xgram > 1
            all_toks = []
            for line in corpus_by_class[label]:
                for idx, tok in enumerate(line[:-args.xgram]):
                    # if text_is_clean(tok.text): # cleaning or not mostly only affect unigrams
                    all_toks.append(' '.join([t.text for t in line[idx:idx+args.xgram]]))
        class_word_count[label] = Counter(all_toks)
    
    corpus_size = sum(sum(sub_cnt_dict.values()) for sub_cnt_dict in class_word_count.values())
    
    # Get mutual information stats
    print("\nCalculating mutual information...\n")
    mis_by_label = {}
    for label in labels:
        mis = get_all_MI(class_word_count, label, corpus_size)
        mis = {mi_type: Counter(mi).most_common() for mi_type, mi in mis.items()}
        for mi_type, mi_tps in mis.items():
            for idx, tp in enumerate(mi_tps):
                w = tp[0]
                mi = tp[1]
                # add class freq
                mi_tps[idx] = (w, (mi, class_word_count[label][w]))
        mis_by_label[label] = mis
    
    # Print MI summary
    for label in mis_by_label:
        print(INDEX_TO_LABEL[label])
        for mi_type, mis in mis_by_label[label].items():
            print(f"Top 10 {mi_type.upper()}")
            print_tuple_list_rounded_values(mis[:10], 3)
        print()

    dataset = args.dataset
    with jsonlines.open(f"outputs/{dataset}_mi_info_{args.xgram}gram.jsonl", "w") as f:
        for label, mis in mis_by_label.items():
            f.write({label: mis})
    print(f"Mutual information saved to outputs/{dataset}_mi_info_{args.xgram}gram.jsonl\n")

    # Get rank ratios
    print("Calculating rank ratio...\n")
    rank_ratios_all_labels = {}
    for label in labels:
        rank_ratios_all_labels[label] = get_rank_ratio(class_word_count, label)

    # Print rank ratio summary
    for label, rr in rank_ratios_all_labels.items():
        print(INDEX_TO_LABEL[label])
        print("Top 10 rank ratio:")
        print_tuple_list_rounded_values(rr[:10], 3)
        print()

    with jsonlines.open(f"outputs/{dataset}_rank_ratio_{args.xgram}gram.jsonl", "w") as f:
        for label, rr in rank_ratios_all_labels.items():
            f.write({label: rr})
    print(f"Rank ratio saved to outputs/{dataset}_rank_ratio_{args.xgram}gram.jsonl")



if __name__=="__main__":
    args = build_args()

    if args.dataset == "pubmed":
        LABEL_TO_INDEX = {
            'BACKGROUND': 0,
            'OBJECTIVE': 1,
            'METHODS': 2,
            "RESULTS": 3,
            "CONCLUSIONS": 4,
        }
    elif args.dataset == "az":
        LABEL_TO_INDEX = {
            'AIM': 0,
            'BAS': 1,
            'BKG': 2,
            "CTR": 3,
            "OTH": 4,
            "OWN": 5,
            "TXT": 6
        }

    INDEX_TO_LABEL = {v:k for k, v in LABEL_TO_INDEX.items()}
    main(args)