import sys
import json
from collections import defaultdict
from pattern2label import LABEL_PATTERNS


# TODO: multi-process code for matching patterns
# change multith_match to match all labels instead of just AIM!
 
def main(in_file, out_file):
    #with open("all_patterns_with_lexicon.json") as f:
    with open(in_file) as f:
        all_patterns = json.load(f)
    
    # all_patterns = {pattern_name: {pattern_head: [patterns]}}
    patterns_by_label = defaultdict(lambda: dict())

    for pat_name, pat_head_dict in all_patterns.items():
        for label, label_pat_name in LABEL_PATTERNS.items():
            if pat_name in label_pat_name:
                patterns_by_label[label][pat_name] = pat_head_dict
    
    assert set(patterns_by_label.keys()) == set(['BAS', 'OTH', 'TXT', 'OWN', 'AIM', 'CTR', 'BKG'])

    #out_file = "all_patterns_with_lexicon_by_label.json"
    with open(out_file, 'w') as f:
        json.dump(patterns_by_label, f)
    print(f"Saved to {out_file}!")
    

if __name__=="__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    main(in_file, out_file)