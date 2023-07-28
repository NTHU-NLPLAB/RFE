import jsonlines
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from teufel_patterns.retag_teufel_patterns import check_overlap, resolve_overlap
from pprint import PrettyPrinter

from teufel_patterns.pattern2label import LABEL_PATTERNS

LABEL_PATTERNS_ = {pat: str(lab)+"_"+str(pat) for lab, pat_set in LABEL_PATTERNS.items() for pat in pat_set}
#print(LABEL_PATTERNS_)

OVERLAP_CHECK_ITER=3
#LABELS=["AIM", "BAS", "BKG", "CTR", "OTH", "OWN", "TXT"]
all_tags_from_label = defaultdict(lambda: list())
all_tagnames_from_label = defaultdict(lambda: defaultdict(lambda: int()))
pp=PrettyPrinter()

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--check_overlap", action="store_true")
    parser.add_argument("--labelling_type", default="normal", choices=["normal", "match_class", "match_class_strict"])
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()

def span_to_bio(data, match_idx, matched_labels):
    """Convert the given span of text into a BIO (Beginning, Inside, Outside) tagged sequence.
    
    Args:
        data (dict): A dictionary of data containing "text" (input sentence), among other keys (not used).
        match_idx (tuple): A tuple containing the start and end indices of the span of the tag.
        matched_labels (list): A list of labels to use for the tagging, one for each matched pattern.
        matched_pats (list): A list of patterns that were matched in the text.
        
    Returns:
        out_data (List[Dict]): Each dict in the list contains the original data and a new "seq_tag" field
        which contains the BIO tags for the given span.
    
    """
    out_data = []
    for lab in matched_labels:
        tags = ["O"] * len(data["text"])
        data_cp = {k: v for k, v in data.items()}
        tags[match_idx[0]] = f"B-{lab}"
        for idx in range((match_idx[0]+1), match_idx[1]):
            tags[idx] = f"I-{lab}"
        data_cp["seq_tag"] = tags
        data_cp["tagged_label"] = lab
        out_data.append(data_cp)
    # make BIO for the patterns here if needed: for pat in matched_pats...
    return out_data


def main():
    args = build_args()
    total_len = 139264
    line_cnt = 0
    with jsonlines.open(args.in_file) as in_f,\
        jsonlines.open(args.out_file, "w") as out_f,\
        tqdm(total=total_len) as pbar:

        parag = in_f.read()
        class_cnt = defaultdict(lambda: int())
        for parag in in_f:
            if isinstance(parag, dict): # for s2orc_matches.jsonl
                parag = [parag]

            for line in parag:
                line_cnt += 1
                assert isinstance(line, dict), f"Input should be a dict but found {type(line)}"
                if line["patterns"]:
                    if args.check_overlap:
                        for __ in range(OVERLAP_CHECK_ITER):
                            overlap_span_idx = check_overlap(line, line['pattern_indices'])
                            if overlap_span_idx:
                                resolve_overlap(
                                    line,
                                    line['patterns'],
                                    line['pattern_indices'],
                                    overlap_span_idx,
                                    merge=True,
                                    keep_min=False
                                    )
                    assert len(line["patterns"])==len(line["pattern_indices"])

                    for match_idx, pat_name in zip(line["pattern_indices"], line["patterns"]):
                        split_pat_name = pat_name.split("+")
                        # split pattern name into "label" and "pattern" (originally: "label+pattern")
                        matched_labels = []
                        matched_pats = []
                        class_label = line['label']
                        for idx in range(0, len(split_pat_name), 2):
                            tag_label, pat = split_pat_name[idx], split_pat_name[idx+1]
                            if args.labelling_type == "match_class_strict" and tag_label != class_label:
                                continue
                            matched_labels.append(tag_label)
                            matched_pats.append(pat)
                            class_cnt[tag_label]+=1
                            all_tags_from_label[line['label']].append(tag_label)
                            all_tagnames_from_label[line['label']][LABEL_PATTERNS_[pat]] += 1

                        if args.labelling_type == "match_class_strict" and matched_labels:
                            assert all([lab == class_label for lab in matched_labels])
                        if args.labelling_type == "match_class":
                            matched_labels = [class_label for lab in matched_labels]
                        bio_tagged_data = span_to_bio(line, match_idx, matched_labels)
                        
                        for d in bio_tagged_data:
                            out_f.write(d)
                else:
                    line["seq_tag"] = ["O"] * len(line["text"])
                    line["tagged_label"] = "NONE"
                    class_cnt["NO_MATCH"] += 1
                    out_f.write(line)
            pbar.update(1)
            
        print(f"sequence tag distribution: {dict(class_cnt)}")
        print(f"line count: {line_cnt}")

        for class_name in all_tags_from_label:
            all_tags_from_label[class_name] = Counter(all_tags_from_label[class_name])
        print(f"which classes each tag belongs to: ")
        # for cls_label, tag_counter in all_tags_from_label.items():
        #     print(cls_label, dict(tag_counter))
        # print()

        for cls_label, tagname_counter in all_tagnames_from_label.items():
            print(cls_label, dict(all_tags_from_label[cls_label]))
            pp.pprint(tagname_counter)
            print()

        print("#"*50+"\n\n")
        

if __name__=="__main__":
    main()