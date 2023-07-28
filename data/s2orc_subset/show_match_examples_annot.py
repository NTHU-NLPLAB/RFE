import jsonlines
import json
import argparse

from teufel_patterns.retag_teufel_patterns import check_overlap, resolve_overlap

OUT_LEN = 100
INCL_PAT = False
OVERLAP_CHECK_ITER = 3

def main():
    outdata = {"AIM": [], "BAS": [], "BKG": [], "CTR": [], "OTH": [], "OWN": [], "TXT": [], "NO_MATCH": []}
    orig_data = {"AIM": [], "BAS": [], "BKG": [], "CTR": [], "OTH": [], "OWN": [], "TXT": [], "NO_MATCH": []}
    line_cnt = 0
    with open("s2orc_matches.jsonl") as f:
        for line in f:
            line_cnt+=1
            data = json.loads(line)
            if data["patterns"]:
                for i in range(OVERLAP_CHECK_ITER):
                    overlap_span_idx = check_overlap(data, data['pattern_indices'])
                    if overlap_span_idx:
                        resolve_overlap(
                            data,
                            data['patterns'],
                            data['pattern_indices'],
                            overlap_span_idx,
                            merge=True,
                            keep_min=False
                            )
                    
                for pat_idx, pat_name in enumerate(data["patterns"]):
                    split_pat_name = pat_name.split("+")
                    found_labels = set()
                    for tok in split_pat_name:
                        if tok in outdata.keys():
                            found_labels.add(tok)
                    # get tags
                    marked_labels = ""
                    marked_pats = ""
                    for idx in range(0, len(split_pat_name), 2):
                        label, pat = split_pat_name[idx], split_pat_name[idx+1]
                        if label not in marked_labels:
                            marked_labels += "/"+label
                            marked_pats += "/"+pat

                    # tag sentence
                    match_idx = data["pattern_indices"][pat_idx]
                    orig_text = [w for w in data['text']]
                    data['text'][match_idx[0]] = "[[" + data['text'][match_idx[0]]
                    data['text'][match_idx[1]-1] = data['text'][match_idx[1]-1] + f"{marked_labels}]]"
                
                for lab in found_labels:
                    if len(outdata[lab]) < OUT_LEN:
                        outdata[lab].append(data['text'])
                        orig_data[lab].append(orig_text)
                    else: pass
            else:
                if len(outdata["NO_MATCH"]) < OUT_LEN:
                    outdata["NO_MATCH"].append(data['text'])
                    orig_data["NO_MATCH"].append(orig_text)
                else: pass

            min_lines = OUT_LEN
            all_done = [False for lab in outdata.keys()]
            if line_cnt%500 == 0:
                for idx, (lab, sent_list) in enumerate(outdata.items()):
                    print(line_cnt, "\t", lab, "\t", len(sent_list))
                    if len(sent_list) < min_lines:
                        all_done[idx] = False
                        continue
                    else:
                        all_done[idx] = True
            if all(all_done):
                break

    with open("s2orc_matches_examples_annot.txt", "w") as f:
        for label in outdata.keys():
            f.write(label+'\n\n')
            for sent in outdata[label]:
                f.write(" ".join(sent)+'\n\n')
            f.write("\n"+"="*100+"\n\n")
    print("example sents saved to s2orc_matches_examples_annot.txt")

    with open("s2orc_examples_annot.json", "w") as f:
        json.dump(orig_data, f)
    print("orig sents saved in s2orc_examples_annot.json") # for s2orc_ex_ngrams.py


if __name__=="__main__":
    main()
