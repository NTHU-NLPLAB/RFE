import jsonlines
import json, sys

from teufel_patterns.retag_teufel_patterns import check_overlap, resolve_overlap

OUT_LEN = 100
INCL_PAT = False
OVERLAP_CHECK_ITER = 3

def main(in_file, out_file):
    outdata = {"AIM": [], "BAS": [], "BKG": [], "CTR": [], "OTH": [], "OWN": [], "TXT": [], "NO_MATCH": []}
    line_cnt = 0
    # in_file = "s2orc_matches.jsonl"
    with open(in_file) as f:
        for line in f:
            line_cnt+=1
            line_data = json.loads(line)
            if isinstance(line_data, dict):
                line_data = [line_data]
            
            for data in line_data:
                #breakpoint()
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
                                marked_labels += "\\"+label
                                marked_pats += "\\"+pat

                        # tag sentence
                        match_idx = data["pattern_indices"][pat_idx]
                        for tok_idx, tok in enumerate(data['text'][match_idx[0]:match_idx[1]]):
                            actual_idx = tok_idx + match_idx[0]
                            data['text'][actual_idx] = data['text'][actual_idx]+marked_labels
                    
                    for lab in found_labels:
                        if len(outdata[lab]) < OUT_LEN:
                            outdata[lab].append((data['label'], data['text'], data['patterns']))
                        else: pass
                else:
                    if len(outdata["NO_MATCH"]) < OUT_LEN:
                        outdata["NO_MATCH"].append((data['label'], data['text'], []))
                    else: pass

                min_lines = OUT_LEN
                all_done = [False for lab in outdata.keys()]
                if line_cnt%500 == 0:
                    for idx, (lab, sent_tp_list) in enumerate(outdata.items()):
                        print(line_cnt, "\t", lab, "\t", len(sent_tp_list))
                        if len(sent_tp_list) < min_lines:
                            all_done[idx] = False
                            continue
                        else:
                            all_done[idx] = True
                if all(all_done):
                    break

    #out_file = "s2orc_matches_examples.txt"
    with open(out_file, "w") as f:
        for label in outdata.keys():
            f.write(label+'\n\n')
            for sent_match_tp in outdata[label]:
                f.write("{ "+sent_match_tp[0]+" } "+\
                        " ".join(sent_match_tp[1])+'\n'\
                        +str(sent_match_tp[2])+'\n\n')
            f.write("\n"+"="*100+"\n\n")
    print(f"examples saved in {out_file}")



if __name__=="__main__":
    in_file, out_file = sys.argv[1], sys.argv[2]
    main(in_file, out_file)
