import jsonlines
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay
from pprint import PrettyPrinter

from teufel_patterns.pattern2label import LABEL_PATTERNS

#LABEL_PATTERNS_ = {pat: str(lab)+"_"+str(pat) for lab, pat_set in LABEL_PATTERNS.items() for pat in pat_set}
#print(LABEL_PATTERNS_)


sns.set(rc={"figure.dpi":100, 'savefig.dpi':300})

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True) # the already bio-tagged data
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--type", choices=["bio-tagged", "matched-patterns"])
    parser.add_argument("--normalise", action="store_true")
    return parser.parse_args()

pprinter = PrettyPrinter()
class_tag_corr = defaultdict(lambda: defaultdict(lambda: int()))
class_tagname_corr = defaultdict(lambda: defaultdict(lambda: int()))
class_count = defaultdict(lambda: int())
tagname_counts = defaultdict(lambda: int())

args = build_args()
for line in jsonlines.open(args.in_file):
    tag_types = []
    tagnames = []
    if args.type == "bio-tagged":
        if isinstance(line, dict):
            line = [line]
        for actual_line in line:
            for tag in actual_line['seq_tag']:
                if len(tag.split("-")) > 1:
                    tags.append(tag.split("-")[1])
                else: 
                    tags.append(tag)
            tags = set(tags)
            for tag in tags:
                if tag != 'O':
                    class_tag_corr[actual_line['label']][tag] += 1

    elif args.type == "matched-patterns":
        if isinstance(line, dict):
            line = [line]
        for actual_line in line:
            if actual_line['patterns']:
                for pat in actual_line['patterns']:
                    #print(pat)
                    split_pat = pat.split("+")
                    for i in range(len(split_pat[::2])):
                        #print(split_pat, i)
                        actual_idx = i*2
                        tag_type, tag_name = split_pat[actual_idx], split_pat[actual_idx+1] # OTH - US_PREVIOUS
                        
                        class_tagname_corr[actual_line['label']][tag_name] += 1
                        class_tag_corr[actual_line['label']][tag_type] += 1

                        tag_types.append(tag_type) # US_PREVIOUS
                        tagnames.append(tag_name)

                        tagname_counts[tag_name]+=1
                        class_count[actual_line['label']] += 1
        

# -------------------------------plot class_tag_corr--------------------------------------- #
sorted_labels = sorted(list(class_tag_corr.keys()))
IDX_TO_LABEL = {i: l for i, l in enumerate(sorted_labels)}
num_labels = len(IDX_TO_LABEL.keys())


if args.type == "bio-tagged":
    class_tag_cm = np.zeros((num_labels, num_labels))
    tag_class_cm = np.zeros((num_labels, num_labels))

    for class_idx in range(num_labels):
        for tag_idx in range(num_labels):
            class_tag_cm[class_idx][tag_idx] = class_tag_corr[IDX_TO_LABEL[class_idx]][IDX_TO_LABEL[tag_idx]]

    print(f"class_tag_cm: {class_tag_cm}", class_tag_cm.shape)

    class_tag_corr_norm_tag = defaultdict(lambda: defaultdict(lambda: int()))
    class_tag_corr_norm_cls = defaultdict(lambda: defaultdict(lambda: int()))
    if args.normalise:
        for class_idx in range(num_labels):
            for tag_idx in range(num_labels):
                class_tag_corr_norm_tag[IDX_TO_LABEL[class_idx]][IDX_TO_LABEL[tag_idx]] = class_tag_cm[class_idx][tag_idx] / sum(class_tag_cm[class_idx, :])
                class_tag_corr_norm_cls[IDX_TO_LABEL[class_idx]][IDX_TO_LABEL[tag_idx]] = class_tag_cm[class_idx][tag_idx] / sum(class_tag_cm[:, tag_idx])

    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=class_tag_cm,
    #     display_labels=sorted_labels,
    #     #normalize=normalize,
    # )
    # disp.plot(cmap=plt.cm.Blues, )
    # #disp.ax_.set_title(title)
    # print(disp.confusion_matrix)

    # plt.show()

    # class_tag_corr: keys = class labels; values = tag labels
    # class_tag_corr = {k+"_lab": v for k, v in class_tag_corr.items()}
    class_tag_corr = pd.DataFrame.from_dict(dict(class_tag_corr)).fillna(0) # normalised wrt class label
    ax = sns.heatmap(pd.DataFrame.transpose(class_tag_corr), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel("tag labels")
    ax.set_ylabel("class labels")
    ax.set_title("Class Distribution for Each Tag", fontsize = 16)
    plt.show()

    class_tag_corr_norm_tag = pd.DataFrame.from_dict(dict(class_tag_corr_norm_tag)).fillna(0) # normalised wrt class label
    ax = sns.heatmap(pd.DataFrame.transpose(class_tag_corr_norm_tag), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel("tag labels")
    ax.set_ylabel("class labels")
    ax.set_title("Class Distribution for Each Tag "+("(normalized wrt tag)" if args.normalise else ""), fontsize = 16)
    plt.show()

    class_tag_corr_norm_cls = pd.DataFrame.from_dict(dict(class_tag_corr_norm_cls)).fillna(0) # normalised wrt class label
    ax = sns.heatmap(pd.DataFrame.transpose(class_tag_corr_norm_cls), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel("tag labels")
    ax.set_ylabel("class labels")
    ax.set_title("Class Distribution for Each Tag (norm wrt class)", fontsize = 16)
    plt.show()


# ----------------------------------plot class_tagname_corr-------------------------------- #

tagname_vs_class = defaultdict(lambda: defaultdict(lambda: int())) # pattern_name: {label: count}
if args.type=="matched-patterns":

    for label in class_tagname_corr:
        for name in class_tagname_corr[label]:
            tagname_vs_class[name][label] += class_tagname_corr[label][name]
            #tagname_vs_class[LABEL_PATTERNS_[name]][label] += class_tagname_corr[label][name]
    
    if args.normalise:
        for label in class_tagname_corr:
            label_cnt = 0
            for name in class_tagname_corr[label]:
                if tagname_counts[name]:
                    # tagname_vs_class[name][label] /= tagname_counts[name] # normalised wrt tag name
                    label_cnt += class_tagname_corr[label][name]
                    class_tagname_corr[label][name] /= class_count[label]
            print(f"label_cnt: {label_cnt}, class_count[{label}]: {class_count[label]}")
            
                
        for name in tagname_vs_class:
            name_cnt = 0
            for label in tagname_vs_class[name]:
                #class_tagname_corr[label][name] /= class_count[label] # normalised wrt class label
                name_cnt += tagname_vs_class[name][label]
                tagname_vs_class[name][label] /= tagname_counts[name] # normalised wrt class label
            print(f"name_cnt: {name_cnt}, tagname_counts[{name}]: {tagname_counts[name]}")
            
    class_tagname_corr = pd.DataFrame.from_dict(dict(class_tagname_corr)).fillna(0) # normalised wrt class label
    tagname_vs_class = pd.DataFrame.from_dict(dict(tagname_vs_class)).fillna(0)
    # tag_vs_tagname = tag_vs_tagname.where(tag_vs_tagname>10).fillna(0)
    # print(tagname_vs_tag)
    # print(tag_vs_tagname)

    ax = sns.heatmap(pd.DataFrame.transpose(tagname_vs_class), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    #ax = sns.heatmap((tagname_vs_class), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("Class distribution for each pattern "+(" (normalized wrt pattern type)" if args.normalise else ""), fontsize = 16)
    plt.show()

    #ax = sns.heatmap(pd.DataFrame.transpose(class_tagname_corr), annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    ax = sns.heatmap(class_tagname_corr, annot=True, cmap='crest', xticklabels=True, yticklabels=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("Pattern distribution per for each class"+(" (normalized wrt class label)" if args.normalise else ""), fontsize = 16)
    plt.show()

    
    # pprinter.pprint(class_tagname_corr)
    # all_tags=[]
    # for label in class_tagname_corr:
    #     for name in class_tagname_corr[label]:
    #         all_tags.append(name)

    # class_tagname_corr_cm = np.zeros((len(all_tags), num_labels))

    # for label_idx, label in enumerate(class_tagname_corr):
    #     for name_idx, pat_name in enumerate(class_tagname_corr[label]):
    #         class_tagname_corr_cm[name_idx][label_idx] = class_tagname_corr[label][pat_name]

    # print(class_tagname_corr_cm, class_tagname_corr_cm.shape)
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(class_tagname_corr_cm)

    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(all_tags)), labels=all_tags)
    # ax.set_yticks(np.arange(len(class_tagname_corr)), labels=class_tagname_corr.keys())

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(class_tagname_corr)):
    #     for j in range(len(all_tags)):
    #         text = ax.text(i, j, class_tagname_corr_cm[j, i],
    #                     ha="center", va="center", color="w")

    # ax.set_title("class-tagname correlation")
    # fig.tight_layout()
    # plt.show()





