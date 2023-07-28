import numpy as np
import matplotlib.pyplot as plt

import argparse
import jsonlines
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay

MODEL = "SciBERT-mt"
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()

def normalize_cm_pred(confusion_matrix, round=None):
    row_sums = confusion_matrix.sum(axis=1)
    normed = confusion_matrix / row_sums
    if round is not None:
        return np.around(normed, 2)
    return normed

def normalize_cm_true(confusion_matrix, round=None):
    col_sums = confusion_matrix.sum(axis=0)
    normed = confusion_matrix / col_sums
    if round is not None:
        return np.around(normed, 2)
    return normed



def main():

    args = build_args()
    data = list(jsonlines.open(args.in_file))
    
    gold = [ex[0] for line in data for ex in line['sent_labels']]
    pred = [ex[0] for line in data for ex in line['pred_sent_labels']]

    with open(args.label_file) as f:
        LABEL_TO_INDEX = {label.strip(): i for i, label in enumerate(f.readlines())}
    INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

    assert len(gold) == len(pred)

    num_labels = len(LABEL_TO_INDEX)
    cm = np.zeros((num_labels, num_labels))

    # Plot non-normalized confusion matrix
    # for i, paragraph in enumerate(tqdm(gold)):
    #     parag_probs = pred[i]['probs']
    #     for j, sent in enumerate(paragraph):
    #         pred_label = np.argmax(np.asarray(parag_probs[j]))
    #         gold_label = LABEL_TO_INDEX[sent['label']]
    #         cm[gold_label][pred_label] += 1
    
    #breakpoint()
    for i, gl in enumerate(gold):
        pl = pred[i]
        cm[gl][LABEL_TO_INDEX[pl]] += 1

    
    # for i in range(num_labels):
    #     cm[i][i] = 0
    pred_norm_cm = normalize_cm_pred(cm, round=2)
    true_norm_cm = normalize_cm_true(cm, round=2)

    title_options = [
        (f"Normalized against True Labels", true_norm_cm),
        (f"Normalized against Predicted Labels", pred_norm_cm),
    ]

    fig, axs = plt.subplots(1, 2)
    
    class_names = [INDEX_TO_LABEL[i] for i in range(num_labels)]
    #class_names = [i for i in range(num_labels)]
    #pred = [LABEL_TO_INDEX[i] for i in pred]
    gold = [INDEX_TO_LABEL[i] for i in gold]
    for i, (title, cm_) in enumerate(title_options):
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=cm,
        #     display_labels=class_names,
        #     ax=axs[i],
        #     cmap=plt.cm.Blues,
        #     #normalize=normalize, # ERROR :(
        # )
        # axs[i].set_title(title)
        # #disp.plot(cmap=plt.cm.Blues)
        # #disp.ax_.set_title(title) # https://github.com/scikit-learn/scikit-learn/discussions/20690#discussioncomment-1139252
        # print(title)
        # print(disp.confusion_matrix)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=gold,
            y_pred=pred,
            labels=class_names,
            normalize=title.split()[-2][:4].lower(),
            cmap=plt.cm.Blues,
            display_labels=class_names,
            ax=axs[i],
        )
        #axs[i].set_title(title)
        disp.ax_.set_title(title)
    print(fig.axes)
    fig.delaxes(fig.axes[2]) # the colorbar is actually a sepearate ax / plot lol. 
    # So we simply print each ax out and delete the one we dont' want
    #https://matplotlib-users.narkive.com/KFBDKVY3/how-to-remove-colorbar#post2
    fig.suptitle(f'Confusion Matrices of {MODEL}', fontsize=16)
    plt.tight_layout()
    fig.show()

    ######## Un-normalised ## Confusion ## Matrix ##################
    title = (f"Confusion matrix of {MODEL}")
    disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
            #cmap=plt.cm.Blues,
            #normalize=normalize,
        )
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    print(title)
        
    plt.show()


if __name__=="__main__":
    main()