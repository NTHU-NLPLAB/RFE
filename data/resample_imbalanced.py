import jsonlines
import argparse
import pathlib

from random import seed, sample
from collections import Counter
import numpy as np
from imblearn.over_sampling import RandomOverSampler

# ref: 
# https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--task", default="classification", choices=["classification", "sequence_labelling",],)
    parser.add_argument(
            "--method",
            default="oversample_to_maj",
            choices=["oversample_to_maj", "sub_after_oversample", "oversample_to_avg",],
        )
    parser.add_argument("--out_file", type=str, required=True)

    return parser.parse_args()

def seperate_Xy(args, data):
    """return y = line['label'], X = [{line[everything else]}]
    """
    if args.task == "classification":
        y_key = "label"
    elif args.task == "sequence_labelling":
        y_key = "tagged_label"

    #X = np.asarray([[ex["text"], ex["fileno"]] for parag in data for ex in parag])
    if isinstance(data[0], list):
        X = np.asarray([[{k: ex[k] for k in ex.keys() if k != y_key}] for parag in data for ex in parag])
        y = [ex[y_key] for parag in data for ex in parag]
    elif isinstance(data[0], dict):
        X = np.asarray([[{k: ex[k] for k in ex.keys() if k != y_key}] for ex in data])
        y = [ex[y_key] for ex in data]
    else:
        raise TypeError
    return X, y

def get_sequence_labelling_data(data):
    """return y = line['tagged_label'], X = [{line[everything else]}]
    """
    #X = np.asarray([[ex["text"], ex["fileno"]] for parag in data for ex in parag])
    if isinstance(data[0], list):
        X = np.asarray([[{k: ex[k] for k in ex.keys() if k != "tagged_label"}] for parag in data for ex in parag])
        y = [ex["tagged_label"] for parag in data for ex in parag]
    elif isinstance(data[0], dict):
        X = np.asarray([[{k: ex[k] for k in ex.keys() if k != "tagged_label"}] for ex in data])
        y = [ex["tagged_label"] for ex in data]
    else:
        raise TypeError
    return X, y


def main():
    args = build_args()
    data = list(jsonlines.open(args.in_file))

    
    X, y = seperate_Xy(args, data)

    orig = Counter(y)
    print(f"Original data: {orig}")
    num_examples = len(y)
    maj_label = orig.most_common()[0][0]
    all_labels = set(y)

    seed(42)
    # oversample minor classes

    if args.method == "oversample_to_avg":
        avg = int(num_examples / len(all_labels))
        sampling_dict = {label: avg for label in all_labels if (orig[label] < avg)}
        #sampling_dict[maj_label] = orig[maj_label] # keep majority class

        oversampler = RandomOverSampler(sampling_strategy=sampling_dict)
        X_over, y_over = oversampler.fit_resample(X, y)
        print(f"After oversampling: {Counter(y_over)}")
    else:
        oversampler = RandomOverSampler(sampling_strategy="not majority")
        X_over, y_over = oversampler.fit_resample(X, y)
        print(f"After oversampling: {Counter(y_over)}")

        if args.method == "sub_after_oversample":
            # subsample all classes to get original dataset size
            sub_idx = sample(list(range(len(X_over))), k=num_examples)
            X_over, y_over = [X_over[i] for i in sub_idx] , [y_over[i] for i in sub_idx]
            print(f"After subsampling from oversampled examples: {Counter(y_over)}")
        elif args.method == "oversample_to_maj":
            pass

    if args.task == "classification":
        y_key = "label"
    elif args.task == "sequence_labelling":
        y_key = "tagged_label"
    out_data = [
            {y_key: y_over[i],
                **X_over[i][0]
            } for i in range(len(X_over))
        ]
    
    
    out_file = pathlib.Path(args.out_file)
    with jsonlines.open(out_file, "w") as f:
        f.write_all(out_data)
        
    print(f"Saved resampled examples to {args.out_file}")



if __name__=="__main__":
    main()