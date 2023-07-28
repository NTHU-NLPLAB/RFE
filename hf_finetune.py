import os

import numpy as np
import jsonlines
import argparse

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# globals
LABEL_TO_INDEX = {
    "pubmed": {
        'BACKGROUND': 0,
        'OBJECTIVE': 1,
        'METHODS': 2,
        "RESULTS": 3,
        "CONCLUSIONS": 4,
    },
    "az": {
        'AIM': 0,
        'BAS': 1,
        'BKG': 2,
        "CTR": 3,
        "OTH": 4,
        "OWN": 5,
        "TXT": 6
    }
}

IDX_TO_LABEL = {
        "az": {
            0: "AIM",
            1: "BAS",
            2: "BKG",
            3: "CTR",
            4: "OTH",
            5: "OWN",
            6: "TXT"
        }
    }

wandb.init(project="huggingface")


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_is_split_into_words", action="store_true")
    parser.add_argument("--dataset", type=str, default="az")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--oversample", action="store_true")

    return parser.parse_args()

# functions

tokenizer = None

def init_tokenizer(model):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

def tokenize_function(examples, args=None):
    return tokenizer(examples["text"],
                     padding="max_length",
                     max_length=128,
                     truncation=True,
                     is_split_into_words=args.data_is_split_into_words
                    )


# def prepare_compute_metrics(label_list):
#     def compute_metrics(eval_pred):
#         nonlocal label_list
#         ...
#     return compute_metrics
    
# compute_metrics = prepare_compute_metrics(label_list)


def compute_metrics(eval_pred):
    logits, labels = eval_pred # HF's model.predict() returns 
    # Depending on the dataset and your use case, your test dataset may 
    # contain labels. In that case, this method will also return metrics,
    #  like in evaluate().
    # from docs:https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.Trainer.predict
    # predictions (np.ndarray): The predictions on test_dataset.
    # label_ids (np.ndarray, optional): The labels (if the dataset contained some).
    # metrics (Dict[str, float], optional): The potential dictionary of metrics (if the dataset contained labels).
    pred = np.argmax(logits, axis=1)
    _IDX_TO_LABEL = IDX_TO_LABEL["az"]
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average=None)
    precision = precision_score(y_true=labels, y_pred=pred, average=None)
    f1 = f1_score(y_true=labels, y_pred=pred, average=None)
    f1_weighted = f1_score(y_true=labels, y_pred=pred, average='weighted')
    prec_weighted = precision_score(y_true=labels, y_pred=pred, average='weighted')
    reca_weighted = recall_score(y_true=labels, y_pred=pred, average='weighted')
    res = {
        "accuracy": accuracy.tolist(),
        "overall_f1_weighted": f1_weighted.tolist(),
        "overall_precision_weighted": prec_weighted.tolist(),
        "overall_recall_weighted": reca_weighted.tolist()
        #"precision": precision.tolist(),
        #"recall": recall.tolist(),
        #"f1": f1.tolist()
        }
    class_res = {}
    for i, (r, p, f) in enumerate(zip(recall, precision, f1)):
        class_res[_IDX_TO_LABEL[i]+"_recall"] = r
        class_res[_IDX_TO_LABEL[i]+"_precision"] = p
        class_res[_IDX_TO_LABEL[i]+"_f1"] = f
    res.update(class_res)
    wandb.log(res)
    return res


def convert_label_to_index(jsonl_dataset, dataset_name):
    """turns labels (str) into indices (int) in-place
    """
    for line in jsonl_dataset:
        line["label"] = LABEL_TO_INDEX[dataset_name][line["label"]]
    return jsonl_dataset


def main():
    args = build_args()
    init_tokenizer(args.tokenizer)
    data_path = args.data_dir
    DATASET=args.dataset
    if DATASET=="pubmed":
        num_labels=5
        #data_path = os.path.join("data", DATASET, "sents") # change to parag if needed
    elif DATASET=="az":
        num_labels=7
        #data_path = os.path.join("data", DATASET, "az_papers", "sents") # change to parag if needed
    else:
        raise NotImplementedError

    # do the data loader thing
    data_files = {'train': os.path.join(data_path, 'train.jsonl'),\
                'validation': os.path.join(data_path, 'dev.jsonl'),\
                'test': os.path.join(data_path, 'test.jsonl')
                }
    if args.oversample:
        data_files['train'] = os.path.join(data_path, "train_resampled.jsonl")

    # dataset = load_dataset('json', data_files = data_files)
    for split, path in data_files.items():
        data_files[split] = list(jsonlines.open(path))
        data_files[split] = convert_label_to_index(data_files[split], args.dataset)

    dataset = DatasetDict()
    for k, v in data_files.items():
        dataset[k] = Dataset.from_list(v)
        dataset[k] = dataset[k].map(tokenize_function, batched=True, fn_kwargs={"args": args})
    #tokenized_datasets = dataset.map(tokenize_function, batched=True)
    if args.shuffle_train:
        shuffled_train_dataset = dataset["train"].shuffle(seed=42) #.select(range(1000))
    else:
        shuffled_train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
        )

    # https://huggingface.co/docs/transformers/training
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        #evaluation_strategy="epoch",
        evaluation_strategy = IntervalStrategy.STEPS, # "steps"
        eval_steps = 50, # Evaluation and Save happens every 50 steps
        save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
        learning_rate=args.lr,
        num_train_epochs = args.epochs,
        per_device_train_batch_size=args.train_batch_size, # 8 works okay for az but causes memory for pubmed?? UPDATE: it was bc someone else was using a gpu...
        per_device_eval_batch_size=args.eval_batch_size,
        metric_for_best_model = 'accuracy',
        load_best_model_at_end=True
    )

    # init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    trainer.train()
    if args.do_test:
        trainer.predict(dataset['test'])


if __name__=="__main__":
    main()
