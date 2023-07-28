import os
from pathlib import Path

import numpy as np
import jsonlines
import argparse

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import wandb

from tqdm import tqdm

from hf_finetune import compute_metrics, convert_label_to_index, tokenize_function, init_tokenizer, LABEL_TO_INDEX, IDX_TO_LABEL


os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
wandb.init(project="huggingface")

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_is_split_into_words", action="store_true")
    parser.add_argument("--dataset", type=str, default="az")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--inference_file", type=str, default="../../data/az+s2/bio/s2_stitched_train_head_100000.jsonl")
    #parser.add_argument("--inference_file", type=str, default="../../data/az+s2/bio/test_100.jsonl")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--checkpoint", type=str, default="./models/checkpoint-2500")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=3)

    return parser.parse_args()

# functions

def main():
    args = build_args()
    data_dir = Path(args.data_dir)
    DATASET=args.dataset
    if DATASET=="pubmed":
        num_labels=5
        #data_dir = os.path.join("data", DATASET, "sents") # change to parag if needed
    elif DATASET=="az":
        num_labels=7
        #data_dir = os.path.join("data", DATASET, "az_papers", "sents") # change to parag if needed
    else:
        raise NotImplementedError

    # do the data loader thing
    test_path = data_dir / "test.jsonl"
    if Path.exists(test_path):
        data_files = {'test': test_path}
    else:
        if os.path.exists(args.inference_file):
            data_files = {'test': args.inference_file}
        else:
            raise FileNotFoundError

    
    for split, path in data_files.items():
        data_files[split] = list(jsonlines.open(path))
        if 'label' in data_files[split][0].keys(): # hacky. i don't like it >:(
            data_files[split] = convert_label_to_index(data_files[split], DATASET)

    init_tokenizer(args.tokenizer)
    dataset = DatasetDict()
    for k, v in data_files.items():
        dataset[k] = Dataset.from_list(v)
        dataset[k] = dataset[k].map(tokenize_function, batched=True, fn_kwargs = {"args": args})
    #tokenized_datasets = dataset.map(tokenize_function, batched=True)
    #shuffled_train_dataset = dataset["train"]
    #eval_dataset = dataset["validation"]
    eval_dataset = dataset['test']

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
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
        #per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        metric_for_best_model = 'accuracy',
        load_best_model_at_end=True,
    )

    # init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        #train_dataset=shuffled_train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    
    predictions = trainer.predict(dataset['test'])
    probfile_dir = Path(training_args.output_dir) / "predictions"
    probfile_dir.mkdir(parents=True, exist_ok=True)
    np.savez(probfile_dir / "predictions.npz", probs=np.asanyarray(predictions), dtype=object)
    print(f"Written probs to {probfile_dir}")

    
    predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    output_predict_file = probfile_dir / "predictions.jsonl"
    with jsonlines.open(output_predict_file, "w") as f:
        for i, line in enumerate(tqdm(dataset['test'])):
            out_d = {'text': line['text'], 
                     'seq_tag': line['seq_tag'], 
                     'pattern_indices': line['pattern_indices'],
                     'patterns': line['patterns'],
                     "label": IDX_TO_LABEL[DATASET][int(predictions[i])],
                     'is_az': False,
                     }
            f.write(out_d)
    print("Written postprocessed preds to", output_predict_file)


if __name__=="__main__":
    main()
