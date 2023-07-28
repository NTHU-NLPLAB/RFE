import jsonlines

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    # source: https://github.com/dead/rhetorical-structure-pubmed-abstracts

    train_file = "./data/pubmed/train.jsonl"
    dev_file = "./data/pubmed/dev.jsonl"
    
    X_train = [line['text'] for line in jsonlines.open(train_file)]
    y_train = [line['label'] for line in jsonlines.open(train_file)]

    X_dev = [line['text'] for line in jsonlines.open(dev_file)]
    y_dev = [line['label'] for line in jsonlines.open(dev_file)]


    _, X_toy_tr, _, y_toy_tr \
        = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
    
    _, X_toy_dev, _, y_toy_dev \
        = train_test_split(X_dev, y_dev, test_size=0.05, random_state=42)

    toy_tr_out = []
    for text, lab in zip(X_toy_tr, y_toy_tr):
        toy_tr_out.append({"text": text, "label": lab})

    toy_dev_out = []
    for text, lab in zip(X_toy_dev, y_toy_dev):
        toy_dev_out.append({"text": text, "label": lab})
    
    with jsonlines.open("./data/pubmed/toy_train.jsonl", "w") as f:
        f.write_all(toy_tr_out)
    print("Toy training subset saved at ./data/pubmed/toy_train.jsonl")

    with jsonlines.open("./data/pubmed/toy_dev.jsonl", "w") as f:
        f.write_all(toy_dev_out)
    print("Toy development subset saved at ./data/pubmed/toy_dev.jsonl")

if __name__ == '__main__':
    main()