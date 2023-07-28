import jsonlines

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    # source: https://github.com/dead/rhetorical-structure-pubmed-abstracts

    in_dir = Path("./pubmed/parag")
    
    # line['paragraph'] is a list of dicts: [{'text': ..., "label": ...}, {...}, ...]
    data_train = [line for line in jsonlines.open(in_dir / "train.jsonl")]
    data_dev = [line for line in jsonlines.open(in_dir / "dev.jsonl")]

    dummy_y_tr = [-1 for i in range(len(data_train))]
    dummy_y_dev = [-1 for i in range(len(data_dev))]

    _, X_toy_tr, _, _ \
        = train_test_split(data_train, dummy_y_tr, test_size=0.05, random_state=42)
    
    _, X_toy_dev, _, _ \
        = train_test_split(data_dev, dummy_y_dev, test_size=0.05, random_state=42)

    toy_tr_out = []
    for paragraph in X_toy_tr:
        toy_tr_out.append(paragraph)

    toy_dev_out = []
    for paragraph in X_toy_dev:
        toy_dev_out.append(paragraph)
    
    with jsonlines.open(in_dir / "toy_train.jsonl", "w") as f:
        f.write_all(toy_tr_out)
    print(f"Toy training subset saved at {in_dir}/toy_train.jsonl")

    with jsonlines.open(in_dir / "toy_dev.jsonl", "w") as f:
        f.write_all(toy_dev_out)
    print(f"Toy development subset saved at {in_dir}/toy_dev.jsonl")

if __name__ == '__main__':
    main()