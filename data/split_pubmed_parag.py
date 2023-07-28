import jsonlines
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Split pubmed by abstract (paragraph) instead of by individual sentences

def format_out(X, y):
    out_data = []
    for paragraph, paragraph_lab in zip(X, y):
        out_paragraph = []
        for text, lab in zip(paragraph, paragraph_lab):
            out_paragraph.append({"text": text, "label": lab})
        out_data.append(out_paragraph)
    return out_data

def main():
    
    # source: https://github.com/dead/rhetorical-structure-pubmed-abstracts
    with open('./pubmed/abstracts.pickle', 'rb') as f:
        data = pickle.load(f)
    
    X_abstr = []
    y_abstr = []
    for abstr in tqdm(data):
        X = []
        y = []
        for tp in abstr:
            X.append(tp[0])
            y.append(tp[1])
        X_abstr.append(X)
        y_abstr.append(y)

    X_train, X_test, y_train, y_test \
        = train_test_split(X_abstr, y_abstr, test_size=0.2, random_state=42)

    X_test, X_dev, y_test, y_dev \
        = train_test_split(X_test, y_test, test_size=0.5, random_state=42) # 0.2 * 0.5 = 0.1
    # We end up with a 80, 10, 10 split

    train_out = format_out(X_train, y_train)
    dev_out = format_out(X_dev, y_dev)
    test_out = format_out(X_test, y_test)
    
    out_dir = Path("./pubmed/parag/")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # parag = paragraph
    with jsonlines.open(out_dir / "train.jsonl", "w") as f:
        f.write_all(train_out)
    print(f"Training set saved at {out_dir}/train.jsonl")

    with jsonlines.open(out_dir / "dev.jsonl", "w") as f:
        f.write_all(dev_out)
    print(f"Development set saved at {out_dir}/dev.jsonl")

    with jsonlines.open(out_dir / "test.jsonl", "w") as f:
        f.write_all(test_out)
    print(f"Test set saved at {out_dir}/test.jsonl")

if __name__ == '__main__':
    main()