import jsonlines
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Split pubmed by abstract (paragraph) instead of by individual sentences

def format_out(X, y):
    out_data = []
    for paragraph, paragraph_lab in zip(X, y):
        out_paragraph = []
        for tp, lab in zip(paragraph, paragraph_lab):
            out_paragraph.append({"label": lab, "text": tp[0], "fileno": tp[1], 'header': tp[2], "s_id": tp[3]})
        out_data.append(out_paragraph)
    return out_data

def main(in_file, out_dir):
    
    # source: https://www.cl.cam.ac.uk/~sht25/az.html
    #data = list(jsonlines.open('./az/az_papers_all.jsonl'))
    data = list(jsonlines.open(in_file))
    
    X_parag = []
    y_parag = []
    for parag in tqdm(data):
        X = []
        y = []
        for it in parag:
            X.append([it['text'], it['fileno'], it['header'], it['s_id']])
            y.append(it['label'])
        X_parag.append(X)
        y_parag.append(y)

    X_train, X_test, y_train, y_test \
        = train_test_split(X_parag, y_parag, test_size=0.2, random_state=42)

    X_test, X_dev, y_test, y_dev \
        = train_test_split(X_test, y_test, test_size=0.5, random_state=42) # 0.2 * 0.5 = 0.1
    # We end up with a 80, 10, 10 split

    train_out = format_out(X_train, y_train)
    dev_out = format_out(X_dev, y_dev)
    test_out = format_out(X_test, y_test)
    
    #out_dir = Path("./az/az_papers")
    out_dir = Path(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

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
    in_file, out_dir = sys.argv[1], sys.argv[2]
    main(in_file, out_dir)