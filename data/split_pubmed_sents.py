import jsonlines
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    # source: https://github.com/dead/rhetorical-structure-pubmed-abstracts
    with open('./data/pubmed/abstracts.pickle', 'rb') as f:
        data = pickle.load(f)
    
    X = []
    y = []
    for abstr in tqdm(data):
        for tp in abstr:
            X.append(tp[0])
            y.append(tp[1])

    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test, X_dev, y_test, y_dev \
        = train_test_split(X_test, y_test, test_size=0.5, random_state=42) # 0.2 * 0.5 = 0.1
    # We end up with a 80, 10, 10 split

    train_out = []
    for text, lab in zip(X_train, y_train):
        train_out.append({"text": text, "label": lab})
    
    dev_out = []
    for text, lab in zip(X_dev, y_dev):
        dev_out.append({"text": text, "label": lab})

    test_out = []
    for text, lab in zip(X_test, y_test):
        test_out.append({"text": text, "label": lab})
    
    with jsonlines.open("./data/pubmed/train.jsonl", "w") as f:
        f.write_all(train_out)
    print("Training set saved at ./data/pubmed/train.jsonl")

    with jsonlines.open("./data/pubmed/dev.jsonl", "w") as f:
        f.write_all(dev_out)
    print("Development set saved at ./data/pubmed/dev.jsonl")

    with jsonlines.open("./data/pubmed/test.jsonl", "w") as f:
        f.write_all(test_out)
    print("Test set saved at ./data/pubmed/test.jsonl")

if __name__ == '__main__':
    main()