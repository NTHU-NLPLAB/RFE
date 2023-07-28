import jsonlines
import os
import sys

def main(in_dir, out_dir):
    data = {}
    splits = ["train", "dev", "test"]

    for split in splits:
        data[split] = list(jsonlines.open(os.path.join(in_dir, split+".jsonl")))
        data[split] = [line for parag in data[split] for line in parag]

    target_dir = os.path.join(out_dir, "sents")
    if os.path.exists(target_dir):
        print(f"{target_dir} exists! Aborting...")
        raise FileExistsError
    else:
        os.mkdir(target_dir)

    for split in splits:
        with jsonlines.open(os.path.join(target_dir, split+".jsonl"), mode="w") as f:
            f.write_all(data[split])
        print(f"{split} ({len(data[split])} lines) written to {os.path.join(target_dir, split)}.jsonl !")

if __name__=="__main__":
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    main(in_dir, out_dir)
    
