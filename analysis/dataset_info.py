import sys
import jsonlines

def recursive_flatten_to_dicts(in_list):
    # TODO: modify this to suit az papers format
    """
    Recursively iterates through a list of lists of lists of..., whose
    bottom layer is a dict, and flatten all that into a list of dict.
    """
    out_list = []

    for it in in_list:
        if isinstance(it, list):
            out_list.extend(recursive_flatten_to_dicts(it))
        elif isinstance(it, dict): # the item is a dict => reached final layer
            out_list.append(it)
    return out_list

def print_class_info(dataset, labels):
    # get all classes
    # get count of examples for each class
    # get percentage of examples for each class
    classes = {label: 0 for label in labels}
    for line in dataset:
        classes[line['label']] += 1
    
    total = len(dataset)
    
    for label, cnt in classes.items():
        print(f"{label}:\t{cnt},\t{cnt/total:.2%}")

    return

def main(filepath):
    dataset = list(jsonlines.open(filepath))
    dataset = recursive_flatten_to_dicts(dataset)
    dataset = [line for line in dataset if line['label']]
        
    labels = sorted(list(set(line['label'] for line in dataset)))

    print(f"Dataset size: {len(dataset)} examples\n"\
        f"Labels: {labels}")

    print_class_info(dataset, labels)

if __name__=="__main__":
    filepath = sys.argv[1]
    main(filepath)