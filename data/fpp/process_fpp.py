import os
import re
import json
import jsonlines
import spacy

from pprint import PrettyPrinter

DEBUG = True
pp = PrettyPrinter()

def flatten_dict(d, _keys=()):
    if not isinstance(d, dict):
        return d
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_children = flatten_dict(v, _keys=_keys+(k,))
            res.update(flat_children)
        else:
            res[_keys+(k,)] = v
    return res

class FpProcessor(object):

    # To test: `pytest tests/data/test_process_fpp.py`
    # patterns
    def __init__(self,):
        self.main_categ = re.compile(r"[A-Z]\.\s([A-Z /\.]+)(\s*)") # e.g. "D. EXPRESSING CAUSE/EFFECT"
        self.main_func = re.compile(r"\d\. ([A-Z])([a-z \.]+)(\s*)") # e.g. "1. Expressing cause"
        self.sub_func = re.compile(r"\d\.\d ([A-Z])([a-z \.]+)(\s*)") # e.g. "1.1 Using conjunctions such as because and since"
        self.annotator = re.compile(r"\(.*?\)") #re.compile(r"\(([a-zA-Z]+)\)") doesn't work, dunno why
        self.fp = re.compile(r"\[\[.*?\]\]") # .*? = greedy-match everything
        self.key_in_fp = re.compile(r"\[.*?\]") # used after a fp match is found
        self.TITLE_PLACEHOLDER = "Title placeholder"

    def is_not_sure(self, line):
        if line[0]=="?":
            return True
        return False
    
    def parse_nested_patterns_whitespace_tok(self, line):
        """
        Tokenise line by white space AND punctuations (. , ? ! " ' \ )
        We don't use spacy tokenizer because it views [ and ] as individual tokens
        Records which indices of tokens that are marked with '[' and ']'

        Input
        line: List[str]. A white-space tokenised line. Literally only processed with line = line.split()
        (other preprocessing may be done to separate the punctuations too, but not '[' and ']', from each token)
        
        Output
        full_patterns: List[Dict]. Each dict in the list contains the start and end index of the 
            pattern wrt the input sentence. `is_nested` indicates whether the pattern is nested 
            within another pattern or not.
        """
        line = line.split()
        full_patterns = []
        stack = []
        for i, tok in enumerate(line):
            if tok.startswith("["):
                stack.append({"start": i})
            if tok.endswith("]"):
                if stack:
                    is_nested = (len(stack)>1)
                    start = stack.pop()
                    full_patterns.append({
                        "start": start["start"],
                        "end": i,
                        "is_nested": is_nested
                        })
                        
        return full_patterns
                

    def parse_nested_patterns_spacy_tok(self, line):
        """
        Identify annotated pattern within a spacy-tokenised sentence. 
        Annotated pattern is marked by a set of square brackets '[' and ']'

        Input
        line: spacy.tokens.doc.Doc, an iterable of tokens
        Output
        full_patterns: List[Dict]. Each dict in the list contains the start and end index of the 
            pattern wrt the input sentence. `is_nested` indicates whether the pattern is nested 
            within another pattern or not.
        """
        full_patterns = []
        stack = []
        no_bracket_indices = []
        cur_no_bracket_idx = 0
        prev_is_start = False
        prev_is_nested = False
        for i, tok in enumerate(line):
            if tok.text == "[":
                if prev_is_start:
                    prev_is_nested = True
                prev_is_start = True
                continue

            if prev_is_start:
                stack.append({"start": i, "no_bracket_start": cur_no_bracket_idx, "prev_is_nested": prev_is_nested})
                if prev_is_nested:
                    stack.append({"start": i, "no_bracket_start": cur_no_bracket_idx, "prev_is_nested": False})
                    prev_is_nested = False
                prev_is_start = False

            if tok.text == "]":
                if stack:
                    is_nested = (len(stack)>1)
                    start = stack.pop()
                    full_patterns.append({
                        "start": start["start"],
                        "end": i,
                        "no_bracket_start": start["no_bracket_start"],
                        "no_bracket_end": cur_no_bracket_idx,
                        "is_nested": (is_nested or not start["prev_is_nested"])
                        })
                continue

            no_bracket_indices.append(i)
            cur_no_bracket_idx += 1

        no_bracket_text = [line[i].text for i in no_bracket_indices]
        #breakpoint()
        return {"patterns": full_patterns, "no_bracket_text": no_bracket_text,}
                


    def parse_fp(self, line):
        """Extract marked phrase and words marked within from input sentence

        Input
        line: sentence containing one or more annotated phrase. 
            An annotated phrase should include at lease one pair of brackets '[]' embedded in a pair of outer brackets '[]'
                Example: 
                Planning controls operate in rural areas [[in the same way as]] in urban areas. (JS)
                [[Like] many others [,]] Berkeley objected to the complete materialism of Hobbes. (JS)
            
        Output
        plaintext_fp
        phrases (how to represent the embedded word(s)? span?)
        annotator
        """
        assert not line[0].isdigit
        assert not line[1] == '.'
        
        matched_fps = re.finditer(self.fp, line)
        for match in matched_fps:
            matched_str = match.group(0)[1:-1]
            start, end = match.start()+1, match.end()-1 # i should be using tokenised outputs for matching instead...
            key_fp = re.findall(self.key_in_fp, matched_str)
            ...


    def file_to_dict(self, in_file):
        """Turns raw input file into structured dictionary. Note that the "text" field is not yet parsed,
        i.e., the annotated spans are still marked with square brackets. This requires further processing.
        
        I use a nested dict instead of a defaultdict or 1 dict for each example, because it is 
        easier to detect errors in the hierarchy when parsing the file as key errors. 
        You may change the returned dict into something easier to access later (using e.g. `flatten_dict`)

        Input: 
        in_file[str]: file name

        Output:
        out_dict: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]]
        out_dict is structured hierarchically:
        First layer of out_dict is the MAIN CATEGORY
        Second layer of out_dict is the MAIN FUNCTION
        Third layer of out_dict is the SUB FUNCTION
        Under SUB FUNCTION is a list
        In each list if a dict: {"annotator": [str], "text": [str]}
        """
        out_dict = {}
        cur_main_categ = ""
        cur_main_func = ""
        cur_sub_func = ""
        with open(in_file) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if not line:
                    line = f.readline()
                    continue

                main_categ_match = re.match(self.main_categ, line)
                main_func_match = re.match(self.main_func, line)
                sub_func_match = re.match(self.sub_func, line)
                
                if DEBUG:
                    print(f"line: {line}")
                    print(f"main_categ_match: {main_categ_match}")
                    print(f"main_func_match: {main_func_match}")
                    print(f"sub_func_match: {sub_func_match}")
                    print(flatten_dict(out_dict).keys())

                    print(f"cur_main_categ: {cur_main_categ}")
                    print(f"cur_main_func: {cur_main_func}")
                    print(f"cur_sub_func: {cur_sub_func}")
                if main_categ_match is not None:
                    if DEBUG: print(f"Matched main_categ: {main_categ_match}")
                    cur_main_categ = main_categ_match.group(0)
                    out_dict[cur_main_categ] = {}

                elif main_func_match is not None:
                    if DEBUG: print(f"Matched main_func: {main_func_match}")
                    cur_main_func = main_func_match.group(0)
                    out_dict[cur_main_categ][cur_main_func] = {}
                    
                elif sub_func_match is not None:
                    if DEBUG: print(f"Matched sub_func: {sub_func_match}")
                    cur_sub_func = sub_func_match.group(0)
                    out_dict[cur_main_categ][cur_main_func][cur_sub_func] = []
                    
                else:
                    annotator_match = re.findall(self.annotator, line)
                    example_dict = {}
                    if annotator_match != []:
                        example_dict['annotator'] = re.findall(self.annotator, line)[0][1:-1]
                    else:
                        example_dict['annotator'] = ""
                    
                    if line.split("("):
                        if len(line.split("(")) >= 2: # more than 1 ( in line, split by final one
                            example_dict["text"] = "".join(line.split("(")[:-1]).strip()
                        elif len(line.split("(")) == 1: # no "(" in the line, i.e. no annotator
                            example_dict["text"] = line.strip()
                    elif line.isspace():
                        continue
                    else:
                        raise RuntimeError
                    assert example_dict["text"]

                    try:
                        out_dict[cur_main_categ][cur_main_func][cur_sub_func].append(example_dict)
                    except KeyError:
                        if self.TITLE_PLACEHOLDER not in out_dict[cur_main_categ][cur_main_func].keys():
                            out_dict[cur_main_categ][cur_main_func][self.TITLE_PLACEHOLDER] = []
                        else:
                            out_dict[cur_main_categ][cur_main_func][self.TITLE_PLACEHOLDER].append(example_dict)
                
                if DEBUG: print(out_dict.keys())
                if DEBUG: print("\n"+"="*20+" current line done "+ "="*20+"\n")
                line = f.readline()
            return out_dict


def main():
    in_file = "./FPP.raw.txt"
    fpp = FpProcessor()

    intermediate_file = "intermediate_data.json"
    if not os.path.exists(intermediate_file):
        out_d = fpp.file_to_dict(in_file)
        with open(intermediate_file, "w") as f:
            json.dump(out_d, f)
    else:
        with open(intermediate_file) as f:
            out_d = json.load(f)
    
    # Turn the output dict into something easier to manage
    flattened_d = flatten_dict(out_d)
    out_d_list = []
    for key_tp, example_l in flattened_d.items():
        main_categ, main_func, sub_func = key_tp
        for example_d in example_l:
            new_d = {
                "main_category": main_categ,
                "main_function": main_func,
                "sub_function": sub_func,
                "text": example_d["text"],
                "annotator": example_d["annotator"]
            }
            out_d_list.append(new_d)

    # Parse patterns from each sentence
    print("Loading spacy...")
    nlp = spacy.load("en_core_web_lg", exclude = ['ner', 'custom', 'textcat'])
    print("Spacy loaded")

    for example in out_d_list:
        line = nlp(example["text"])
        parsed_res = fpp.parse_nested_patterns_spacy_tok(line)
        example.update(parsed_res)

    with jsonlines.open("parsed_fpp.jsonl", "w") as f:
        f.write_all(out_d_list)
    print("Saved to parsed_fpp.jsonl")
    


    

if __name__=="__main__":
    main()