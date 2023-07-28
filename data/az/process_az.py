from bs4 import BeautifulSoup
from tqdm import tqdm
from spacy.lang.en import English
import jsonlines
import os, sys
from pathlib import Path
import re


def clean_bs_text(line):
    """
    Returns bs datatype, not string!
    """
    if line.eqn:
        for eq in line.find_all("eqn"):
            eq.replace_with("EQN")
    if line.cref:
        for ref in line.find_all("cref"):
            ref.replace_with("CREF")
    if line.ref:
        for ref in line.find_all("ref"):
            ref.replace_with("CITATION")
    if line.refauthor:
        for ref in line.find_all("ref"):
            ref.replace_with("CITATION")

    replaced_str = line.get_text()
    replaced_str = re.sub("``", '"', replaced_str)
    replaced_str = re.sub("''", '"', replaced_str)
    line.string = replaced_str

    return line

def split_sents_get_labels(combined_sent):
    doc = nlp(combined_sent["text"])
    out_sents = []
    for sent in doc.sents:
        out_sents.append(
            {
            "text": sent.text,
            "label": combined_sent["label"],
            "s_id": combined_sent['s_id'],
            "header": combined_sent['header'],
            "fileno": combined_sent['fileno'],
            }
            )
    return out_sents

def save_to_pubmed_format(out_dir):
    out_dir = Path(out_dir)
    az = list(jsonlines.open(out_dir / "az_papers.jsonl"))

    az_abs = [it["abstract"] for it in az]
    with jsonlines.open(out_dir / "az_papers_abstract.jsonl", "w") as f:
        f.write_all(az_abs)
    print(f"Saved {len(az_abs)} abstracts to {out_dir}/az_papers_abstracts.jsonl")

    # doc['body'] = [
    # {'header':..., 'sentences': [{'text': ..., "label": ...}, {...}]}, {...}
    # ]
    az_body = [parag['sentences'] for doc in az for parag in doc["body"]]
    for parag in tqdm(az_body):
        for i, sent in enumerate(parag): 
            assert isinstance(sent, dict)
            
    with jsonlines.open(out_dir / "az_papers_body.jsonl", "w") as f:
        f.write_all(az_body)
    print(f"Saved {len(az_body)} body paragraphs to {out_dir}/az_papers_body.jsonl")

    az_all = az_abs + az_body
    with jsonlines.open(out_dir / "az_papers_all.jsonl", "w") as f:
        f.write_all(az_all)
    print(f"Saved {len(az_all)} abstracts + body paragraphs to {out_dir}/az_papers_all.jsonl")
    return




def main(in_dir, out_dir):
    # parse all data
    # TO DEBUG: recommend using read_az.ipynb
    docs = []

    print("Processing all az-scixml files...")
    no_label = 0
    for scixml in tqdm(os.listdir("./az_papers/raw")):
        if not scixml.endswith(".az-scixml"):
            continue
        doc = {}
        with open(os.path.join(".", "az_papers", "raw", scixml)) as f:
            soup = BeautifulSoup(f, 'html.parser', from_encoding='iso-8859-1')
            #soup = BeautifulSoup(f, features="xml", from_encoding='iso-8859-1')
            soup = soup.paper
            doc['title'] = soup.title.text
            doc['year'] = soup.year.text
            doc['fileno'] = soup.fileno.text
            doc['classification'] = soup.classification.text
            doc['abstract'] = []

            abstr = soup.abstract
            combined_sent = {"text": "", "label": "", "header": "abstract", "fileno": doc['fileno']}
            for abstr_idx, line in enumerate(abstr.find_all('a-s')):
                line = clean_bs_text(line)
                label = line.get('az')
                if line.get("type")=="ITEM":
                    # all ITEM lists in the dataset have the same labels, so not checking labels here
                    combined_sent['label'] = label
                    combined_sent['s_id'] = line.get("id")
                    combined_sent['text'] = "".join([combined_sent['text'], line.get_text()])

                    if abstr_idx==(len(abstr.find_all('a-s'))-1):
                        combined_sent = split_sents_get_labels(combined_sent)
                        doc['abstract'].extend(combined_sent)
                        combined_sent = {"text": "", "label": "", "header": "abstract", "fileno": doc['fileno']}
                        
                elif (line.get("type")!="ITEM") and combined_sent["text"]: # prev lines are combined items
                    combined_sent = split_sents_get_labels(combined_sent)
                    doc['abstract'].extend(combined_sent)
                    combined_sent = {"text": "", "label": "", "header": "abstract", "fileno": doc['fileno']}
                
                sent_dict = {'text': line.get_text(), 'label': label, "header": "abstract", 'fileno': doc['fileno'], "s_id": line.get("id")}
                doc['abstract'].append(sent_dict)
            
            doc['body'] = []
            # doc['body'] = [
            # {'header':..., 'sentences': [{'text': ..., "label": ...}, {...}]}, {...}
            # ]
            body = soup.body
            header = ""
            for div in body.find_all("div"):
                header = div.header.get_text()
                if int(div.get("depth")) > 1: # update header
                    continue
                
                for parag in div.find_all('p'):
                    parag_dict = {}
                    parag_dict['header'] = header
                    parag_dict['sentences'] = []

                    combined_sent = {"text": "", "label": "", "header": header, "fileno": doc['fileno']}
                    for i, line in enumerate(parag.find_all('s')):
                        line = clean_bs_text(line)
                        label = line.get('az')
                        if not label:
                            no_label += 1
                            continue

                        s_id = line.get("id")

                        if line.get("type")=="ITEM": # sentence is likely split into many items
                            if combined_sent["label"]: # we've already collected some previous sent items already
                                if combined_sent["label"] != label:
                                    combined_sent = split_sents_get_labels(combined_sent)
                                    parag_dict['sentences'].extend(combined_sent) # save previous aggregated sents
                                    combined_sent = {"text": line.get_text(), "label": label, "s_id": s_id, "header": header, "fileno": doc["fileno"], }
                                    continue
                                else:
                                    pass # else sents with the same label are lumped as one big sentence. 
                                    # this is dealt with in the next elif.
                            else: # else we encounter the first item of a split-up sentence
                                combined_sent["label"] = label
                            
                            combined_sent["text"] = "".join([combined_sent["text"], line.get_text()])
                            combined_sent["s_id"] = s_id

                            if (i==(len(parag.find_all('s'))-1)) and combined_sent["text"]: # the final item is the final line in the parag
                                combined_sent = split_sents_get_labels(combined_sent)
                                parag_dict['sentences'].extend(combined_sent)
                                combined_sent = {"text": "", "label": "", "header": header, "fileno": doc["fileno"]} # reset
                            continue

                        elif (line.get("type")!="ITEM") and combined_sent["text"]: # prev lines are combined items
                            # => deal w them before dealing w current line
                            combined_sent = split_sents_get_labels(combined_sent)
                            parag_dict['sentences'].extend(combined_sent)
                            combined_sent = {"text": "", "label": "", "header": header, "fileno": doc['fileno'], "s_id": ""} # reset

                        # deal with current line
                        label = line.get('az')
                        if not label: 
                            no_label += 1
                            continue
                        sent_dict = {'text': line.get_text(), 'label': label, "header": header, "s_id": s_id, "fileno": doc['fileno']}
                        parag_dict['sentences'].append(sent_dict)
                    
                    if parag_dict['sentences']:
                        for sent in parag_dict['sentences']:
                            assert sent['header']
                        if parag.find_all('image') is not None:
                            parag_dict['sentences'][-1]['text'] = parag_dict['sentences'][-1]['text'] + "IMG"
                        doc['body'].append(parag_dict)
            docs.append(doc)
    print("# of no label sentences: ", no_label)



    with jsonlines.open(os.path.join(out_dir, "az_papers.jsonl"), "w") as f:
        f.write_all(docs)
    print(f"Saved to {out_dir}/az_papers.jsonl")

    save_to_pubmed_format(out_dir)


if __name__=="__main__":
    nlp = English()
    nlp.add_pipe("sentencizer")
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    main(in_dir, out_dir)