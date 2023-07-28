import pandas as pd
import numpy as np
import sys

from collections import defaultdict

HUMANS = [1, 3, 4]

def f1(prec, rec):
    return 2*(prec * rec) / (prec + rec)

def main():
    # PART 1
    agg_df_p1 = pd.DataFrame()
    for human in HUMANS:
        FILENAME=f"human_eval_p1_{human}.csv"
        df = pd.read_csv(FILENAME)

        # get tag types, append to aggregated df
        tag_types = []
        for idx, example in df.iterrows():
            sent = example['sentence']
            tag_type = sent.split("]]\\")[1]
            tag_type = tag_type.split(" ")[0][:3]
            tag_types.append(tag_type)
        if human==1: # we only need to do this once
            agg_df_p1['pred_tag_type'] = tag_types
            agg_df_p1['sentence'] = df['sentence']

        agg_df_p1[f"class_correctness_{human}"] = df['Class correctness']
    
    cols = [f"class_correctness_{str(human)}" for human in HUMANS]
    majority = agg_df_p1[cols].mode(axis=1)
    majority = np.where(majority.isna().any(axis=1), majority[0], 'split') # majority[0] the mode
    
    majority_tags = []
    for i, ex in enumerate(majority):
        if ex == "class is CORRECT":
            majority_tags.append(agg_df_p1['pred_tag_type'][i])
        else:
            majority_tags.append("None")
    agg_df_p1['majority_gold'] = majority_tags

    # PART 2
    agg_df_p2 = pd.DataFrame()
    for human in HUMANS:
        FILENAME=f"human_eval_p2_{human}.csv"
        df = pd.read_csv(FILENAME)
        agg_df_p2[f'class_correctness_{human}'] = df['Class correctness']
        agg_df_p2['sentence'] = df['sentence']
    agg_df_p2['pred_tag_type'] = ["None" for it in range(len(agg_df_p2))]

    majority = agg_df_p2[cols].mode(axis=1)
    majority = np.where(majority.isna().any(axis=1), majority[0], 'split') # majority[0] the mode
    majority_tags = []
    for ex in majority:
        if "SHOULD contain RFE" in ex:
            majority_tags.append(ex.split(": ")[1])
        else:
            majority_tags.append("None")
    agg_df_p2['majority_gold'] = majority_tags


    # Combine the two dfs
    all_sents = pd.concat([agg_df_p1['sentence'], agg_df_p2['sentence']], axis=0, ignore_index=True)
    all_pred_tags = pd.concat([agg_df_p1['pred_tag_type'], agg_df_p2['pred_tag_type']], axis=0, ignore_index=True)
    all_maj_tags = pd.concat([agg_df_p1['majority_gold'], agg_df_p2['majority_gold']], axis=0, ignore_index=True)

    agg_p1_p2 = pd.DataFrame()
    agg_p1_p2['sentence'] = all_sents
    agg_p1_p2['pred_tag_type'] = all_pred_tags
    agg_p1_p2['majority_gold'] = all_maj_tags


    ALL_TAGS = agg_df_p1['pred_tag_type'].unique()

    res = defaultdict(dict)
    class_weights = defaultdict(dict)
    for tag in ALL_TAGS:
        pred_tag_df = agg_p1_p2[agg_p1_p2['pred_tag_type']==tag]
        tp = len(pred_tag_df[pred_tag_df['majority_gold']==tag])
        prec = tp/len(pred_tag_df)

        gold_tag_df = agg_p1_p2[agg_p1_p2['majority_gold']==tag]
        tp = len(gold_tag_df[gold_tag_df['pred_tag_type']==tag])
        rec = tp/len(gold_tag_df)

        res[tag]['precision'] = prec
        res[tag]['recall'] = rec
        res[tag]['f1'] = f1(prec, rec)

        class_weights[tag] = len(gold_tag_df) / len(agg_p1_p2[agg_p1_p2['majority_gold']!='None'])

    total_tp = len(agg_p1_p2[agg_p1_p2['pred_tag_type']==agg_p1_p2['majority_gold']])
    acc = total_tp / len(agg_p1_p2)
    weighted_f1 = 0
    for tag in ALL_TAGS:
        weighted_f1 += res[tag]['f1'] * class_weights[tag]
    
    res['overall']['f1'] = weighted_f1

    for k in res:
        print(f"{k}: {res[k]}")


if __name__=="__main__":
    main()
