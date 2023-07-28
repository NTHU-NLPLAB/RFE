import pandas as pd
import sys

HUMANS = [1, 3, 4]


def main():
    # PART 1
    agg_df = pd.DataFrame()
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
            agg_df['tag_type'] = tag_types
            agg_df['sentence'] = df['sentence']

        agg_df[f"class_correctness_{human}"] = df['Class correctness']
        agg_df[f"span_correctness_{human}"] = df['Span correctness']

    agg_df.to_csv("agg_human_evals_p1.csv")

    # PART 2
    agg_df = pd.DataFrame()
    for human in HUMANS:
        FILENAME=f"human_eval_p2_{human}.csv"
        df = pd.read_csv(FILENAME)
        agg_df[f'class_correctness_{human}'] = df['Class correctness']
    agg_df.to_csv("agg_human_evals_p2.csv")


            




if __name__=="__main__":
    main()
