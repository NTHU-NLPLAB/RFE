import pandas as pd
import numpy as np
import krippendorff as kd 
from statsmodels.stats import inter_rater as irr
from sklearn.metrics import cohen_kappa_score as cohen

HUMANS = [1, 3, 4]
CLASS_OPTIONS_1=['class is CORRECT', 'class is NOT CORRECT', 'Should NOT contain RFE']
CLASS_OPTIONS_2=['No RFE needed', 'SHOULD contain RFE: AIM', 'SHOULD contain RFE: BAS', 
                 "SHOULD contain RFE: BKG", "SHOULD contain RFE: CTR", "SHOULD contain RFE: OTH",
                 "SHOULD contain RFE: OWN", "SHOULD contain RFE: TXT"]
SPAN_OPTIONS = ["Too short", "Too long", "Just right", "Should not contain RFE", "Span should be elsewhere"]

CLASS_OPTION_1_IDX = {
    'class is CORRECT': 0, 
    'class is NOT CORRECT': 1,
    'Should NOT contain RFE': 2
}

CLASS_OPTIONS_2_IDX={
    'No RFE needed': 0, 
    'SHOULD contain RFE: AIM': 1,
    'SHOULD contain RFE: BAS': 2,
    "SHOULD contain RFE: BKG": 3,
    "SHOULD contain RFE: CTR": 4,
    "SHOULD contain RFE: OTH": 5,
    "SHOULD contain RFE: OWN": 6,
    "SHOULD contain RFE: TXT": 7
}

SPAN_OPTIONS_IDX = {
    "Too short": 0,
    "Too long": 1,
    "Just right": 2,
    "Should not contain RFE": 3, 
    "Span should be elsewhere": 4
}

class TagStats():
    def __init__(self, tag_type, data_df):
        self.data_df = data_df
        self.tag_type = tag_type
        self.num_examples = 0
        self.randolf = 0
        self.all_agree_cnt = 0
        self.all_agree_pcnt = 0
        self.majority_agree_cnt = 0
        self.majority_agree_pcnt = 0

        self.precision = 0
        self.recall = 0

    

 
def get_irr_stats(ans_arr, method):
    ans_arr_T = ans_arr.transpose()
    agg = irr.aggregate_raters(ans_arr_T)
    return irr.fleiss_kappa(agg[0], method=method)


def get_all_agree_sub_df(col_name_prefix, df):
    df = df[df[col_name_prefix+'1']==df[col_name_prefix+'3']]
    df = df[df[col_name_prefix+'1']==df[col_name_prefix+'4']]
    df = df[df[col_name_prefix+'4']==df[col_name_prefix+'3']]
    return df

def get_maj_agree_sub_df(col_name_prefix, df):
    # https://stackoverflow.com/questions/55749114/how-to-do-a-majority-voting-on-columns-in-pandas
    cols = [col_name_prefix+str(human) for human in HUMANS]
    majority = df[cols].mode(axis=1) # vote for each row
    # return val of df.mode is a df where row is the mode of each col
    if majority.shape[1] > 1: # has split values => collapse df into N x 1
        majority = np.where(majority.isna().any(axis=1), majority[0], 'split') # majority[0] is the name of the mode
        # a row contains NaN means it has a mojority vote => keep its value; split rows have no NaN => give it 'split'
        return pd.DataFrame(majority)
    elif majority.shape[1]==1:
        return majority

def part_1_class():
    df = pd.read_csv("agg_human_evals_p1.csv")
    print(df.columns.values)
    ALL_TAGS = list(set(df['tag_type']))
    for tag in ALL_TAGS:
        print("#"*30 + " " + tag + " " + "#"*30)
        tag_df = df[df['tag_type']==tag]
        all_agree = get_all_agree_sub_df('class_correctness_', tag_df)
        print(f"All agree percentage: {len(all_agree)/len(tag_df):.2%}")
        maj_agree = get_maj_agree_sub_df('class_correctness_', tag_df)
        non_split = maj_agree[maj_agree[0]!="split"] # col index 0, because maj_agree is a datagrame w a single (nameless) column
        # TODO: fix majority agree percentage is 0
        print(f"Majority agree percentage: {len(non_split)/len(tag_df):.2%}")

        for option in CLASS_OPTIONS_1:
            num_agree = len(all_agree[all_agree['class_correctness_1']==option])
            print(f"{option}, among all_agree: num_agree ({num_agree/len(tag_df):.2%} among all {tag})")

            num_maj_agree = len(maj_agree[maj_agree[0]==option])
            print(f"{option}, majority agree: {num_maj_agree} ({num_maj_agree/len(tag_df):.2%} among all {tag})") # len(maj_agree) should == len(tag_df)
            
        # Agreement scores
        ans_arr = []
        for human in HUMANS:
            ans_arr.append([CLASS_OPTION_1_IDX[it] for it in tag_df[f'class_correctness_{human}']])
        ans_arr = np.asarray(ans_arr)
        
        #print(f"aggregate  d array for fleiss alpha: {agg[0].shape}, categories: {agg[1]}")
        print(f"randolf's kappa among all raters: {get_irr_stats(ans_arr, 'randolph')}")
        print(f"krippendorff alpha among all raters: {kd.alpha(ans_arr, level_of_measurement='nominal')}")
        #print(f"Majority agree percentage: {}")

        print()

    ans_arr = []
    for human in HUMANS:
        ans_arr.append([CLASS_OPTION_1_IDX[it] for it in df[f'class_correctness_{human}']])
    ans_arr = np.asarray(ans_arr)
    print("#"*30 + " OVERALL " + "#"*30)
    print(f"randolf's kappa among all raters: {get_irr_stats(ans_arr, 'randolph')}")
    print(f"krippendorff alpha among all raters: {kd.alpha(ans_arr, level_of_measurement='nominal')}")

def part_1_span():
    df = pd.read_csv("agg_human_evals_p1.csv")
    print(df.columns.values)
    ALL_TAGS = list(set(df['tag_type']))
    for tag in ALL_TAGS:
        print("#"*30 + " " + tag + " " + "#"*30)
        tag_df = df[df['tag_type']==tag]
        all_agree = get_all_agree_sub_df('span_correctness_', tag_df)
        print(f"All agree percentage: {len(all_agree)/len(tag_df):.2%}")

        maj_agree = get_maj_agree_sub_df('span_correctness_', tag_df)
        non_split = maj_agree[maj_agree[0]!="split"]
        print(f"Majority agree percentage: {len(non_split)/len(tag_df):.2%}")

        for option in SPAN_OPTIONS:
            num_agree = len(all_agree[all_agree['span_correctness_1']==option])
            print(f"{option}, among all_agree: num_agree ({num_agree/len(tag_df):.2%} among all {tag})")

            num_maj_agree = len(maj_agree[maj_agree[0]==option])
            print(f"{option}, majority agree: {num_maj_agree} ({num_maj_agree/len(tag_df):.2%} among all {tag})") # len(maj_agree) should == len(tag_df)
            # above is precision (correct / predicted)
            
        
        # Agreement score
        ans_arr = []
        for human in HUMANS:
            ans_arr.append([SPAN_OPTIONS_IDX[it] for it in tag_df[f'span_correctness_{human}']])
        ans_arr = np.asarray(ans_arr)
        #print(f"aggregated array for fleiss alpha: {agg[0].shape}, categories: {agg[1]}")
        print(f"randolf's kappa among all raters: {get_irr_stats(ans_arr, 'randolph')}")
        print(f"krippendorff alpha among all raters: {kd.alpha(ans_arr, level_of_measurement='nominal')}")
        #print(f"Majority agree percentage: {}")

        print()

    ans_arr = []
    for human in HUMANS:
        ans_arr.append([SPAN_OPTIONS_IDX[it] for it in df[f'span_correctness_{human}']])
    ans_arr = np.asarray(ans_arr)
    print("#"*30 + " OVERALL " + "#"*30)
    print(f"randolf's kappa among all raters: {get_irr_stats(ans_arr, 'randolph')}")
    print(f"krippendorff alpha among all raters: {kd.alpha(ans_arr, level_of_measurement='nominal')}")


def part_2():
    df = pd.read_csv("agg_human_evals_p2.csv")
    print(df.columns.values)
    all_agree = get_all_agree_sub_df('class_correctness_', df)
    maj_agree = get_maj_agree_sub_df('class_correctness_', df)
    non_split = maj_agree[maj_agree[0]!='split']
    
    print(f"All agree percentage: {len(all_agree)/len(df):.2%}")
    print(f"Majority agree percentage: {len(non_split)/len(df):.2%}")

    for option in CLASS_OPTIONS_2:
        num_agree = len(all_agree[all_agree['class_correctness_1']==option])
        print(f"{option} all agree: {num_agree} ({num_agree/len(df):.2%})")

        num_maj_agree = len(maj_agree[maj_agree[0]==option])
        print(f"{option} majority agree: {num_maj_agree} ({num_maj_agree/len(df):.2%})\n")

        # TODO: calculate F1s here

    # Agreement score
    ans_arr = []
    for human in HUMANS:
        ans_arr.append([CLASS_OPTIONS_2_IDX[it] for it in df[f'class_correctness_{human}']])
    ans_arr = np.asarray(ans_arr)
    #print(f"aggregated array for fleiss alpha: {agg[0].shape}, categories: {agg[1]}")
    print(f"randolf's kappa among all raters: {get_irr_stats(ans_arr, 'randolph')}")
    print(f"krippendorff alpha among all raters: {kd.alpha(ans_arr, level_of_measurement='nominal')}")
        

    

def main():
    #part_1_class()
    part_1_span()

    #part_2()


if __name__=="__main__":
    main()
