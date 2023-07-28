# merged cites in joanna's dependency-parsed s2orc.0.txt
# example from original :
#    [       [       PUNCT   -LRB-   22
#    4       4       X       LS      7       21 23 25 44 48 50 53
#    ]       ]       PUNCT   -RRB-   22
# Desired output:
#   CREF   CREF     N       N       -1      -1 
from tqdm import tqdm

DEBUG=True

if DEBUG:
    in_file = "merge_cite_test.txt"
    total_lines = 281
else:
    in_file = "s2orc.0.dependency.txt"
    total_lines = 593725306

if __name__=="__main__":
    with tqdm(total=total_lines+1) as pbar,\
        open(in_file) as in_f,\
        open("s2orc.0.dependency.mcite.txt", "a") as out_f:
        in_line = in_f.readline()
        prev_is_cite = False
        while in_line:
            if in_line[0] == "[":
                prev_is_cite = True
                in_line = in_f.readline()
                pbar.update(1) 

            if prev_is_cite:
                if in_line[0].isdigit():
                    prev_is_cite = True
                    in_line = in_f.readline()
                    pbar.update(1)
                else:
                    prev_is_cite = False
                    in_line = in_f.readline()
                    pbar.update(1)
                if in_line[0]=="]":
                    out_f.write("\t".join(["CREF", "CREF", "N", "N", "-1", "-1"])+"\n")
                    prev_is_cite = False
                    in_line = in_f.readline()
                    pbar.update(1) 
            else:
                out_f.write(in_line)
                in_line = in_f.readline()
                pbar.update(1) 


