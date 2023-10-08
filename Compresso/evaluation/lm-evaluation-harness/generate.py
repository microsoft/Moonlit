import sys
import json

def print_all(path):
    all_score = 0
    with open(path, "r") as fp:
        curr = json.load(fp)
    
    for task in ["piqa", "storycloze_2018", "arc_challenge", "arc_easy", "winogrande", "boolq", "openbookqa", "hellaswag"]:
        if task not in curr['results']:
            continue
        if "acc_norm" in curr["results"][task]:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}/{round(curr['results'][task]['acc_norm']*100, 2)}")
        else:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}")
        all_score += curr['results'][task]['acc']
    print("all: ", all_score/8.*100)

print_all(sys.argv[1])
