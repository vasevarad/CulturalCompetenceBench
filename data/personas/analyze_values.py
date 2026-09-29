
from glob import glob
import json
import pandas as pd
import sys

cultures = ["afghan", "chinese", "nepali","burmese","maori", "vietnamese", "northeast", "midwest"]

base_dir = sys.argv[1] if len(sys.argv) > 1 else "data/personas"

culture_count = {}
for culture in cultures:
    persona_file = f"{base_dir}/{culture}/persona_*.json"
    persona_files = glob(persona_file)
    val_done = False
    norm_done = False
    for p in persona_files:
        if p.endswith("values.json") and not val_done:
            with open(p, "r") as f:
                num_values = len(f.readlines()) - 2
            print(f"{culture} persona {p.split('/')[-1].split('.')[0]} has {num_values} values")
            val_done = True
        elif not p.endswith(".txt") and not norm_done:
            with open(p, "r") as f:
                norms = json.load(f)
            num_norms = len(norms)
            print(f"{culture} persona {p.split('/')[-1].split('.')[0]} has {num_norms} norms")
            norm_done = True

        if  val_done and norm_done:
            break

    
    count_persona = {}
    for p in persona_files:
        if p.endswith("values.json"):
            pass
        elif not p.endswith(".txt"):
            with open(p, "r") as f:
                norms = json.load(f)
            for norm in norms:
                adherence = norm["adherence"]
                if adherence not in count_persona:
                    count_persona[adherence] = 0
                count_persona[adherence] += 1

    for key in count_persona:

        culture_count[(culture, key)] = count_persona[key]/10

for cul in cultures:
    print(f"{cul}: Follow - {culture_count[(cul, 'Follow')]}, Avoid - {culture_count[(cul, 'Avoid')]}, Neutral - {culture_count[(cul, 'Neutral')]}, total norms - {culture_count[(cul, 'Follow')] + culture_count[(cul, 'Avoid')] + culture_count[(cul, 'Neutral')]}")