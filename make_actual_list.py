import pandas as pd
import os

# mode = "train"
mode = "test"

db = pd.read_csv("vox2_meta.csv")
with open(f"original_{mode}_list.txt", mode="rt") as f:
    original_list = [x.rstrip() for x in f.readlines()]

term = "dev" if mode == "train" else "test"
dev = db[db["Set"] == term]
vf2_ids = dev["VGGFace2ID"].tolist() # actually used ids

print(len(vf2_ids))

actual_list = [x for x in original_list if os.path.dirname(x) in vf2_ids]
print(len(original_list))
print(len(actual_list))

with open(f"{mode}_list.txt", mode="wt") as f:
    for d in actual_list:
        f.write("%s\n" % d)