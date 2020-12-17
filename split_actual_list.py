import os

def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))

with open("train_list.old.txt", mode="rt") as f:
    tr_paths = [x.strip() for x in f.readlines()]

# TODO : test時の検証方法の確認（テスト時は人物集合が異なる）
te_paths = []
tr_raw_labels = [extract_lab(x) for x in tr_paths]
temp_tr_paths = tr_paths
temp_tr_raw_labels = tr_raw_labels
temp, last_temp = 0, 0
print(len(tr_paths))
for cl in sorted(list(set(tr_raw_labels))):
    print(cl)
    # 各クラスの初めのインデックス取得
    temp = tr_raw_labels[last_temp:].index(cl)
    # 各クラスの先頭１０枚はtest用に
    te_paths.extend(tr_paths[temp:temp+10])
    del temp_tr_paths[temp:temp+10]
    del temp_tr_raw_labels[temp:temp+10]
    last_temp = temp+10
# 更新
tr_paths = temp_tr_paths
tr_raw_labels = temp_tr_raw_labels
te_raw_labels = [extract_lab(x) for x in te_paths]
print(te_paths)

with open("train_list.txt", mode="wt") as f:
    for d in tr_paths:
        f.write("%s\n" % d)
with open("test_list.txt", mode="wt") as f:
    for d in te_paths:
        f.write("%s\n" % d)