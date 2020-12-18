import os
from glob import glob
from datetime import datetime
import torch


def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))


def read_path_list(fname, inp_dir, s_id=0):
    if os.path.exists(fname):
        print(f'use {fname}')
        with open(f'{fname}', mode='rt') as f:
            lines = f.readlines()
            s_from = 0
            # 指定したidが存在しない場合はidスキップ
            while s_id > 0 and s_from == 0:
                try:
                    s_from = [os.path.dirname(x) for x in lines].index(f'n{str(s_id).zfill(6)}')
                except:
                    s_id + 1
            img_paths = [os.path.join(inp_dir, x.strip()) for x in lines[s_from:]]
    else:
        print('use glob')
        img_paths = glob(os.path.join(inp_dir, '*/*'))

    return img_paths


def extract_limited_class(img_paths, raw_labels, unique_classes, limit_n_classes):
    e_idx = raw_labels.index(unique_classes[limit_n_classes])
    return img_paths[:e_idx], raw_labels[:e_idx]


def save_weight(model, epoch, acc, output_dir="weight"):
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fname = f'{date}-epoch{epoch}-acc{acc}.pth'
    torch.save(model.state_dict(), os.path.join(output_dir, fname))
