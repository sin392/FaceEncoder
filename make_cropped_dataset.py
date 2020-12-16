from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
from glob import glob
import os
from tqdm import tqdm
import argparse
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='./data/test', type=str)
parser.add_argument('--batch', '-b', default=15, type=int)
parser.add_argument('--size', '-s', default=224, type=int)

args = parser.parse_args()
inp_dir = args.input
batch_size = args.batch
image_size = args.size

mode = os.path.basename(inp_dir)
if os.path.exists(f"{mode}_list.txt"):
    print(f"use {mode}_list.txt")
    with open(f"{mode}_list.txt", mode="rt") as f:
        actual_list = [x.rstrip() for x in f.readlines()]
    # WARN : ファイル欠損してる可能性
    img_paths = [os.path.join(inp_dir, x) for x in actual_list]
else:
    img_paths = glob(os.path.join(inp_dir, '*/*'))

print('input_dir :', inp_dir)
print('batch_size :', batch_size)
print('image_size :', image_size)
print('images :', len(img_paths))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device :', device)
model = MTCNN(image_size=image_size, select_largest=True,
              post_process=False, device=device)

for i in tqdm(range(0, len(img_paths), batch_size)):
    if i+batch_size <= len(img_paths):
        batch_paths = img_paths[i:i+batch_size]
    else:
        batch_paths = img_paths[i:i+(len(batch_size - batch_size))]
    batch_img = [Image.open(path).resize((image_size, image_size))
                 for path in batch_paths if os.path.exists(path)]
    save_paths = [path.replace('data', 'processed_data')
                  for path in batch_paths if os.path.exists(path)]
    try:
        model(batch_img, save_path=save_paths)
    except Exception:
        # バッチ処理したいが、顔検出できなかったときの個別対応ができない
        # →　例外発生したら個別に処理
        for j in range(batch_size):
            try:
                model(batch_img[j], save_path=save_paths[j])
            except Exception:
                print(f'face not found : {batch_paths[j]}')
