from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
import os
from tqdm import tqdm
import argparse
import torch
from glob import glob
import logging

from utils import read_path_list

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(filename='logs/make_cropped_dataset.log')

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='./data/train', type=str)
parser.add_argument('--batch', '-b', default=150, type=int)
parser.add_argument('--size', '-s', default=224, type=int)
parser.add_argument('--startid', default=0, type=int)
parser.add_argument('--diff', action='store_true')

args = parser.parse_args()
inp_dir = args.input
batch_size = args.batch
image_size = args.size
s_id = args.startid
diff = args.diff

mode = os.path.basename(inp_dir)
img_paths = read_path_list(
    fname=f'{mode}_list.txt', inp_dir=f'./data/{mode}', s_id=s_id)

if diff:
    print('process only diff file')
    processed_img_paths = glob(os.path.join(
        inp_dir.replace('data', 'processed_data'), '*/*'))
    processed_img_paths
    img_paths = list(set(
        img_paths) - set([x.replace('processed_data', 'data') for x in processed_img_paths]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'start from id: n{str(s_id).zfill(6)}')
print('input_dir :', inp_dir)
print('batch_size :', batch_size)
print('image_size :', image_size)
print('images :', len(img_paths))
print('device :', device)

model = MTCNN(image_size=image_size, select_largest=True,
              post_process=False, device=device)

for i in tqdm(range(0, len(img_paths), batch_size)):
    if i + batch_size <= len(img_paths):
        batch_paths = img_paths[i:i + batch_size]
    else:
        batch_paths = img_paths[i:]
    try:
        batch_img = [Image.open(path).resize((image_size, image_size))
                     for path in batch_paths]
        save_paths = [path.replace('data', 'processed_data')
                      for path in batch_paths]
        model(batch_img, save_path=save_paths)
    except Exception:
        # バッチ処理したいが、顔検出できなかったときの個別対応ができない
        # →　例外発生したら個別に処理
        for path in batch_paths:
            try:
                img = Image.open(path).resize((image_size, image_size))
                save_path = path.replace('data', 'processed_data')
                model(img, save_path)
            except TypeError:
                logging.warning(f'face not found : {path}')
            except OSError:
                logging.error(f'Input Error : {path}')
