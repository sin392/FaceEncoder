# %%
from IPython.display import display
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
from glob import glob
import os
from tqdm import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='./data/test', type=str)
parser.add_argument('--batch', '-b', default=15, type=int)
parser.add_argument('--size', '-s', default=224, type=int)

args = parser.parse_args()

inp_dir = args.input
batch_size = args.batch
image_size = args.size
img_paths = glob(os.path.join(inp_dir, '*/*'))
print('input_dir :', inp_dir)
print('batch_size :', batch_size)
print('image_size :', image_size)
print('images :', len(img_paths))

model = MTCNN(image_size=image_size, select_largest=True, post_process=False)

# %%
for i in tqdm(range(0, len(img_paths), batch_size)):
    batch_paths = img_paths[i:i+batch_size]
    batch_img = [Image.open(path).resize((image_size, image_size))
                 for path in batch_paths]
    save_paths = [path.replace('data', 'processed_data')
                  for path in batch_paths]
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
# %%
