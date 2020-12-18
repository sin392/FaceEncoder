# flake8: noqa

from tqdm import tqdm
from PIL import ImageFile
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import argparse

from model import my_vgg16_bn
from dataset import FaceDataset
from utils import extract_lab, read_path_list, extract_limited_class, save_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True  # デフォルトでは無視される画像もロード
torch.cuda.empty_cache()  # メモリーのクリア

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', default=50, type=int)
parser.add_argument('--loss_weight', default=0.0, type=float)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--limit', default=0, type=int)
parser.add_argument('--epoch', '-e', default=60, type=int)
parser.add_argument('--weight', '-w', default=None, type=str)

args = parser.parse_args()
batch_size = args.batch_size
lr = args.lr
loss_weight = args.loss_weight  # for additional MSELoss
limit = args.limit
epoch = args.epoch
weight = args.weight

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('batch_size :', batch_size)
print('lr :', lr)
print('loss_weight :', loss_weight)
print('limit :', limit)
print('weight :', weight)
print("device :", device)
print("-" * 30)

# NOTE : 元々のtestデータセットは人物集合が異なり、softmaxでの学習における検証に適さないので、
# trainセットから各クラス１０サンプルずつサンプリングしてテストデータセットとした。(split_actual_list.py)
# TODO : 元々のtestのデータも学習に含める？
tr_img_paths = read_path_list(fname='train_list.txt', inp_dir='data/train')
tr_raw_labels = [extract_lab(x) for x in tr_img_paths]
te_img_paths = read_path_list(fname='test_list.txt', inp_dir='data/test')
te_raw_labels = [extract_lab(x) for x in te_img_paths]

# NOTE : trとteのクラス数は共通
unique_classes = sorted(list(set(tr_raw_labels)))
n_classes = len(unique_classes)

if limit:
    tr_img_paths, tr_raw_labels = extract_limited_class(tr_img_paths, tr_raw_labels, unique_classes, limit)
    te_img_paths, te_raw_labels = extract_limited_class(te_img_paths, te_raw_labels, unique_classes, limit)
    unique_classes = unique_classes[:limit]
    n_classes = limit

dummy_embedding = torch.zeros(2048)
embedding_dict = {key: dummy_embedding for key in unique_classes}

print("n_classes :", n_classes)
print("Number of train images :", len(tr_img_paths))
print("Number of test images :", len(te_img_paths))
print("-" * 30)

# exit()
"""     setup data         """
# TODO: リサイズのサイズ策定
# Normalizeは？
transforms = Compose([
    Resize((224, 224)),  # for vgg16
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

tr_dataset = FaceDataset(tr_img_paths, tr_raw_labels, embedding_dict, transform=transforms)
te_dataset = FaceDataset(te_img_paths, te_raw_labels, None, transform=transforms)
tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=True)

"""     setup model         """
# NOTE: 特徴抽出層は完全に凍結してるが、学習する内容的に学習し直した方がいい
#       人物分類で事前学習したほうがよいかもしれない
model = my_vgg16_bn(out_features=n_classes)
if weight:
    model.load_state_dict(torch.load(weight))
# HACK: featuresの上にmodelという階層ができてしまっているので、モデルクラス内での学習済みモデルの利用を改良したい
#       CNNのrequires_gradをTrueにするとメモリーが枯渇する
for param in model.model.features.parameters():  # CNN
    param.requires_grad = False
for param in model.model.features[-10:].parameters():  # CNN
    param.requires_grad = True
for param in model.model.avgpool.parameters():  # AVGPOOL
    param.requires_grad = True
for param in model.model.classifier.parameters():  # FC
    param.requires_grad = True
for param in model.last.parameters():  # FC
    param.requires_grad = True

# model.parameter()利用する前にmodelのGPU転送が先
model = model.to(device).train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
# LofSoftmaxとNULLLossの利用を検討してもいいのかも
criterion = nn.CrossEntropyLoss()
additional_criterion = nn.MSELoss()

# TODO : スケジューラーの利用
print("optimizer :", optimizer)
# print("loss :", criterion)
# print("additional_loss :", additional_criterion)
print("-" * 30)

"""     train loop          """
for epoch in range(epoch):
    running_loss = total = correct = 0.0
    for imgs, true_labs, embs in tqdm(tr_loader):
        # .to(deveice)が再代入じゃないと機能しないときあり
        imgs, true_labs, embs = imgs.to(device), true_labs.to(device), embs.to(device)

        with torch.set_grad_enabled(True):  # これがないときまったく学習すすまなかった、デフォだと無効？
            optimizer.zero_grad()
            middles, outputs = model(imgs)
            pred_labs = outputs.argmax(dim=1)
            loss = criterion(outputs, true_labs) + loss_weight * (0.5 * n_classes * additional_criterion(middles, embs))
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        total += true_labs.size(0)
        correct += (pred_labs == true_labs).sum().item()
    acc = 100 * float(correct / total)
    print(f"epoch:{epoch+1} , train_acc: {acc:.2f}, train_loss:{running_loss:.2f}")

    if (epoch + 1) % 5 == 0:
        print('-' * 30)
        running_loss = total = correct = 0.0
        for imgs, true_labs, embs in tqdm(tr_loader):
            # .to(deveice)が再代入じゃないと機能しないときあり
            imgs, true_labs, embs = imgs.to(device), true_labs.to(device), embs.to(device)

            with torch.set_grad_enabled(False):
                middles, outputs = model(imgs)
                pred_labs = outputs.argmax(dim=1)
                # NOTE : test時にはMSEは見ない
                # loss = criterion(outputs, true_labs) + loss_weight * (0.5 * tr_n_classes * additional_criterion(middles, embs))
                loss = criterion(outputs, true_labs)

            running_loss += loss.item()
            total += true_labs.size(0)
            correct += (pred_labs == true_labs).sum().item()
        acc = 100 * float(correct / total)
        print(f"epoch:{epoch+1} , test_acc: {acc:.2f} , test_loss:{running_loss:.2f}")
        print('-' * 30)

        save_weight(model, epoch=epoch + 1, acc=acc, output_dir="weight")
