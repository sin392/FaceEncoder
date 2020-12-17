# flake8: noqa

import torchvision.models as models
from model import FaceEncoder
import collections
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
from PIL import Image, ImageFile
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import argparse


ImageFile.LOAD_TRUNCATED_IMAGES = True  # デフォルトでは無視される画像もロード
torch.cuda.empty_cache()  # メモリーのクリア


def save_weight(model, epoch, output_dir="weight"):
    date = datetime.now.strftime("%Y-%m-%d-%H-%M-%S")
    fname = f'{date}-epoch{epoch}.pth'
    torch.save(model.state_dict(), os.path.join(output_dir, fname))


def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))


class FaceDataset(Dataset):
    def __init__(self, img_paths, labels, emb_dict, transform=None):
        self.img_paths = img_paths
        self.raw_labels = labels
        self.labels = LabelEncoder().fit_transform(labels)
        self.emb_dict = emb_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        lab = self.labels[idx]
        emb = self.emb_dict[self.raw_labels[idx]]

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return (img, lab, emb)


class my_vgg16_bn(nn.Module):
    def __init__(self, middle_features=2048, out_features=1000, pretrained=True):
        super().__init__()
        self.model = models.vgg16_bn(pretrained=pretrained)
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=in_features,
                          out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096,
                          out_features=middle_features, bias=True),
            )
        )
        self.last = nn.Sequential(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=middle_features,
                          out_features=out_features, bias=True),
                # nn.Softmax(dim=1) # pytochにおいてsoftmaxは不要
            )
        )

    def forward(self, x):
        middle = self.model(x)
        out = self.last(middle)
        return middle, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=50, type=int)
    parser.add_argument('--loss_weight', default=0.0, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)

    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    loss_weight = args.loss_weight # for additional MSELoss

    print('batch_size :', batch_size)
    print('lr :', lr)
    print('loss_weight :', loss_weight)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    print("-" * 30)

    inp_dir = 'data/train'
    if os.path.exists(f'train_list.txt'):
        print(f'use train_list.txt')
        with open(f"train_list.txt", mode="rt") as f:
            img_paths = [os.path.join(inp_dir, x.strip()) for x in f.readlines()]
    else:
        print('use glob')
        img_paths = glob(os.path.join(inp_dir, '*/*'))

    img_paths = img_paths[:5000]

    raw_labels = [extract_lab(x) for x in img_paths]
    unique_classes = list(set(raw_labels))
    n_classes = len(set(raw_labels))


    dummy_embedding = torch.zeros(2048)
    embedding_dict = {key:dummy_embedding for key in unique_classes}

    print("Number of images :", len(img_paths))
    print("n_classes :", n_classes)
    # print(collections.Counter(raw_labels))  # NOTE: count value is sorted
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

    dataset = FaceDataset(img_paths, raw_labels, embedding_dict, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """     setup model         """
    # NOTE: 特徴抽出層は完全に凍結してるが、学習する内容的に学習し直した方がいい
    #       人物分類で事前学習したほうがよいかもしれない
    model = my_vgg16_bn(out_features=n_classes)
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
    for epoch in range(60):
        running_loss = total = correct = 0.0

        for imgs, true_labs, embs in tqdm(loader):
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
        print('Accuracy: {:.2f} %'.format(100 * float(correct / total)))
        print(f"epoch:{epoch+1} , loss:{running_loss}")

        if (epoch + 1) % 5 == 0:
            save_weight(model, epoch=epoch + 1, output_dir="weight")
