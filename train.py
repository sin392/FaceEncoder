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


def save_weight(model, epoch, output_dir="weight", best=True):
    date = datetime.now.strftime("%Y-%m-%d-%H-%M-%S")
    if best:
        fname = f'{date}-epoch{epoch}.pth'
    else:
        fname = f'{date}-epoch{epoch}-best.pth'
    torch.save(model.state_dict(), os.path.join(output_dir, fname))


def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))


class FaceDataset(Dataset):
    def __init__(self, img_paths, labels, embedding_dict, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.embedding_dict = embedding_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        lab = self.labels[idx]
        emb = self.embedding_dict[lab]

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
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=middle_features,
                          out_features=out_features, bias=True),
                # nn.Softmax(dim=1) # pytochにおいてsoftmaxは不要
            )
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/train", type=str)
    parser.add_argument("--batch_size", "-b", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--loss_weight", default=0.0, type=float)

    args = parser.parse_args()
    inp_dir = args.input
    batch_size = args.batch_size
    lr = args.lr
    loss_weight = args.loss_weight

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device :", device)
    print("-" * 30)

    mode = os.path.basename(inp_dir)
    if os.path.exists(f"{mode}_list.txt"):
        print(f"use {mode}_list.txt")
        with open(f"{mode}_list.txt", mode="rt") as f:
            actual_list = [x.rstrip() for x in f.readlines()]
        # WARN : ファイル欠損してる可能性
        img_paths = sorted([os.path.join(inp_dir, x) for x in actual_list])
    else:
        img_paths = sorted(glob(os.path.join(inp_dir, '*/*')))

    # NOTE : comment out following state when actual training
    img_paths = img_paths[:5000]


    raw_labels = [extract_lab(x) for x in img_paths]
    n_classes = len(set(raw_labels))

    # labels = OneHotEncoder().fit_transform(np.array(raw_labels).reshape(-1, 1))
    labels = LabelEncoder().fit_transform(raw_labels)

    print("Number of images :", len(set(img_paths)))
    print("Number of unique labels :", len(set(raw_labels)))
    print(collections.Counter(raw_labels))  # NOTE: count value is sorted
    print("-" * 30)

    # dummy embedding dict
    dummy_embeddings = torch.zeros(n_classes, 2048)
    print(dummy_embeddings)
    # TODO: embeddingsとraw_labelsのクラスの対応があってるか要確認
    # どこかでソートはいってるかもしれない
    embedding_dict = {list(set(raw_labels))[i]: dummy_embeddings[i]
                      for i in range(n_classes)}

    """     setup data         """
    # TODO: リサイズのサイズ策定
    # Normalizeは？
    transforms = Compose([
        Resize((224, 224)),  # for vgg16
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = FaceDataset(img_paths, labels, embedding_dict, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    """     setup model         """
    # n_classes = 1000
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

    # model.parameter()利用する前にmodelのGPU転送が先
    model = model.to(device).train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # LofSoftmaxとNULLLossの利用を検討してもいいのかも
    criterion = nn.CrossEntropyLoss()
    additional_criterion = nn.MSELoss()
    print("optimizer :", optimizer)
    print("loss :", criterion)
    print("additional_loss :", additional_criterion)
    print("additional_loss_weight :", loss_weight)
    print("-" * 30)


    """     train loop          """
    for epoch in range(60):
        running_loss = total = correct = 0.0

        for imgs, true_labs, embs in tqdm(loader):
            # .to(deveice)が再代入じゃないと機能しないときあり
            imgs, true_labs, embs = imgs.to(device), true_labs.to(device), embs.to(device)

            with torch.set_grad_enabled(True):  # これがないときまったく学習すすまなかった、デフォだと無効？
                optimizer.zero_grad()
                outputs = model(imgs)
                pred_labs = outputs.argmax(dim=1)
                loss = criterion(outputs, true_labs) + loss_weight * \
                    (0.5 * n_classes * additional_criterion(outputs, embs))
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total += true_labs.size(0)
            correct += (pred_labs == true_labs).sum().item()
        acc = 100 * float(correct / total)
        print('Accuracy: {:.2f} %'.format(acc))
        print(f"epoch:{epoch+1} , loss:{running_loss}")

        if (epoch + 1) % 5 == 0:
            save_weight(model, epoch=epoch + 1, output_dir="weight", loss)
