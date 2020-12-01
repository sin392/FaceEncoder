# flake8: noqa

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.preprocessing import LabelEncoder
from glob import glob
from PIL import Image, ImageFile
from tqdm import tqdm
import time

from model import FaceEncoder
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True  # デフォルトでは無視される画像もロード
torch.cuda.empty_cache()  # メモリーのクリア


def extract_lab(path):
    # data/test/{label}/image.jpg
    return os.path.basename(os.path.dirname(path))


img_paths = sorted(glob('data/test/*/*.jpg'))  # noqa

img_paths = img_paths[:2500]

raw_labels = [extract_lab(x) for x in img_paths]
labels = LabelEncoder().fit_transform(raw_labels)

print("Number of images :", len(set(img_paths)))
print("Number of unique labels :", len(set(labels)))
print("-"*30)


class FaceDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        lab = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return (img, lab)


"""     setup data         """
# TODO: リサイズのサイズ策定
# Normalizeは？
transforms = Compose([
    Resize((224, 224)),  # for vgg16
    ToTensor(),
])
batch_size = 50

dataset = FaceDataset(img_paths, labels, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
                nn.Softmax(dim=1)
            )
        )

    def forward(self, x):
        # x = self.convs(x)
        # x = self.avgpool(x)
        # x = self.denses(x)
        # x = self.last(x)
        x = self.model(x)
        return x


"""     setup model         """
# n_classes = 1000
n_classes = len(set(labels))
lr = 0.001
# NOTE: 特徴抽出層は完全に凍結してるが、学習する内容的に学習し直した方がいい
#       人物分類で事前学習したほうがよいかもしれない
model = my_vgg16_bn(out_features=n_classes)
# HACK: featuresの上にmodelという階層ができてしまっているので、モデルクラス内での学習済みモデルの利用を改良したい
#       CNNのrequires_gradをTrueにするとメモリーが枯渇する
for param in model.model.features.parameters():  # CNN
    param.requires_grad = False
# for param in model.model.features[-10:].parameters():  # CNN
#     param.requires_grad = True
for param in model.model.classifier.parameters():  # FC
    param.requires_grad = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device :", device)
model = model.to(device).train()

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
print(optimizer)
print(criterion)
print("-"*30)


"""     train loop          """
# img = iter(loader).next()[0]
# out = model(img)
# print(out.size())
# print(out)

for epoch in range(10):
    running_loss = 0.0
    total = 0.0
    correct = 0.0
    for imgs, true_labs in tqdm(loader):
        # .to(deveice)が再代入じゃないと機能しないときあり
        imgs, true_labs = imgs.to(device), true_labs.to(device)

        with torch.set_grad_enabled(True):  # これがないときまったく学習すすまなかった、デフォだと無効？
            optimizer.zero_grad()
            outputs = model(imgs)
            pred_labs = outputs.argmax(dim=1)
            loss = criterion(outputs, true_labs)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        total += true_labs.size(0)
        correct += (pred_labs == true_labs).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))
    print(f"epoch:{epoch+1} , loss:{running_loss}")

torch.save(model.state_dict(), './weight/model.pth')
