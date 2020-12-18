from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor


class FaceDataset(Dataset):
    def __init__(self, img_paths, labels, emb_dict=None, transform=None):
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
        emb = self.emb_dict[self.raw_labels[idx]] if self.emb_dict else None

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return (img, lab, emb)
