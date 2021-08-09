from torch import nn
import torch

from torchvision import transforms as tf
import torchvision

from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import os

from models.senet import senet50


device = "cuda" if torch.cuda.is_available() else "cpu"


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms, remove_idx, mode="train"):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "img_align_celeba")

        with open(os.path.join(data_dir, "list_attr_celeba.txt"), "r") as f:
            attrs = f.read().split("\n")[2:-1]

        if mode == "train":
            self.attrs = attrs[:162770]
        elif mode == "val":
            self.attrs = attrs[162770:182637]
        elif mode == "test":
            self.attrs = attrs[182637:]

        self.transforms = transforms
        self.remove_idx = remove_idx

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, index):
        attrs = self.attrs[index].split()

        img = Image.open(os.path.join(self.img_dir, attrs[0]))

        t = []
        for i in range(len(attrs)-1):
            if not i+1 in self.remove_idx:
                t.append(float(attrs[i+1]))

        img = self.transforms(img)
        t = torch.tensor(t)

        return img, t


def train(args, remove_idx):
    model = senet50(num_classes=8631)
    model.fc = nn.Linear(model.fc.in_features, 40-len(remove_idx))

    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(device)


    loss = nn.MSELoss()
    loss.to(device)


    test_transforms = tf.Compose([tf.CenterCrop(178), 
                                tf.Resize(224),
                                tf.ToTensor(),
                                tf.Normalize(0.5, 0.5)
                            ])
    test_dataset = CelebADataset(args.data_dir, test_transforms, remove_idx, mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    print(f"test data size : {len(test_dataset)}")
    
    model.eval()
    test_loss = 0

    for x, t in tqdm(test_dataloader):
        x = x.to(device)
        t = t.to(device)
        y = model(x)
        l = loss(y, t)

        test_loss += l.to("cpu").item()

    print("test loss : ", test_loss / len(test_dataloader))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--data_dir", type=str, default="data",
                                            help="dataset directory")
    parse.add_argument("--batch", type=int, default=64,
                                            help="batch size")
    parse.add_argument("--ckpt", type=str, default="checkpoints/classifier_009.pt",
                                            help="pretrained weights")

    args = parse.parse_args()

    train(args, [2,7,8,9,10,12,14,18,22,24,26,32])