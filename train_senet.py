from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch import optim
from torch import nn
import torch

from torchvision import transforms as tf
import torchvision

from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import pickle
import os

from models.senet import senet50


device = "cuda" if torch.cuda.is_available() else "cpu"


def backward(loss, opt, use_amp):
    opt.zero_grad()

    if use_amp:
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backwrad()

    opt.step()


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
    writer = SummaryWriter("logs/sgd1e-4")


    model = senet50(num_classes=8631)
    
    with open(args.resume, "rb") as f:
        ckpt = pickle.load(f, encoding="latin1")
    
    weights = model.state_dict()
    for key, weight in ckpt.items():
        weights[key].copy_(torch.from_numpy(weight))

    model.conv1.requres_grad = False
    model.bn1.requres_grad = False
    model.layer1.requres_grad = False
    model.layer2.requres_grad = False
    model.layer3.requres_grad = False
    model.fc = nn.Linear(model.fc.in_features, 40-len(remove_idx))
    
    model.to(device)


    loss = nn.MSELoss()
    loss.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)

    if args.use_amp:
        model, opt = amp.initialize(model, opt, opt_level="O1")


    train_transforms = tf.Compose([tf.CenterCrop(178), 
                                    tf.Resize(224), 
                                    tf.RandomHorizontalFlip(0.5),
                                    tf.ToTensor(),
                                    tf.Normalize(0.5, 0.5)
                                ])
    train_dataset = CelebADataset(args.data_dir, train_transforms, remove_idx, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    val_transforms = tf.Compose([tf.CenterCrop(178), 
                                tf.Resize(224),
                                tf.ToTensor(),
                                tf.Normalize(0.5, 0.5)
                            ])
    val_dataset = CelebADataset(args.data_dir, val_transforms, remove_idx, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    print(f"train data size : {len(train_dataset)}")
    print(f"val data size : {len(val_dataset)}")
            

    iter_ = 0
    total_bar = tqdm(total=args.total_epoch * len(train_dataloader))

    for epoch in range(args.total_epoch):
        model.train()

        for x, t in train_dataloader:
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            l = loss(y, t)

            backward(l, opt, args.use_amp)

            writer.add_scalar("train/loss", l.to("cpu").item(), iter_)
            total_bar.update(1)
            iter_ += 1


        model.eval()
        val_loss = 0

        for x, t in val_dataloader:
            x = x.to(device)
            t = t.to(device)
            y = model(x)
            l = loss(y, t)

            val_loss += l.to("cpu").item()

        writer.add_scalar("val/loss", val_loss / len(val_dataloader), epoch)


        torch.save(model.state_dict(), f"checkpoints/classifier_{epoch:03}.pt")



if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--total_epoch", type=int, default=100,
                                            help="total epochs")
    parse.add_argument("--data_dir", type=str, default="data",
                                            help="dataset directory")
    parse.add_argument("--batch", type=int, default=64,
                                            help="batch size")
    parse.add_argument("--resume", type=str, default="checkpoints/senet50_scratch_weight.pkl",
                                            help="pretrained weights for fine-tuning")
    parse.add_argument("--use_amp", action="store_true",
                                            help="use apex/amp")

    args = parse.parse_args()

    if args.use_amp:
        from apex import amp

    train(args, [2,7,8,9,10,12,14,18,22,24,26,32])