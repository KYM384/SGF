from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch import optim
from torch import nn
import torch

import torchvision

from tqdm import tqdm
import numpy as np
import argparse
import os

from models.network import AuxiliaryMappingNetwork


device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.paths = os.listdir(base_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        a = torch.load(os.path.join(self.base_dir, path))
        return a["w"].float(), a["c"].float()


def train(args):
    writer = SummaryWriter("logs")

    F = AuxiliaryMappingNetwork(args.n_layer, args.c_dim)
    F.to(device)
    F.train()

    loss = nn.MSELoss()
    loss.to(device)

    opt = optim.Adam(F.parameters(), lr=0.0002)

    dataset = Dataset(args.data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    total_bar = tqdm(total = args.total_iter)
    total_epoch = args.total_iter // len(dataloader)
    iter_ = 0

    for i in range(total_epoch + 1):
        l_epoch = []
        for w, c in dataloader:
            w = w.to(device)
            c = c.to(device)
            w_hat = F(w, c)

            l = loss(w, w_hat)

            opt.zero_grad()
            l.backward()
            opt.step()

            writer.add_scalar("loss", l.to("cpu").item(), iter_)
            l_epoch.append(l.to("cpu").item())

            iter_ += 1
            total_bar.update()

        if (i+1) % 1000 == 0:
            print(f"[epoch] {i+1} / {total_epoch+1}   [loss] {sum(l_epoch) / len(l_epoch)}")

        torch.save(F.state_dict(), f"checkpoints/mapping.pt")



if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--total_iter", type=int, default=500000,
                                            help="total iterations")
    parse.add_argument("--data_dir", type=str, default="data",
                                            help="dataset directory")
    parse.add_argument("--batch", type=int, default=8,
                                            help="batch size")
    parse.add_argument("--c_dim", type=int, default=68*2,
                                            help="dimention of a feature vector")
    parse.add_argument("--n_layer", type=int, default=15,
                                            help="number of mapping network layers")

    args = parse.parse_args()

    train(args)