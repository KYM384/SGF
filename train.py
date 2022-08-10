from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch import optim
from torch import nn
import torch

import torchvision

from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import os

from models.network import AuxiliaryMappingNetwork
from models.stylegan2.model import Generator


device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.paths = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        c = torch.load(os.path.join(self.data_dir, path))

        seed = int(os.path.splitext(path)[0])
        torch.cuda.manual_seed(seed)
        z = torch.randn(1, 512, device=device)

        return z, c



def train(args):
    writer = SummaryWriter("logs")

    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt)["g_ema"])
    G.to(device).eval()

    torch.cuda.manual_seed(0)
    mean_latent = G.mean_latent(5000)


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
        for z, c in dataloader:
            if i > args.total_iter:
                break

            c = c.to(device)
            with torch.no_grad():
                w = G.style(z).squeeze(1)
                w -= (1 - args.truncation) * (mean_latent - w)

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

    parse.add_argument("--total_iter", type=int, default=500000, help="total iterations")
    parse.add_argument("--data_dir", type=str, default="data", help="dataset directory")
    parse.add_argument("--batch", type=int, default=8, help="batch size")
    parse.add_argument("--n_layer", type=int, default=15, help="number of mapping network layers")

    parse.add_argument("--c_dim", type=int, default=515, help="conditional vector length")
    parse.add_argument("--size", type=int, default=1024, help="size of generate")
    parse.add_argument("--g_ckpt", type=str, default="checkpoints/sg2_1024_ffhq.pt", help="pretrained weights of generator")
    parse.add_argument("--truncation", type=float, default=0.8, help="truncation value")

    args = parse.parse_args()

    train(args)