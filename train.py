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
from models.classifier import Classifier


device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, C):
        super().__init__()
        self.base_dir = base_dir
        self.paths = os.listdir(os.path.join(base_dir, "images"))

        self.C = C
        self.transforms = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(0.5, 0.5)
                            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.base_dir, "images", path))
        w = torch.load(os.path.join(self.base_dir, "latents", path.replace(".png", ".pt")))
        try:
            return w, self.C(self.transforms(img).unsqueeze(0).to(device))
        except:
            return self.__getitem__(np.random.randint(0, self.__len__()-1))


def train(args):
    writer = SummaryWriter("logs")

    C = Classifier(args.detector_ckpt, args.classifier_ckpt, args.parsing_ckpt)

    F = AuxiliaryMappingNetwork(args.n_layer, C.c_dim)
    F.to(device)
    F.train()

    loss = nn.MSELoss()
    loss.to(device)

    opt = optim.Adam(F.parameters(), lr=0.0002)

    dataset = Dataset(args.data_dir, C)
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
    parse.add_argument("--n_layer", type=int, default=15,
                                            help="number of mapping network layers")

    parse.add_argument("--detector_ckpt", type=str, default="checkpoints/shape_predictor_68_face_landmarks.dat",
                                            help="weights of keypoints detector")
    parse.add_argument("--classifier_ckpt", type=str, default="checkpoints/attributes_classifier.pt",
                                            help="weights of classifier")
    parse.add_argument("--parsing_ckpt", type=str, default="checkpoints/parsing.pt",
                                            help="weights of parsing model")

    args = parse.parse_args()

    train(args)