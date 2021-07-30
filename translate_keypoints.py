from torch import optim
from torch import nn
import torch
from torchvision import transforms as tf
import torchvision
from tqdm import tqdm
import argparse
import random
import os

from models.network import AuxiliaryMappingNetwork
from models.stylegan2.model import Generator
from models.classifier import Classifier
from sgf import translate


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt, map_location="cpu")["g_ema"], strict=False)
    G.to(device).eval()

    C = Classifier(args.detector_ckpt)

    F = AuxiliaryMappingNetwork(args.n_layer, C.c_dim)
    F.load_state_dict(torch.load(args.f_ckpt, map_location="cpu"))
    F.to(device).eval()

    with torch.no_grad():
        w_mean = G.mean_latent(1000)
        z = torch.randn(1, 512)
        w0 = G.style(z)
        w0 += 0.2 * (w_mean - w0)
    
    with torch.no_grad():
        z = torch.randn(1, 512)
        image_t, _ = G([z], truncation=0.8, truncation_latent=w_mean, randomize_noise=False)
        c1 = C(image_t).unsqueeze(0)

    images, _ = translate(G, C, F, w0.to(device), c1.to(device), max_iteration=5, step=0.2)

    imgs = torch.cat(images + [image_t], 0)
    torchvision.utils.save_image(imgs, "translate.png", nrow=len(imgs), normalize=True, range=(-1, 1))



if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--detector_ckpt", type=str, default="checkpoints/shape_predictor_68_face_landmarks.dat",
                        help="pretrained weights of keypoints detector")
    parse.add_argument("--size", type=int, default=256,
                        help="size of generated images")
    parse.add_argument("--g_ckpt", type=str, default="checkpoints/sg2_256_ffhq.pt",
                        help="pretrained weights of generator")
    parse.add_argument("--f_ckpt", type=str, default="checkpoints/mapping.pt",
                        help="pretrained weights of mapping network")
    parse.add_argument("--n_layer", type=int, default=15,
                        help="number of mapping network layers")

    args = parse.parse_args()

    main(args)