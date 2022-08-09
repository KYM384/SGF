import torch
import torchvision
from tqdm import tqdm
import argparse
import random
import os

from models.network import AuxiliaryMappingNetwork
from models.stylegan2.model import Generator
from models.classifier import Classifier


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt)["g_ema"])
    G.to(device).eval()
    mean_latent = G.mean_latent(5000)

    C = Classifier(args.emb_ckpt, args.detector_ckpt, device=device)

    F = AuxiliaryMappingNetwork(args.n_layer, C.c_dim)
    F.load_state_dict(torch.load(args.ckpt))
    F.to(device).eval()

    with torch.no_grad():
        z = torch.randn(1, 512, device=device)
        image, w = G([z], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent)
        w = w[:, 0, :]

        c = C(image)

    c = c.unsqueeze(0).to(device)
    w_hat = F(w, c)

    print((w-w_hat).pow(2).mean().item())

    rec_image, _ = G([w_hat], input_is_latent=True)

    save_image = torch.cat([image, rec_image], 0)
    torchvision.utils.save_image(save_image, "rec.png", nrow=2, normalize=True, range=(-1, 1))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--ckpt", type=str, default="checkpoints/mapping.pt", help="weights of mapping network")
    parse.add_argument("--detector_ckpt", type=str, default="checkpoints/shape_predictor_68_face_landmarks.dat", help="pretrained weights of dlib shape_predictor")
    parse.add_argument("--emb_ckpt", type=str, default="checkpoints/id_embedder.pt", help="pretrained weights of id embedder")
    parse.add_argument("--size", type=int, default=1024, help="size of generate")
    parse.add_argument("--g_ckpt", type=str, default="checkpoints/sg2_1024_ffhq.pt", help="pretrained weights of generator")
    parse.add_argument("--n_layer", type=int, default=15, help="number of mapping network layers")
    parse.add_argument("--truncation", type=float, default=0.8, help="truncation value")

    args = parse.parse_args()

    main(args)