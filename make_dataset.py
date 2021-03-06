import torch
import torchvision
from tqdm import tqdm
import argparse
import os

from models.stylegan2.model import Generator


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt, map_location="cpu")["g_ema"], strict=False)
    G.to(device).eval()
    mean_latent = G.mean_latent(5000)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "images"), exist_ok=True)
    latents = []

    p_bar = tqdm(total=args.n_sample)

    for i in range(args.n_sample // args.batch):
        with torch.no_grad():
            z = torch.randn(args.batch, 512, device=device)
            image, w = G([z], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent)
            w = w[:, 0, :]

        latents.append(w)

        for j in range(args.batch):
            torchvision.utils.save_image(image[j:j+1], 
                    os.path.join(args.data_dir, "images", f"{i*args.batch+j:06}.png"), normalize=True, range=(-1, 1))

        p_bar.update(args.batch)

    latents = torch.cat(latents)
    torch.save(latents, os.path.join(args.data_dir, "latents.pt"))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--size", type=int, default=1024,
                                            help="size of generated images")
    parse.add_argument("--g_ckpt", type=str, default="checkpoints/sg2_1024_ffhq.pt",
                                            help="pretrained weights of generator")
    parse.add_argument("--data_dir", type=str, default="data",
                                            help="directory to save latents and labels")

    parse.add_argument("--n_sample", type=int, default=200000,
                                            help="number of training samples")
    parse.add_argument("--batch", type=int, default=8,
                                            help="batch size")
    parse.add_argument("--truncation", type=float, default=0.8,
                                            help="truncation value")

    args = parse.parse_args()

    main(args)