import torch
import torchvision
from tqdm import tqdm
import argparse
import os

from models.stylegan2.model import Generator
from models.classifer import Classifier


device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt, map_location="cpu")["g_ema"], strict=False)
    G.to(device).eval()
    mean_latent = G.mean_latent(5000)

    C = Classifier(args.detector_ckpt)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "latents"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "images"), exist_ok=True)

    exist_n = len(os.listdir(os.path.join(args.data_dir, "images")))
    print(f"continue from {exist_n}")

    p_bar = tqdm(total=args.n_sample)

    for i in range(args.n_sample // args.batch):
        if i * args.batch < exist_n:
            p_bar.update(args.batch)
            continue

        with torch.no_grad():
            z = torch.randn(args.batch, 512, device=device)
            image, w = G([z], return_latents=True, truncation=args.truncation, truncation_latent=mean_latent)
            w = w[:, 0, :]
        
        for j in range(args.batch):
            c = C(image[j:j+1])

            if c is None:
                continue

            torch.save({"w":w[j], "c":c}, os.path.join(args.data_dir, "latents", f"{i*args.batch+j:06}.pt"))
            torchvision.utils.save_image(image[j:j+1], 
                    os.path.join(args.data_dir, "images", f"{i*args.batch+j:06}.png"), normalize=True, range=(-1, 1))

        p_bar.update(args.batch)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--detector_ckpt", type=str, default="checkpoints/shape_predictor_68_face_landmarks.dat",
                                            help="weights of keypoints detector")
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