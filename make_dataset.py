import torch
import torchvision
from tqdm import tqdm
import argparse
import os

from models.stylegan2.model import Generator
from models.classifier import Classifier


device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main(args):
    G = Generator(args.size, 512, 8)
    G.load_state_dict(torch.load(args.g_ckpt)["g_ema"])
    G.to(device).eval()

    torch.cuda.manual_seed(0)
    mean_latent = G.mean_latent(5000)

    C = Classifier(args.emb_ckpt, args.detector_ckpt, device=device)


    os.makedirs(args.data_dir, exist_ok=True)
    latents = []

    for i in tqdm(range(args.n_sample)):
        torch.cuda.manual_seed(i)

        z = torch.randn(1, 512, device=device)
        image, _ = G([z], truncation=args.truncation, truncation_latent=mean_latent, randomize_noise=False)
        c = C(image)

        torch.save(c, os.path.join(args.data_dir, f"{i:06}.pt"))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--size", type=int, default=1024, help="size of generate")
    parse.add_argument("--g_ckpt", type=str, default="checkpoints/sg2_1024_ffhq.pt", help="pretrained weights of generator")
    parse.add_argument("--detector_ckpt", type=str, default="checkpoints/shape_predictor_68_face_landmarks.dat", help="pretrained weights of dlib shape_predictor")
    parse.add_argument("--emb_ckpt", type=str, default="checkpoints/id_embedder.pt", help="pretrained weights of id embedder")

    parse.add_argument("--data_dir", type=str, default="data", help="directory to save latents and labels")
    parse.add_argument("--n_sample", type=int, default=200000, help="number of training samples")
    parse.add_argument("--truncation", type=float, default=0.8, help="truncation value")

    args = parse.parse_args()

    main(args)