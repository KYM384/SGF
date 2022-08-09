from torch.nn import functional as F
from torch import nn
import torch
import torchvision
from imutils import face_utils
import numpy as np
import dlib
import cv2


def build_id_embedder(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path)

    resnet = torchvision.models.resnet50()
    resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )

    resnet.load_state_dict(ckpt)
    resnet.to(device).eval()

    return resnet


def distance(p0, p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)


class Classifier:
    def __init__(self, embedder_ckpt, detector_ckpt, device="cuda"):
        self.id_emb = build_id_embedder(embedder_ckpt, device)

        self.bbox = dlib.rectangle(int(256*0.2), int(256*0.3), int(256*0.8), int(256*0.9))
        self.landmark_predictor = dlib.shape_predictor(detector_ckpt)

        # ID embedder = 512
        # eye size    = 2
        # mouth size  = 1
        self.c_dim = 512 + 2 + 1

    def get_identity(self, img_tensor):
        y = self.id_emb(img_tensor)
        return y[0]

    def get_landmark(self, img_tensor):
        img = img_tensor[0].to("cpu").permute(1, 2, 0).detach().numpy()
        img = np.uint8(np.clip((img+1)*127.5, 0, 255))
        gry_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        landmark = self.landmark_predictor(gry_img, self.bbox)
        landmark = face_utils.shape_to_np(landmark)

        leye = distance(landmark[38], landmark[40]) / distance(landmark[36], landmark[39])
        reye = distance(landmark[43], landmark[47]) / distance(landmark[42], landmark[45])
        mouth = distance(landmark[62], landmark[66]) / distance(landmark[48], landmark[54])

        return torch.tensor([leye, reye, mouth]).to(img_tensor)

    def __call__(self, img_tensor):
        size = img_tensor.shape[3]
        if size > 256:
            img_tensor = F.max_pool2d(img_tensor, kernel_size=size//256, stride=size//256)

        features1 = self.get_identity(img_tensor)
        features2 = 10 * self.get_landmark(img_tensor)

        return torch.cat([features1, features2])
