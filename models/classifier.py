from torch.nn import functional as F
import torch
from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys
import os

from models.senet import senet50


device = "cuda" if torch.cuda.is_available() else "cpu"


class Classifier:
    def __init__(self, detector_ckpt, classifier_ckpt):
        ckpt = torch.load(classifier_ckpt, map_location="cpu")
        num_cls = ckpt["fc.weight"].shape[0]
        self.senet = senet50(num_classes=num_cls)
        self.senet.load_state_dict(ckpt)
        self.senet.to(device).eval()

        self.face_predictor = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(detector_ckpt)

        self.c_dim = num_label + 68*2

    def get_attributes(self, img_tensor):
        img_tensor = F.interpolate(img_tensor, size=(224, 224))
        y = self.senet(img_tensor)
        return y[0]

    def get_landmark(self, img_tensor):
        img_tensor = F.interpolate(img_tensor, size=(128, 128))
        img = img_tensor[0].to("cpu").permute(1, 2, 0).detach().numpy()
        img = np.uint8(np.clip((img+1)*127.5, 0, 255))

        gry_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face = self.face_predictor(gry_img, 1)

        if len(face) == 0:
            return None

        landmark = self.landmark_predictor(gry_img, face[0])
        landmark = face_utils.shape_to_np(landmark)

        features = landmark.flatten() / 64.0 - 1

        return torch.from_numpy(features).to(device).float()

    def __call__(self, img_tensor):
        features1 = self.get_attributes(img_tensor)
        features2 = self.get_landmark(img_tensor)

        return torch.cat([features1, features2]).to(img_tensor.device)
