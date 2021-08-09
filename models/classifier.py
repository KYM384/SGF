from torch.nn import functional as F
import torch
from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys
import os

from models.senet import senet50
sys.path.append("models/face_parsing/")
from model import BiSeNet


device = "cuda" if torch.cuda.is_available() else "cpu"


class Classifier:
    def __init__(self, detector_ckpt, classifier_ckpt, parsing_ckpt):
        self.senet = senet50(num_classes=28)
        self.senet.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))
        self.senet.to(device).eval()

        self.face_predictor = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(detector_ckpt)
        
        self.bisenet = BiSeNet(n_classes=19)
        self.bisenet.load_state_dict(torch.load(parsing_ckpt, map_location="cpu"))
        self.bisenet.to(device).eval()

        self.c_dim = 28 + 68*2 + 3

    def get_attributes_label(self, data_dir):
        remove_idx = [2,7,8,9,10,12,14,18,22,24,26,32]

        with open(os.path.join(data_dir, "list_attr_celeba.txt"), "r") as f:
            attrs = f.read().split("\n")[1]

        attrs_label = []
        for i, attr in enumerate(attrs.split()):
            if not i+1 in remove_idx:
                attrs_label.append(attr)
        
        return attrs_label

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

    def get_haircolor(self, img_tensor):
        img_tensor = F.interpolate(img_tensor, size=(512, 512))
        out = self.bisenet(img_tensor)[0]
        out = out.argmax(1)
        hair = img_tensor[:, :, out[0]==17].mean(2)[0]

        if any(torch.isnan(hair)):
            hair = torch.zeros_like(hair).to(img_tensor.device).float()

        return hair

    def __call__(self, img_tensor):
        features = torch.zeros(self.c_dim).to(img_tensor.device).float()
        features[:28] = self.get_attributes(img_tensor)
        features[28:28+68*2] = self.get_landmark(img_tensor)
        features[28+68*2:28+68*2+3] = self.get_haircolor(img_tensor)

        return features
