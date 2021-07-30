from torch.nn import functional as F
import torch
from imutils import face_utils
import numpy as np
import dlib
import cv2

class Classifier:
    def __init__(self, detector_ckpt):
        self.face_predictor = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(detector_ckpt)

        self.c_dim = 68*2

    def __call__(self, img_tensor):
        img_tensor = F.interpolate(img_tensor, size=(128, 128))
        img = img_tensor[0].to("cpu").permute(1, 2, 0).detach().numpy()
        img = np.uint8(np.clip((img+1)*127.5, 0, 255))

        features = np.zeros(self.c_dim, dtype=np.float32)

        gry_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face = self.face_predictor(gry_img, 1)

        if len(face) == 0:
            return None

        landmark = self.landmark_predictor(gry_img, face[0])
        landmark = face_utils.shape_to_np(landmark)

        features[:68*2] = landmark.flatten() / 64.0 - 1

        return torch.from_numpy(features)
