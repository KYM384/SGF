# SGF
Unofficial PyTorch implementation of [Surrogate Gradient Field for Latent Space Manipulation](https://arxiv.org/abs/2104.09065)

## Required
 - PyTorch
 - dlib
 - OpenCV


## Pretrained Checkpoints
 - StyleGAN2 model (from [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch#pretrained-checkpoints))
 - shape_predictor_68_face_landmarks.dat ([link](http://dlib.net/files/))
 - Face Parsing model (from [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch))
 - Face Attributes Classifier ([my checkpoint](https://drive.google.com/file/d/1dRGpRYpZr0BlLICpVhFxFEZrUUzmnG7D/view?usp=sharing))
 - Auxiliary Mapping ([my checkpoint](https://drive.google.com/file/d/1kE7xcqIr63aHDyCeutfyVoG7Kr682ceI/view?usp=sharing))


## Inference
Run `python translate_keypoints.py`. This script aligns the face keypoints of a randomly generated image with another randomly generated image. 


## Train on your own dataset
1. Create a Classifier Network on `models/classifier.py`.
2. Run `make_dataset.py` to create a dataset by generating images and classifiering their properties.
   ```
   python make_dataset.py \
        --size [images size of stylegan2] \
        --g_ckpt [the pretrained checkpoint of stylegan2] \
        --data_dir [where you want to save training data] \
        --n_sample 200000 \
        --batch 8 \
        --truncation 0.8
   ```
3. Run `train.py`.
   ```
   python train.py \
        --total_iter 500000 \
        --data_dir data \
        --batch 8 \
        --n_layer 15
   ```

## License
StyleGAN2 codes and model are from [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

Face-Parsing codes and checkpoint are from [Zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).

`senet.py` is from [cydonia999/VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch/blob/master/models/senet.py), and the face attributes classifier model was transfer trained from [pretrained weights](https://github.com/cydonia999/VGGFace2-pytorch#pretrained-models) provided by the above repository.
