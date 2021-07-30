# SGF

## Required
 - PyTorch
 - dlib
 - OpenCV
 - FFHQ-StyleGAN2
 - shape_predictor_68_face_landmarks.dat

~~You can use [my pretrained weights]() for Auxiliary Mapping Network. When using it, you have to use [this StyleGAN2 model](https://github.com/rosinality/stylegan2-pytorch#pretrained-checkpoints).~~


## Inference
Run `python translate_keypoints.py`.
 - left : A randomly generated source image.
 - right : A randomly generated image with a target property.
 - middle : Output images during processing.

## Train on your own dataset
1. Create a Classifier Network on `models/classifier.py`.
2. Run `make_dataset.py` to create a dataset by generating images and classifiering their properties.
   ```
   python make_dataset.py \
        --detector_ckpt checkpoints/shape_predictor_68_face_landmarks.dat \
        --size 256 \
        --g_ckpt checkpoints/sg2_256_ffhq.pt \
        --data_dir data \
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
        --c_dim 136 \
        --n_layer 15
   ```