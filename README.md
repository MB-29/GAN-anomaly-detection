# Anomaly detection

Anomaly detecction using cycle-GAN, by Matthieu Blanke, Souhail Cadi, Jules Delemotte, and Benoît Oriol.

## Originial Repository 

[Image-to-Image Translation in PyTorch ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Setup

Please install CycleGAN network and download the pix2pix maps dataset following the instructions provided in the original repository.
The checkpoint files and weights we obtained after a training phase of 11 epochs and that we used for our project are available [here](https://drive.google.com/file/d/1IEiccqF0YDm7Jk1aAevImJQbwlmnffCk/view?usp=sharing).

## Compute and save embeddings

```bash
python anomaly_detection/features.py --dataroot ./datasets/[Data folder] --name maps_cyclegan --model cycle_gan --gpu_ids -1
```

## Naive tests

Run 
```bash
python anomaly_detection/data_analysis.py
```

## Anomaly detection
Please find our work in the Jupyter notebooks [here](https://drive.google.com/file/d/1qr9_9XiEm-pW3jW0FLy37xIfX5apNW31/view?usp=sharing).
