# Anomaly detection

Anomaly detecction using cycle-GAN, by Matthieu Blanke, Souhail Cadi, Jules Delemotte, and Benoît Oriol.

## Originial Repository 

[Image-to-Image Translation in PyTorch ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Compute embeddings

```bash
python -i embedding/embedding.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --gpu_ids -1
```

