# VisionGPT-2 Image Captioning Model

This project implements an image captioning model using a combination of Vision Transformer (ViT) and GPT-2. The model generates captions for images by leveraging the visual context from ViT and the textual context from GPT-2.

## Features

- **Dataset Support**: Supports COCO 2017 and Flickr30k datasets.
- **Augmentations**: Uses `albumentations` for image augmentations, which is faster than `torchvision`.
- **Custom Model Architecture**:
  - ViT as the encoder for extracting image features.
  - GPT-2 as the decoder with added cross-attention layers for combining image and text contexts.
- **Dynamic Padding**: Implements a custom `collate_fn` for efficient batching with dynamic padding.
- **Training Pipeline**:
  - Mixed-precision training with `torch.cuda.amp`.
  - Optimizer: Adam.
  - Scheduler: OneCycleLR.
  - Metrics: Cross-entropy loss and perplexity.
- **Caption Generation**:
  - Supports both deterministic and probabilistic generation.
  - Temperature control for sampling.

