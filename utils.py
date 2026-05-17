import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path, target_shape=None):
    """Load an image and convert to tensor"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform_list = []

    if target_shape is not None:
        # target shape: [H, W]
        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)
        transform_list.append(transforms.Resize(target_shape))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = transforms.Compose(transform_list)

    # add batch dimension: [C, H, W] -> [1, C, H, W]
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    image = tensor.clone().detach().cpu().squeeze(0)  # [C, H, W]

    # denormalize
    for c in range(3):
        image[c] = image[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]

    # clamp to [0, 1]
    image = image.clamp(0, 1)

    # convert to numpy: [C, H, W] -> [H, W, C]
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    return Image.fromarray(image)


def save_image(tensor, path):
    """Save tensor as an image file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_image(tensor)
    image.save(path)
    print(f'Image saved to: {path}')


def gram_matrix(feature_map):
    """
    Compute Gram matrix to capture style information
    Input: feature_map (batch, channels, height, width)
    Output: Gram matrix (batch, channels, channels)
    """
    b, c, h, w = feature_map.size()
    features = feature_map.view(b, c, h * w)  # [B, C, HW]
    gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    # normalize
    gram = gram / (c * h * w)
    return gram


def prepare_init_image(content_tensor, init_method='content'):
    """
    Prepare initial image for optimization
    init_method: "content" start from content image, "random" start from random noise
    """
    if init_method == 'content':
        return content_tensor.clone()
    elif init_method == 'random':
        return torch.randn_like(content_tensor)
    else:
        raise ValueError(f'Unknown initialization method: {init_method}')
