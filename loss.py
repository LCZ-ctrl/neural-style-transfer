import torch
from torch.nn.functional import mse_loss

from utils import gram_matrix


def content_loss(current_features, target_features):
    """
    Content loss: MSE of features between current image and content image
    """
    return mse_loss(current_features, target_features)


def style_loss(current_features_list, target_grams):
    """
    Style loss: sum of MSE of Gram matrices across multiple layers between current image and style image
    :param current_features_list: list of style features of current image per layer
    :param target_grams: list of Gram matrices of style image per layer
    """
    loss = 0.0
    for current_feat, target_gram in zip(current_features_list, target_grams):
        current_gram = gram_matrix(current_feat)
        loss += mse_loss(current_gram, target_gram, reduction='sum')

    loss = loss / len(target_grams)
    return loss


def total_variation_loss(image):
    """
    Total variation loss: encourages spatial smoothness and reduces noise

    This regularization encourages neighboring pixels to be similar, which helps
    reduce noise and produces a smoother final image
    """
    tv_h = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    num_pixels = image.numel()  # batch * channels * height * width
    return (tv_h + tv_w) / num_pixels


def build_loss(current_features, content_targets, style_target_grams,
               optimizing_image, content_index, style_indices,
               content_weight, style_weight, tv_weight):
    """
    Compute total loss
    :param current_features: model output of current image
    :param content_targets: features of content image at content layer
    :param style_target_grams: list of Gram matrices of style image at style layers
    :param optimizing_image: current image tensor being optimized
    :param content_index: index of content feature layer
    :param style_indices: list of indices of style feature layers
    :param content_weight: weight for content loss
    :param style_weight: weight for style loss
    :param tv_weight: weight for total variation loss
    """
    # content loss
    c_loss = content_loss(current_features[content_index], content_targets)

    # style loss
    current_style_features = [current_features[i] for i in style_indices]
    s_loss = style_loss(current_style_features, style_target_grams)

    # variation loss
    tv_loss = total_variation_loss(optimizing_image)

    # total loss
    loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss

    return loss, c_loss, s_loss, tv_loss
