import torch

import config as cfg
from model import VGG19
from utils import gram_matrix, load_image, save_image, prepare_init_image
from loss import build_loss


def neural_style_transfer(config):
    device = cfg.device
    if device == 'cuda':
        gpu = torch.cuda.get_device_name(device)
        print(f"💻 Device: {gpu}")
    else:
        print("💻 Device: CPU")

    # --------------- Load model ---------------
    model = VGG19(requires_grad=False, use_relu=config['use_relu']).to(device)
    model.eval()

    content_index = model.content_feature_maps_index
    style_indices = model.style_feature_maps_indices

    # --------------- Load images ---------------
    content_image = load_image(config['content_path'], config['image_size']).to(device)
    style_image = load_image(config['style_path'], config['image_size']).to(device)

    print(f"Content image size: {content_image.shape}")
    print(f"Style image size: {style_image.shape}")

    # --------------- Extract target features ---------------
    with torch.no_grad():
        content_features = model(content_image)
        style_features = model(style_image)

    # content target: conv4_2 layer features
    content_targets = content_features[content_index].detach()

    # style target: gram matrices of each style layer
    style_target_grams = []
    for i in style_indices:
        gram = gram_matrix(style_features[i])
        style_target_grams.append(gram.detach())

    # --------------- Initialize image to optimize ---------------
    optimizing_image = prepare_init_image(content_image, config['init_method']).to(device)
    optimizing_image.requires_grad_(True)

    # --------------- Setup optimizer ---------------
    optimizer = torch.optim.LBFGS(
        [optimizing_image],
        max_iter=config['max_iter'],
        line_search_fn='strong_wolfe'
    )

    # --------------- Optimization loop ---------------
    iteration = [0]  # use list for mutability inside closure

    def closure():
        optimizer.zero_grad()
        current_features = model(optimizing_image)

        loss, c_loss, s_loss, tv_loss = build_loss(
            current_features, content_targets, style_target_grams,
            optimizing_image, content_index, style_indices,
            content_weight=config['content_weight'],
            style_weight=config['style_weight'],
            tv_weight=config['tv_weight']
        )
        loss.backward()

        # Logging
        if iteration[0] % config["log_interval"] == 0:
            print(f"Iteration {iteration[0]:>4d} | "
                  f"Total loss: {loss.item():.4f} | "
                  f"Content: {c_loss.item():.6f} | "
                  f"Style: {s_loss.item():.8f} | "
                  f"TV: {tv_loss.item():.6f}")

        # Save intermediate results
        if iteration[0] % config["save_interval"] == 0:
            save_image(optimizing_image, f"{config['output_dir']}/iter_{iteration[0]:04d}.png")

        iteration[0] += 1
        return loss

    # Run optimization
    optimizer.step(closure)

    # --------------- Save final result ---------------
    final_path = f"{config['output_dir']}/final_result.png"
    save_image(optimizing_image, final_path)
    print('\n✅ Style transfer completed!')
    print(f'💾 Final result saved to: {final_path}')

    return optimizing_image


if __name__ == '__main__':
    config = {
        'content_path': cfg.content_path,
        'style_path': cfg.style_path,
        'output_dir': cfg.output_dir,
        'image_size': cfg.image_size,
        'init_method': cfg.init_method,
        'content_weight': cfg.content_weight,
        'style_weight': cfg.style_weight,
        'tv_weight': cfg.tv_weight,
        'max_iter': cfg.max_iter,
        'use_relu': cfg.use_relu,
        'log_interval': cfg.log_interval,
        'save_interval': cfg.save_interval
    }

    neural_style_transfer(config)
