import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

content_path = "data/content/Bridge.png"
style_path = "data/style/The Night Café.png"
output_dir = 'output'

image_size = None

# "content" starts from the content image, "random" starts from random noise
init_method = "content"

# larger values make the output stay closer to the content image
content_weight = 1.0

# larger values make the output match the style image more strongly
style_weight = 50.0

# larger values encourage smoother images and reduce noise, but may make the result look blurry
# smaller values preserve more detail, but may leave more visual noise.
tv_weight = 0.1

use_relu = True
max_iter = 500
log_interval = 10
save_interval = 50
