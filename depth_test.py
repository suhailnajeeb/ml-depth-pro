from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import depth_pro
import os

# Paths
image_path = './data/example.jpg'
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# Load model and preprocessing
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess your image
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Inference
with torch.no_grad():
    prediction = model.infer(image, f_px=f_px)

# Get depth map
depth = prediction["depth"].detach().cpu().numpy().squeeze()

# Save the depth map as .npz
basename = os.path.splitext(os.path.basename(image_path))[0]
depth_output_path = os.path.join(output_dir, f"{basename}.npz")
np.savez_compressed(depth_output_path, depth=depth)
print(f"Saved depth map (npz) to: {depth_output_path}")

# Prepare color-mapped visualization
inverse_depth = 1 / depth
max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
min_invdepth_vizu = max(1 / 250, inverse_depth.min())
inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
    max_invdepth_vizu - min_invdepth_vizu
)

cmap = plt.get_cmap("turbo")
color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)

# Save color depth image
color_output_path = os.path.join(output_dir, f"{basename}.jpg")
Image.fromarray(color_depth).save(color_output_path, format="JPEG", quality=90)
print(f"Saved color-mapped depth (jpg) to: {color_output_path}")