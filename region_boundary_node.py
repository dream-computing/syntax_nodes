import numpy as np
import torch
from PIL import Image
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from skimage.segmentation import slic, mark_boundaries, watershed
    from skimage.filters import sobel
    from skimage.color import rgb2gray
    from comfy.utils import ProgressBar
except ImportError:
    print("Installing scikit-image...")
    install_package("scikit-image")
    from skimage.segmentation import slic, mark_boundaries, watershed
    from skimage.filters import sobel
    from skimage.color import rgb2gray

class RegionBoundaryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segments": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "compactness": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "line_color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_region_boundary"
    CATEGORY = "SyntaxNodes/Processing"

    def apply_region_boundary(self, image, segments, compactness, line_color):
        batch_size = image.shape[0]
        result = []
        
        # Initialize progress bar
        pbar = ProgressBar(batch_size)

        for b in range(batch_size):
            # Convert to numpy array
            img_np = np.array(self.t2p(image[b]))
            
            # Apply SLIC segmentation
            segments_slic = slic(img_np, n_segments=segments, compactness=compactness, start_label=1)
            
            # Use watershed for additional refinement
            gradient = sobel(rgb2gray(img_np))
            labels = watershed(gradient, segments_slic)
            
            # Draw region boundaries
            line_color_rgb = self.int_to_rgb(line_color)
            boundary_image = mark_boundaries(img_np, labels, color=line_color_rgb)
            
            # Convert back to tensor
            result.append(self.p2t(Image.fromarray((boundary_image * 255).astype(np.uint8))))
            
            # Update progress bar
            pbar.update_absolute(b + 1)
        
        return (torch.cat(result, dim=0),)

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t

    def int_to_rgb(self, color_int):
        return ((color_int >> 16) & 255) / 255.0, ((color_int >> 8) & 255) / 255.0, (color_int & 255) / 255.0

NODE_CLASS_MAPPINGS = {
    "RegionBoundaryNode": RegionBoundaryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionBoundaryNode": "Region Boundary Effect"
}