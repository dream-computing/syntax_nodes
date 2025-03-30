import numpy as np
from PIL import Image, ImageDraw
import torch
import random
from comfy.utils import ProgressBar

class PointillismNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dot_radius": ("INT", {"default": 3, "min": 1, "max": 10}),
                "dot_density": ("INT", {"default": 20000, "min": 1000, "max": 200000}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pointillism"

    def apply_pointillism(self, image, dot_radius, dot_density):
        pil_image = self.t2p(image)
        result_img = self.generate_pointillism(pil_image, dot_radius, dot_density)
        result_tensor = self.p2t(result_img)
        return (result_tensor,)

    def generate_pointillism(self, image, dot_radius, dot_density):
        width, height = image.size
        img_array = np.array(image)

        # Create a blank canvas
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Generate random dots based on the image colors
        for _ in range(dot_density):
            # Randomize the position of the dot
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # Get the color of the pixel at the random position
            color = tuple(img_array[y, x])

            # Draw a circle (dot) on the canvas
            draw.ellipse(
                (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                fill=color,
                outline=color,
            )

        return canvas

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0).to(self.device)

# Register the node
NODE_CLASS_MAPPINGS = {
    "PointillismNode": PointillismNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PointillismNode": "Pointillism Effect"
}