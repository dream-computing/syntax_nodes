import numpy as np
import torch
from PIL import Image, ImageDraw
from comfy.utils import ProgressBar

class PaperCraftNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "triangle_size": ("INT", {
                    "default": 32, 
                    "min": 8, 
                    "max": 128,
                    "step": 4
                }),
                "fold_depth": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 32,
                    "step": 1
                }),
                "shadow_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "🎨 Image/Effects"

    def create_papercraft(self, image, triangle_size, fold_depth, shadow_strength):
        width, height = image.size
        result = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(result)

        # Calculate grid dimensions
        cols = width // triangle_size + 2
        rows = height // triangle_size + 2

        # Create triangular grid
        for row in range(rows):
            for col in range(cols):
                x = col * triangle_size
                y = row * triangle_size
                
                # Calculate points for two triangles that make up a square
                if (row + col) % 2 == 0:
                    points1 = [
                        (x, y),
                        (x + triangle_size, y),
                        (x, y + triangle_size)
                    ]
                    points2 = [
                        (x + triangle_size, y),
                        (x + triangle_size, y + triangle_size),
                        (x, y + triangle_size)
                    ]
                else:
                    points1 = [
                        (x, y),
                        (x + triangle_size, y),
                        (x + triangle_size, y + triangle_size)
                    ]
                    points2 = [
                        (x, y),
                        (x + triangle_size, y + triangle_size),
                        (x, y + triangle_size)
                    ]

                # Sample colors from the center of each triangle
                def get_triangle_center(points):
                    cx = sum(p[0] for p in points) // 3
                    cy = sum(p[1] for p in points) // 3
                    return (min(cx, width-1), min(cy, height-1))

                center1 = get_triangle_center(points1)
                center2 = get_triangle_center(points2)
                
                base_color1 = image.getpixel(center1)
                base_color2 = image.getpixel(center2)

                # Apply lighting effects
                light_color1 = tuple(int(c * (1 + shadow_strength)) for c in base_color1)
                light_color2 = tuple(int(c * (1 - shadow_strength)) for c in base_color2)

                # Draw main triangles
                draw.polygon(points1, fill=light_color1)
                draw.polygon(points2, fill=light_color2)

                # Add fold effects if depth is specified
                if fold_depth > 0:
                    for d in range(fold_depth):
                        shadow_factor = 1 - (d/fold_depth) * shadow_strength
                        edge_color1 = tuple(int(c * shadow_factor) for c in base_color1)
                        edge_color2 = tuple(int(c * shadow_factor) for c in base_color2)
                        
                        # Draw edges with varying shadow
                        draw.line([points1[0], points1[1]], fill=edge_color1, width=2)
                        draw.line([points1[1], points1[2]], fill=edge_color1, width=2)
                        draw.line([points1[2], points1[0]], fill=edge_color1, width=2)
                        
                        draw.line([points2[0], points2[1]], fill=edge_color2, width=2)
                        draw.line([points2[1], points2[2]], fill=edge_color2, width=2)
                        draw.line([points2[2], points2[0]], fill=edge_color2, width=2)

        return result

    def process_image(self, image, triangle_size, fold_depth, shadow_strength, mask=None):
        batch_size = image.shape[0]
        processed_images = []
        
        pbar = ProgressBar(batch_size)
        
        for b in range(batch_size):
            # Convert tensor to PIL Image
            pil_image = self.tensor_to_pil(image[b:b+1])
            
            # Process the image
            processed = self.create_papercraft(pil_image, triangle_size, fold_depth, shadow_strength)
            
            # Convert back to tensor
            processed_tensor = self.pil_to_tensor(processed)
            
            # Apply mask if provided
            if mask is not None:
                mask_b = mask[b:b+1] if mask is not None else None
                if mask_b is not None:
                    processed_tensor = image[b:b+1] * (1 - mask_b) + processed_tensor * mask_b
            
            processed_images.append(processed_tensor)
            pbar.update_absolute(b + 1)
        
        return (torch.cat(processed_images, dim=0),)

    def tensor_to_pil(self, tensor):
        i = 255.0 * tensor.cpu().numpy().squeeze()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def pil_to_tensor(self, pil_image):
        i = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(i).unsqueeze(0).to(self.device)

NODE_CLASS_MAPPINGS = {
    "PaperCraftNode": PaperCraftNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaperCraftNode": "Epic Paper Craft Effect"
}