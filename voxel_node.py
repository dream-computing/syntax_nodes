import numpy as np
from PIL import Image, ImageDraw
import torch
from comfy.utils import ProgressBar

class VoxelNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {
                    "default": 16, 
                    "min": 4, 
                    "max": 64,
                    "step": 1
                }),
                "block_depth": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 32,
                    "step": 1
                }),
                "shading": ("FLOAT", {
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
    CATEGORY = "SyntaxNodes/Processing"
    
    def process_image(self, image, block_size, block_depth, shading, mask=None):
        # Get batch size and create progress bar
        batch_size = image.shape[0]
        pbar = ProgressBar(batch_size)
        
        # Initialize list to store processed images
        processed_tensors = []
        
        # Process each image in the batch
        for idx in range(batch_size):
            # Extract single image from batch
            single_image = image[idx:idx+1]
            
            # Convert from ComfyUI image format to PIL
            pil_image = self.t2p(single_image)
            
            # Ensure the image is in RGB mode
            pil_image = pil_image.convert('RGB')
            original_array = np.array(pil_image)
            
            # Process the image
            processed_image = self.create_voxel(pil_image, block_size, block_depth, shading)
            processed_array = np.array(processed_image)
            
            # Handle masking for single image
            if mask is not None:
                mask_array = mask[idx:idx+1].squeeze().cpu().numpy()
                # Ensure mask has same dimensions as image
                if len(mask_array.shape) == 2:
                    mask_array = mask_array[..., np.newaxis]
                # Apply mask
                masked_array = original_array * (1 - mask_array) + processed_array * mask_array
                processed_image = Image.fromarray(masked_array.astype(np.uint8))
            
            # Convert back to tensor and append to list
            processed_tensor = self.p2t(processed_image)
            processed_tensors.append(processed_tensor)
            
            # Update progress bar
            pbar.update_absolute(idx + 1)
        
        # Concatenate all processed tensors along batch dimension
        final_output = torch.cat(processed_tensors, dim=0)
        
        return (final_output,)

    def create_voxel(self, image, block_size, block_depth, shading):
        width, height = image.size
        
        # Create a new image with the same size
        result = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(result)
        
        # Calculate grid dimensions
        cols = width // block_size + (1 if width % block_size else 0)
        rows = height // block_size + (1 if height % block_size else 0)
        
        # Create blocks
        for row in range(rows):
            for col in range(cols):
                # Calculate block position
                x = col * block_size
                y = row * block_size
                
                # Sample color from original image
                sample_x = min(x + block_size//2, width-1)
                sample_y = min(y + block_size//2, height-1)
                base_color = image.getpixel((sample_x, sample_y))
                
                # Draw main face of block
                block_points = [
                    (x, y),
                    (x + block_size, y),
                    (x + block_size, y + block_size),
                    (x, y + block_size)
                ]
                draw.polygon(block_points, fill=base_color)
                
                if block_depth > 0:
                    # Calculate offset for 3D effect
                    offset = block_depth
                    
                    # Top face (if visible)
                    if y > 0:
                        top_color = tuple(int(c * (1 + shading)) for c in base_color)
                        top_points = [
                            (x, y),
                            (x + offset, y - offset),
                            (x + block_size + offset, y - offset),
                            (x + block_size, y)
                        ]
                        draw.polygon(top_points, fill=top_color)
                    
                    # Right face
                    right_color = tuple(int(c * (1 - shading)) for c in base_color)
                    right_points = [
                        (x + block_size, y),
                        (x + block_size + offset, y - offset),
                        (x + block_size + offset, y + block_size - offset),
                        (x + block_size, y + block_size)
                    ]
                    draw.polygon(right_points, fill=right_color)
        
        return result

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0).to(self.device)
        return t

NODE_CLASS_MAPPINGS = {
    "VoxelNode": VoxelNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxelNode": "Voxel Block Effect"
}