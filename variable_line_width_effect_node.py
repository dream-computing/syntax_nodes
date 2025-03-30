import numpy as np
import cv2
from PIL import Image
import torch
from comfy.utils import ProgressBar  # Added progress bar import


class VariableLineWidthEffectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "line_spacing": ("INT", {"default": 5, "min": 1, "max": 20}),
                "displacement_strength": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 5}),
                "invert": ("BOOLEAN", {"default": False}),
                "color_intensity": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 5.0}),
                "start_color_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "start_color_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "start_color_b": ("INT", {"default": 255, "min": 0, "max": 255}),
                "end_color_r": ("INT", {"default": 255, "min": 0, "max": 255}),
                "end_color_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "end_color_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_variable_line_width"
    CATEGORY = "SyntaxNodes/Processing"

    def get_mask_array(self, mask, target_shape):
        """Convert mask to proper format and shape"""
        if mask is None:
            return np.ones((target_shape[0], target_shape[1]), dtype=np.float32)
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(mask):
            # Handle different tensor formats
            if len(mask.shape) == 4:  # BCHW format
                mask = mask[0, 0]  # Take first batch, first channel
            elif len(mask.shape) == 3:  # CHW format
                mask = mask[0]  # Take first channel
            mask = mask.cpu().numpy()
        
        # Ensure mask is float32 and properly scaled
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Resize mask to match target shape
        if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        return mask

    def get_color_for_displacement(self, displacement, max_displacement, color_intensity,
                                 start_r, start_g, start_b, end_r, end_g, end_b):
        """Color mapping with customizable colors"""
        normalized = min(1.0, displacement / (max_displacement * 0.5))
        
        if normalized < 0.1:
            return (255, 255, 255)
        
        r = int(start_r + (end_r - start_r) * normalized)
        g = int(start_g + (end_g - start_g) * normalized)
        b = int(start_b + (end_b - start_b) * normalized)
        
        r = int(min(255, r * color_intensity))
        g = int(min(255, g * color_intensity))
        b = int(min(255, b * color_intensity))
        
        return (b, g, r)  # OpenCV uses BGR

    def apply_variable_line_width(self, images, line_spacing, displacement_strength, 
                                line_thickness, invert, color_intensity,
                                start_color_r, start_color_g, start_color_b,
                                end_color_r, end_color_g, end_color_b,
                                mask=None):
        device = images.device
        batch_size = images.shape[0]
        out = []

        # Added progress bar
        pbar = ProgressBar(batch_size)

        for b in range(batch_size):
            img = images[b]
            img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            if img_np.shape[-1] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np.squeeze()

            if invert:
                gray = 255 - gray

            height, width = gray.shape
            
            # Get mask array with correct dimensions
            mask_array = self.get_mask_array(mask, (height, width))
            
            # Create base image with white lines
            base_result = np.full((height, width, 3), 255, dtype=np.uint8)
            
            # Create effect image (will be masked)
            effect_result = np.zeros((height, width, 3), dtype=np.uint8)
            
            max_displacement = displacement_strength
            
            for y in range(0, height, line_spacing):
                points = []
                colors = []
                last_displacement = 0
                
                for x in range(width):
                    # Get mask value for current position
                    mask_value = mask_array[y, x] if mask is not None else 1.0
                    intensity = gray[y, x] / 255.0 * mask_value
                    
                    displacement = int(displacement_strength * intensity)
                    displaced_y = min(max(y - displacement, 0), height - 1)
                    
                    displacement_change = abs(displacement - last_displacement)
                    color = self.get_color_for_displacement(
                        displacement_change,
                        max_displacement,
                        color_intensity,
                        start_color_r, start_color_g, start_color_b,
                        end_color_r, end_color_g, end_color_b
                    )
                    
                    points.append((x, displaced_y))
                    colors.append(color)
                    last_displacement = displacement

                # Draw lines
                if len(points) > 1:
                    points_arr = np.array(points, dtype=np.int32)
                    # Draw white base lines
                    cv2.polylines(base_result, [points_arr], False, (255, 255, 255), 
                                thickness=line_thickness, lineType=cv2.LINE_AA)
                    # Draw colored effect lines
                    for i in range(len(points) - 1):
                        pt1 = points[i]
                        pt2 = points[i + 1]
                        color = colors[i]
                        cv2.line(effect_result, pt1, pt2, color, 
                               thickness=line_thickness, lineType=cv2.LINE_AA)
            
            # Blend base and effect images using mask
            if mask is not None:
                mask_array_3d = np.stack([mask_array] * 3, axis=-1)
                result = base_result * (1 - mask_array_3d) + effect_result * mask_array_3d
            else:
                result = effect_result
            
            processed = torch.from_numpy(result.astype(np.float32) / 255.0)
            processed = processed.to(device)
            out.append(processed)

            # Update progress bar
            pbar.update_absolute(b + 1)

        result = torch.stack(out, dim=0)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "VariableLineWidthEffectNode": VariableLineWidthEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VariableLineWidthEffectNode": "Variable Line Width Effect"
}