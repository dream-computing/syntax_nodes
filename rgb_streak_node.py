import numpy as np
import cv2
from PIL import Image
import torch

class RGBStreakNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "streak_length": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1
                }),
                "red_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "green_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "blue_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "decay": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 0.99,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_rgb_streak"
    CATEGORY = "image/effects"

    def apply_rgb_streak(self, image, streak_length, red_intensity, green_intensity, blue_intensity, threshold, decay):
        # Convert input tensor to numpy array
        img = image[0].cpu().numpy()
        
        # Create output array (black background)
        result = np.zeros_like(img)
        height, width = img.shape[:2]
        
        # Process each channel
        for channel_idx, intensity in enumerate([red_intensity, green_intensity, blue_intensity]):
            channel = img[:, :, channel_idx]
            
            # Create mask for pixels above threshold
            mask = channel > threshold
            
            # Create streaks
            streak_mask = np.zeros_like(channel)
            
            # Direction based on channel (Red left, Green right, Blue alternating)
            if channel_idx == 0:  # Red
                direction = -1
            elif channel_idx == 1:  # Green
                direction = 1
            else:  # Blue
                direction = -1
                
            # Apply streaking effect
            for i in range(streak_length):
                # Calculate decay factor
                current_decay = decay ** i
                
                # Shift and apply intensity
                if direction < 0:
                    shifted = np.roll(mask, -i, axis=1)
                else:
                    shifted = np.roll(mask, i, axis=1)
                
                # Add streak with decay and intensity
                streak_contribution = channel * shifted * current_decay * intensity
                streak_mask = np.maximum(streak_mask, streak_contribution)
            
            # Add random variation
            noise = np.random.normal(0, 0.02, streak_mask.shape) * (streak_mask > 0)
            streak_mask += noise
            
            # Add to result
            result[:, :, channel_idx] = streak_mask
        
        # Normalize and clip
        result = np.clip(result, 0, 1)
        
        # Convert back to torch tensor
        return (torch.from_numpy(result).float().unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "RGBStreakNode": RGBStreakNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBStreakNode": "RGB Channel Streak"
}