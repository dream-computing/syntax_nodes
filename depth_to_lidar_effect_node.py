import cv2
import numpy as np
import torch
from PIL import Image
from comfy.utils import ProgressBar

class DepthToLidarEffectNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_frame = None  # Maintain single previous frame for temporal consistency

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "smoothing_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    # Fix: Make FUNCTION match the actual method name
    FUNCTION = "process_depth_map"
    CATEGORY = "SyntaxNodes/Processing"
    
    def process_depth_map(self, depth_map, smoothing_factor, line_thickness):
        batch_size = depth_map.shape[0]
        out = []
        
        # Initialize progress bar
        pbar = ProgressBar(batch_size)

        for b in range(batch_size):
            # Convert depth map tensor to a PIL image
            depth_image = self.t2p(depth_map[b:b+1])
            depth_array = np.array(depth_image)

            # Normalize and process the depth map
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_blurred = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
            edges = cv2.Canny(depth_blurred, 50, 150)

            # Check if prev_frame exists and matches the current frame dimensions
            if self.prev_frame is not None and self.prev_frame.shape != edges.shape:
                self.prev_frame = None  # Reset if sizes differ

            # Temporal smoothing with previous frame
            if self.prev_frame is not None:
                edges = cv2.addWeighted(edges, smoothing_factor, self.prev_frame, 1 - smoothing_factor, 0)
            
            # Update the previous frame
            self.prev_frame = edges.copy()

            # Convert edges to white lines on a black background
            output = np.zeros_like(depth_array)
            output[edges > 0] = 255

            # Convert processed image back to tensor
            output_image = Image.fromarray(output)
            output_tensor = self.p2t(output_image)
            out.append(output_tensor)

            # Update progress bar
            pbar.update_absolute(b + 1)

        return (torch.cat(out, dim=0),)

    def reset(self):
        """Reset the previous frame memory for a new sequence."""
        self.prev_frame = None

    def t2p(self, t):
        # Convert ComfyUI tensor format to PIL image
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        # Convert PIL image to ComfyUI tensor format and move to GPU if available
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0).to(self.device)
        return t

NODE_CLASS_MAPPINGS = {
    "DepthToLidarEffectNode": DepthToLidarEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthToLidarEffectNode": "Depth to LIDAR Effect"
}