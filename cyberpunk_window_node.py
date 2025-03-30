import numpy as np
import cv2
from PIL import Image
import torch
import random
from comfy.utils import ProgressBar

class CyberpunkWindowNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "custom_text": ("STRING", {"default": "MOVEMENT", "multiline": False}),
                "edge_threshold1": ("FLOAT", {
                    "default": 100,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "edge_threshold2": ("FLOAT", {
                    "default": 200,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "min_window_size": ("FLOAT", {
                    "default": 50,
                    "min": 1.0,
                    "max": 1000.0,
                    "step": 10.0
                }),
                "max_windows": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 500,
                    "step": 1
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "text_size": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "preserve_background": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_cyberpunk_effect"
    CATEGORY = "SyntaxNodes/Processing"
    def process_single_image(self, image_tensor, edge_threshold1, edge_threshold2, 
                           min_window_size, max_windows, line_thickness, 
                           glow_intensity, text_size, preserve_background, custom_text):
        # Convert from tensor to PIL
        pil_image = self.t2p(image_tensor)
        frame = np.array(pil_image)
        
        # Create background based on preserve_background flag
        if preserve_background:
            result = frame.copy()
        else:
            result = np.zeros_like(frame)
        
        # Edge detection for finding potential windows
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
        
        # Find contours for windows
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and keep only the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_windows]
        
        # Store window centers for connecting lines
        window_centers = []
        
        # Process each window
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_window_size:
                continue
                
            # Create window mask
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            window_centers.append(center)
            
            if not preserve_background:
                # Copy original image content for this window
                mask = np.zeros_like(frame)
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                window_content = cv2.bitwise_and(frame, mask)
                
                # Add window content to result
                result = cv2.add(result, window_content)
            
            # Add glowing border effect
            glow_color = (
                random.randint(150, 255),  # R
                random.randint(150, 255),  # G
                random.randint(150, 255)   # B
            )
            
            # Multiple borders with decreasing intensity for glow effect
            for i in range(3):
                thickness = line_thickness + i*2
                alpha = glow_intensity * (1 - i*0.2)
                glow = np.zeros_like(frame)
                cv2.rectangle(glow, (x-i*2, y-i*2), (x + w+i*2, y + h+i*2), 
                            glow_color, thickness)
                result = cv2.addWeighted(result, 1, glow, alpha, 0)
            
            # Add measurement text with custom prefix
            label = f"{custom_text}: {area:.0f}px"
            font_scale = text_size
            font_thickness = max(1, int(line_thickness * 0.8))
            
            # Get text size for better positioning
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Position text inside the window near the top
            text_x = x + 5
            text_y = y + text_height + 5
            
            # Add black background for text readability
            cv2.rectangle(result, 
                         (text_x - 2, text_y - text_height - 2),
                         (text_x + text_width + 2, text_y + 2),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 255, 0), font_thickness)
        
        # Connect windows with RGB lines
        if len(window_centers) > 1:
            for i in range(len(window_centers)-1):
                start = window_centers[i]
                end = window_centers[i+1]
                
                # Create RGB offset lines
                offset = 2
                cv2.line(result, 
                        (start[0]-offset, start[1]), 
                        (end[0]-offset, end[1]), 
                        (255, 0, 0), 
                        1)  # Red
                cv2.line(result, 
                        start, 
                        end, 
                        (0, 255, 0), 
                        1)  # Green
                cv2.line(result, 
                        (start[0]+offset, start[1]), 
                        (end[0]+offset, end[1]), 
                        (0, 0, 255), 
                        1)  # Blue
        
        # Convert back to tensor
        output_image = Image.fromarray(result)
        return self.p2t(output_image)

    def create_cyberpunk_effect(self, image, custom_text, edge_threshold1, edge_threshold2, 
                              min_window_size, max_windows, line_thickness, 
                              glow_intensity, text_size, preserve_background):
        # Get batch size
        batch_size = image.shape[0]
        
        # Initialize progress bar
        pbar = ProgressBar(batch_size)
        
        # Process each image in the batch
        processed_images = []
        for i in range(batch_size):
            # Process single image
            processed = self.process_single_image(
                image[i:i+1],
                edge_threshold1,
                edge_threshold2,
                min_window_size,
                max_windows,
                line_thickness,
                glow_intensity,
                text_size,
                preserve_background,
                custom_text
            )
            processed_images.append(processed)
            pbar.update_absolute(i + 1)
        
        # Concatenate results and return
        return (torch.cat(processed_images, dim=0),)

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "CyberpunkWindowNode": CyberpunkWindowNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberpunkWindowNode": "Cyberpunk Window Effect"
}