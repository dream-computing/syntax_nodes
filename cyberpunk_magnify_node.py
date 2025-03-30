import numpy as np
import cv2
from PIL import Image
import torch
import random
import math
from comfy.utils import ProgressBar

class CyberpunkMagnifyNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
                "magnification": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "detail_size": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 500,
                    "step": 10
                }),
                "num_details": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "line_color": (["YELLOW", "RGB", "CYAN", "WHITE"], {"default": "YELLOW"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_magnified_detail_view"
    CATEGORY = "image/effects"

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)

    def process_single_image(self, image_tensor, edge_threshold1, edge_threshold2, 
                           magnification, detail_size, num_details, line_thickness, line_color):
        # Convert from tensor to PIL
        pil_image = self.t2p(image_tensor)
        frame = np.array(pil_image)
        
        # Create canvas that matches input dimensions
        result = np.copy(frame)
        
        # Calculate padding needed for detail views
        padding = detail_size * 2
        extended_height = frame.shape[0] + padding
        extended_width = frame.shape[1] + padding
        extended_result = np.zeros((extended_height, extended_width, 3), dtype=np.uint8)
        
        # Place original image without scaling
        y_offset = padding // 2
        x_offset = padding // 2
        extended_result[y_offset:y_offset+frame.shape[0], 
                      x_offset:x_offset+frame.shape[1]] = result
        
        # Find interesting regions
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours for centers
        centers = []
        if len(contours) > 0:
            complexities = [(i, len(c) * cv2.contourArea(c)) for i, c in enumerate(contours)]
            complexities.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _ in complexities[:min(len(complexities), num_details*2)]:
                M = cv2.moments(contours[idx])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
        
        # If not enough centers, add some default ones
        if len(centers) < num_details:
            height, width = frame.shape[:2]
            default_centers = [
                (width // 4, height // 4),
                (3 * width // 4, height // 4),
                (width // 4, 3 * height // 4),
                (3 * width // 4, 3 * height // 4)
            ]
            for center in default_centers:
                if len(centers) < num_details and center not in centers:
                    centers.append(center)
        
        centers = centers[:num_details]  # Limit to requested number
        
        # Collect obstacles and detail points
        obstacles = []
        detail_points = []
        source_point = (x_offset + centers[0][0], y_offset + centers[0][1])
        
        # Place detail views
        for i, center in enumerate(centers):
            # Extract region
            half_size = detail_size // 2
            x1 = max(0, min(frame.shape[1] - detail_size, center[0] - half_size))
            y1 = max(0, min(frame.shape[0] - detail_size, center[1] - half_size))
            x2 = min(frame.shape[1], x1 + detail_size)
            y2 = min(frame.shape[0], y1 + detail_size)
            
            region = frame[y1:y2, x1:x2]
            
            # Skip if region is empty
            if region.size == 0:
                continue
                
            # Magnify the region
            magnified = cv2.resize(region, None, fx=magnification, fy=magnification)
            
            # Add white border
            border_size = 2
            magnified = cv2.copyMakeBorder(
                magnified,
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            
            # Calculate position for detail view
            margin = 40
            if i == 0:  # First detail view goes on left
                detail_x = margin
                detail_y = extended_result.shape[0] // 2 - magnified.shape[0] // 2
            elif i == len(centers) - 1:  # Last one goes on right
                detail_x = extended_result.shape[1] - magnified.shape[1] - margin
                detail_y = extended_result.shape[0] // 2 - magnified.shape[0] // 2
            else:  # Others alternate top and bottom
                if i % 2 == 0:
                    detail_x = margin + (extended_result.shape[1] - 2*margin) * (i / (len(centers)-1))
                    detail_y = margin
                else:
                    detail_x = margin + (extended_result.shape[1] - 2*margin) * (i / (len(centers)-1))
                    detail_y = extended_result.shape[0] - magnified.shape[0] - margin
            
            detail_x = int(detail_x)
            detail_y = int(detail_y)
            
            # Ensure coordinates are within bounds
            detail_x = max(0, min(detail_x, extended_result.shape[1] - magnified.shape[1]))
            detail_y = max(0, min(detail_y, extended_result.shape[0] - magnified.shape[0]))
            
            # Add to obstacles list
            obstacles.append((detail_x, detail_y, magnified.shape[1], magnified.shape[0]))
            
            # Place detail view
            try:
                extended_result[detail_y:detail_y+magnified.shape[0], 
                              detail_x:detail_x+magnified.shape[1]] = magnified
                # Add detail point
                detail_points.append((detail_x + magnified.shape[1]//2, 
                                   detail_y + magnified.shape[0]//2))
            except Exception as e:
                print(f"Failed to place detail view: {e}")
                continue
        
        # Draw connections
        color_map = {
            "YELLOW": (255, 255, 0),
            "WHITE": (255, 255, 255),
            "CYAN": (0, 255, 255),
            "RGB": None
        }
        
        # Connect each detail view
        for i, point in enumerate(detail_points):
            if line_color == "RGB":
                # Simple path for RGB connections
                for j in range(1):
                    offset = 2
                    cv2.line(extended_result, 
                            (source_point[0]-offset, source_point[1]), 
                            (point[0]-offset, point[1]), 
                            (255, 0, 0), line_thickness)
                    cv2.line(extended_result, 
                            source_point, point, 
                            (0, 255, 0), line_thickness)
                    cv2.line(extended_result, 
                            (source_point[0]+offset, source_point[1]), 
                            (point[0]+offset, point[1]), 
                            (0, 0, 255), line_thickness)
            else:
                # Simple connection for solid colors
                cv2.line(extended_result, source_point, point, color_map[line_color], line_thickness)
            
            if i < len(detail_points) - 1:
                next_point = detail_points[i + 1]
                if line_color == "RGB":
                    for j in range(1):
                        offset = 1
                        cv2.line(extended_result, 
                                (point[0]-offset, point[1]), 
                                (next_point[0]-offset, next_point[1]), 
                                (100, 0, 0), line_thickness)
                        cv2.line(extended_result, 
                                point, next_point, 
                                (0, 100, 0), line_thickness)
                        cv2.line(extended_result, 
                                (point[0]+offset, point[1]), 
                                (next_point[0]+offset, next_point[1]), 
                                (0, 0, 100), line_thickness)
                else:
                    secondary_color = tuple(int(c * 0.7) for c in color_map[line_color])
                    cv2.line(extended_result, point, next_point, secondary_color, line_thickness)
        
        # Convert back to tensor
        output_image = Image.fromarray(extended_result)
        return self.p2t(output_image)

    def create_magnified_detail_view(self, image, edge_threshold1, edge_threshold2, 
                                   magnification, detail_size, num_details, 
                                   line_thickness, line_color):
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
                magnification,
                detail_size,
                num_details,
                line_thickness,
                line_color
            )
            processed_images.append(processed)
            pbar.update_absolute(i + 1)
        
        # Concatenate results and return
        return (torch.cat(processed_images, dim=0),)