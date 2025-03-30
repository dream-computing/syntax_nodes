import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
import torch
from comfy.utils import ProgressBar

class LowPolyNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_points": ("INT", {
                    "default": 100, 
                    "min": 20, 
                    "max": 5000,
                    "step": 1
                }),
                "num_points_step": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 100,
                    "step": 1
                }),
                "edge_points": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "edge_points_step": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 20,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"

    def process_image(self, image, num_points, num_points_step, edge_points, edge_points_step):
        # Convert from ComfyUI image format to numpy array using t2p method
        pil_image = self.t2p(image)

        # Ensure the image is in RGB mode
        pil_image = pil_image.convert('RGB')

        # Process the image using the step values
        processed_image = self.create_low_poly(pil_image, num_points, edge_points)

        # Convert processed PIL image back to tensor using p2t method
        processed_tensor = self.p2t(processed_image)

        return (processed_tensor,)  # Ensure a single image is returned

    def create_low_poly(self, image, num_points, edge_points):
        width, height = image.size

        # Generate random points for Delaunay triangulation
        points = np.random.rand(num_points, 2)
        points = points * [width, height]

        # Add corners to points
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]])
        points = np.vstack([points, corners])

        # Add edge points
        for _ in range(edge_points):
            points = np.vstack([points, [0, np.random.rand() * height]])
            points = np.vstack([points, [width, np.random.rand() * height]])
            points = np.vstack([points, [np.random.rand() * width, 0]])
            points = np.vstack([points, [np.random.rand() * width, height]])

        # Perform Delaunay triangulation
        tri = Delaunay(points)

        # Create an empty image to draw the triangles on
        result = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(result)

        # Draw each triangle with the sampled color from the original image
        for triangle in tri.simplices:
            coords = [(points[vertex][0], points[vertex][1]) for vertex in triangle]
            center_x = sum(coord[0] for coord in coords) / 3
            center_y = sum(coord[1] for coord in coords) / 3
            color = image.getpixel((int(center_x), int(center_y)))  # Sample color
            draw.polygon(coords, fill=color)

        return result

    def t2p(self, t):
        if t is not None:
            # Move tensor to CPU first if it's on GPU for PIL conversion
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p

    def p2t(self, p):
        if p is not None:
            # Convert PIL image to tensor and move to GPU if available
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0).to(self.device)  # Move to GPU here
        return t

NODE_CLASS_MAPPINGS = {
    "LowPolyNode": LowPolyNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LowPolyNode": "Low Poly Image Processor"
}
