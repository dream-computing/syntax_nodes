import numpy as np
import torch
from PIL import Image
import cv2


class EdgeTracingNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 50, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 150, "min": 0, "max": 255}),
                "num_particles": ("INT", {"default": 1000, "min": 1, "max": 50000}),
                "speed": ("INT", {"default": 10, "min": 1, "max": 100}),
                "edge_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "particle_size": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_particle_tracing"

    def generate_particle_tracing(
        self, input_image, low_threshold, high_threshold, num_particles, speed, edge_opacity, particle_size
    ):
        # Step 1: Convert input tensor to grayscale numpy image
        pil_image = self.t2p(input_image)
        np_image = np.array(pil_image.convert("L"))

        # Step 2: Apply Canny edge detection using OpenCV
        edges = cv2.Canny(np_image, threshold1=low_threshold, threshold2=high_threshold)
        edge_coords = np.column_stack(np.where(edges > 0))

        if edge_coords.shape[0] == 0:
            raise ValueError("No edges detected. Adjust thresholds.")

        # Step 3: Initialize particles at random edge coordinates
        particle_positions = edge_coords[
            np.random.choice(edge_coords.shape[0], num_particles, replace=True)
        ]

        # Step 4: Precompute 8-neighbor offsets
        neighbor_offsets = np.array(
            [
                [-1, -1], [-1, 0], [-1, 1],
                [0, -1],          [0, 1],
                [1, -1], [1, 0], [1, 1]
            ]
        )

        # Step 5: Create separate canvases for edges and particles
        edge_canvas = edges.astype(np.float32) * edge_opacity
        particle_canvas = np.zeros_like(edges, dtype=np.float32)

        for _ in range(speed):  # Run particle tracing for the specified number of steps
            # Calculate neighbors
            neighbors = particle_positions[:, None, :] + neighbor_offsets[None, :, :]
            valid_mask = (
                (neighbors[:, :, 0] >= 0) & (neighbors[:, :, 1] >= 0) &
                (neighbors[:, :, 0] < edges.shape[0]) & (neighbors[:, :, 1] < edges.shape[1])
            )
            neighbors = neighbors[valid_mask]

            # Keep only neighbors that are edge pixels
            valid_neighbors = neighbors[edges[neighbors[:, 0], neighbors[:, 1]] > 0]

            if valid_neighbors.shape[0] > 0:
                particle_positions = valid_neighbors[
                    np.random.choice(valid_neighbors.shape[0], num_particles, replace=True)
                ]

            # Draw particles onto the particle canvas
            for y, x in particle_positions:
                cv2.circle(particle_canvas, (x, y), particle_size, 1, thickness=-1)

        # Normalize the particle canvas for visibility
        particle_canvas = (particle_canvas / particle_canvas.max() * 255).clip(0, 255).astype(np.uint8)

        # Step 6: Combine edge and particle layers
        combined_canvas = np.stack(
            [particle_canvas, edge_canvas.astype(np.uint8), np.zeros_like(edge_canvas)], axis=-1
        ).astype(np.uint8)

        # Step 7: Return combined canvas as tensor
        combined_image = Image.fromarray(combined_canvas, "RGB")
        return (self.p2t(combined_image),)

    def t2p(self, tensor):
        """Convert a tensor to a PIL image."""
        if tensor is not None:
            i = 255.0 * tensor.cpu().numpy().squeeze()
            if len(i.shape) == 3 and i.shape[0] == 1:  # Handle extra channel dimension
                i = i[0]
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, pil_image):
        """Convert a PIL image to a tensor."""
        if pil_image is not None:
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            return torch.from_numpy(np_image).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "EdgeTracingNode": EdgeTracingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeTracingNode": "Edge Tracing Node"
}
