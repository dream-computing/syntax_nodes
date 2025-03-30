import cv2
import numpy as np
import torch
from PIL import Image
from comfy.utils import ProgressBar
import random


class LuminanceParticleNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_frame = None
        self.particles = []  # List to store particles

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "num_layers": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
                "smoothing_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "particle_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "particle_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "num_particles": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "particle_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "edge_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "particle_lifespan": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_depth_map"

    def process_depth_map(self, depth_map, num_layers, smoothing_factor, particle_size, particle_speed, num_particles, particle_opacity, edge_opacity, particle_lifespan):
        # Convert depth map tensor to a PIL image and then to a NumPy array
        depth_image = self.t2p(depth_map)
        depth_array = np.array(depth_image)

        # Normalize depth array
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Compute gradients for directional flow and transfer them to the GPU
        grad_x = cv2.Sobel(depth_normalized, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_normalized, cv2.CV_32F, 0, 1, ksize=5)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        angle = torch.from_numpy(angle).to(self.device, dtype=torch.float32)

        # Detect edges
        edges = cv2.Canny(depth_normalized, 50, 150)

        # Transfer edge image to the GPU
        edge_layer = torch.zeros((depth_normalized.shape[0], depth_normalized.shape[1], 3), device=self.device, dtype=torch.float32)
        edge_layer[torch.from_numpy(edges > 0).to(self.device, dtype=torch.bool)] = torch.tensor([0.0, 255.0 * edge_opacity, 0.0], device=self.device, dtype=torch.float32)

        # Generate new particles at random edge points
        num_new_particles = max(0, num_particles - len(self.particles))  # Ensure non-negative count
        new_particles = self.create_particles(edges, angle, num_new_particles, particle_lifespan)

        # Add the new particles to the particle list
        self.particles.extend(new_particles)

        # Create particle layer
        particle_layer = torch.zeros_like(edge_layer)

        # Update particles and draw them onto particle_layer
        updated_particles = []
        for particle in self.particles:
            x, y, dx, dy, lifespan = particle

            # Move particle and clamp within bounds
            x = max(0, min(depth_normalized.shape[1] - 1, x + dx * particle_speed))
            y = max(0, min(depth_normalized.shape[0] - 1, y + dy * particle_speed))
            lifespan -= 1  # Decrease lifespan

            # Check boundaries and add to the updated list if within bounds and still alive
            if 0 <= x < depth_normalized.shape[1] and 0 <= y < depth_normalized.shape[0] and lifespan > 0:
                updated_particles.append((x, y, dx, dy, lifespan))

                # Draw particle as a small circle on the GPU within bounds
                x_int, y_int = int(x), int(y)
                x_end = min(x_int + particle_size, particle_layer.shape[1])  # Ensure x-end is within bounds
                y_end = min(y_int + particle_size, particle_layer.shape[0])  # Ensure y-end is within bounds

                particle_layer[y_int:y_end, x_int:x_end] = torch.tensor([0.0, 255.0, 0.0], device=self.device, dtype=torch.float32)

        # Update particles list
        self.particles = updated_particles

        # Combine edge and particle layers with adjustable opacity
        combined_output = (edge_layer * edge_opacity + particle_layer * particle_opacity).clamp(0, 255).byte()

        # Convert processed image back to tensor format and move to CPU for output
        output_image = Image.fromarray(combined_output.cpu().numpy())
        output_tensor = self.p2t(output_image)

        return (output_tensor,)

    def create_particles(self, edges, angle, num_new_particles, lifespan):
        """Create new particles at edge points with initial directions and lifespan."""
        particles = []
        edge_points = np.argwhere(edges > 0)

        # Ensure we don't sample more than available edge points
        if len(edge_points) > 0:
            sampled_points = random.sample(list(edge_points), min(len(edge_points), num_new_particles))

            for (y, x) in sampled_points:
                # Get the angle for direction from gradient, making sure to extract a scalar
                angle_value = angle[y, x].item() if angle[y, x].numel() == 1 else angle[y, x].mean().item()
                dx = np.cos(np.deg2rad(angle_value))
                dy = np.sin(np.deg2rad(angle_value))
                particles.append((float(x), float(y), float(dx), float(dy), lifespan))

        return particles

    def reset(self):
        """Reset the previous frame memory and particles for a new sequence."""
        self.prev_frame = None
        self.particles = []  # Reset particles for each new sequence

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
    "LuminanceParticleNode": LuminanceParticleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LuminanceParticleNode": "Luminance Particles"
}
