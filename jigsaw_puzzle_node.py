import numpy as np
import torch
from PIL import Image
import cv2

# Use relative imports to import from the current directory
from .puzzle_creator import create as create_puzzle
from .effects_handler import apply_relief_and_shadow, add_background
from .transformations_handler import transform_v1

class JigsawPuzzleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pieces": ("INT", {"default": 50, "min": 10, "max": 500, "step": 10}),
                "piece_size": ("INT", {"default": 64, "min": 32, "max": 100, "step": 1}),
                "background": ("IMAGE", {"optional": True}),
                "num_remove": ("INT", {"default": 3, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_jigsaw_effect"
    CATEGORY = "🖼️ Image/Effects"

    def apply_jigsaw_effect(self, image, pieces, piece_size, num_remove, background=None):
        # Convert the input image tensor to a numpy array (OpenCV-compatible)
        image_np = self.t2p(image)
        
        # Handle background image
        if background is not None:
            background_np = self.t2p(background)
        else:
            # If no background is provided, create a white background
            background_np = np.full(image_np.shape, 255, dtype=np.uint8)

        # Create the puzzle image and puzzle mask
        puzzle_image, puzzle_mask = create_puzzle(image_np, piece_size)

        # Transform the puzzle pieces
        puzzle_image, puzzle_mask, foreground_mask = transform_v1(
            puzzle_image, puzzle_mask, piece_size, background_np.shape, num_remove, select_pieces=False
        )

        # Add the background to the puzzle image
        puzzle_image = add_background(background_np, puzzle_image, foreground_mask)

        # Apply relief and shadow effects to the puzzle image
        puzzle_image, puzzle_mask = apply_relief_and_shadow(puzzle_image, puzzle_mask)

        # Convert the output puzzle image back to a tensor
        return (self.p2t(puzzle_image),)

    def t2p(self, t):
        """Converts a ComfyUI tensor to a NumPy array (for OpenCV)."""
        if t is not None:
            return (t.cpu().numpy().squeeze() * 255).astype(np.uint8)

    def p2t(self, p):
        """Converts a NumPy array (from OpenCV) back to a ComfyUI tensor."""
        if p is not None:
            return torch.from_numpy(p.astype(np.float32) / 255.0).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "JigsawPuzzleNode": JigsawPuzzleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawPuzzleNode": "Jigsaw Puzzle Effect"
}
