import numpy as np
import cv2
from PIL import Image
import torch

class EdgeMeasurementOverlayNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "canny_threshold1": ("FLOAT", {
                    "default": 50,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "canny_threshold2": ("FLOAT", {
                    "default": 150,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 1.0
                }),
                "min_area": ("FLOAT", {
                    "default": 100,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 1.0
                }),
                "bounding_box_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"

    def process_image(self, image, canny_threshold1, canny_threshold2, min_area, bounding_box_opacity):
        # Convert from ComfyUI image format to numpy array
        pil_image = self.t2p(image)
        frame = np.array(pil_image, dtype=np.uint8)

        # Scale up for better resolution during processing
        original_size = frame.shape[:2]
        upscale_factor = 2
        frame = cv2.resize(frame, (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor))

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, int(canny_threshold1), int(canny_threshold2))

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create overlay
        overlay = np.zeros_like(frame)

        for contour in contours:
            # Filter contours by area
            area = cv2.contourArea(contour)
            if area < min_area * (upscale_factor ** 2):  # Scale threshold by the upscale factor
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Draw styled bounding boxes
            box_color = (255, 0, 0)  # Red bounding box
            box_thickness = 2
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, box_thickness)

            # Add transparency effect
            cv2.addWeighted(overlay, bounding_box_opacity, frame, 1 - bounding_box_opacity, 0, frame)

            # Add labels inside the box
            label = f"Area: {area:.2f}"
            cv2.putText(frame, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Scale back down to original size
        frame = cv2.resize(frame, (original_size[1], original_size[0]))

        # Convert back to PIL image
        output_image = Image.fromarray(frame)

        # Convert processed PIL image back to tensor
        return (self.p2t(output_image),)

    def t2p(self, t):
        if t is not None:
            # Convert tensor to PIL image
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        if p is not None:
            # Convert PIL image to tensor and normalize
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "EdgeMeasurementOverlayNode": EdgeMeasurementOverlayNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeMeasurementOverlayNode": "Edge Measurement Overlay"
}
