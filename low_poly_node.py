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
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "SyntaxNodes/Processing"

    def process_image(self, image, num_points, num_points_step, edge_points, edge_points_step, mask=None):
        # Check if the input is a batch (has more than 1 in first dimension)
        if image.shape[0] > 1:
            # Convert batch to list
            image_list_converter = ImageBatchToImageList()
            image_list, = image_list_converter.doit(image)
            
            # Create a progress bar for batch processing
            progress = ProgressBar(len(image_list))
            
            # Process each image in the list
            processed_list = []
            for i, single_image in enumerate(image_list):
                # Extract mask for this image if provided
                single_mask = None
                if mask is not None:
                    # Handle both batch and single masks
                    if mask.shape[0] > 1:
                        mask_list, = image_list_converter.doit(mask)
                        single_mask = mask_list[i]
                    else:
                        single_mask = mask
                
                # Process the single image
                output, = self._process_single_image(single_image, num_points, num_points_step, edge_points, edge_points_step, single_mask)
                processed_list.append(output)
                
                # Update progress
                progress.update(1)
            
            # Convert list back to batch
            image_batch_converter = ImageListToImageBatch()
            output_batch, = image_batch_converter.doit(processed_list)
            
            return (output_batch,)
        else:
            # Process a single image
            return self._process_single_image(image, num_points, num_points_step, edge_points, edge_points_step, mask)

    def _process_single_image(self, image, num_points, num_points_step, edge_points, edge_points_step, mask=None):
        # Convert from tensor to PIL image
        pil_image = self.t2p(image)
        
        # Ensure the image is in RGB mode
        pil_image = pil_image.convert('RGB')
        
        # Store original image for masking if needed
        if mask is not None:
            original_array = np.array(pil_image)
        
        # Process the image using low poly effect
        processed_image = self.create_low_poly(pil_image, num_points, edge_points)
        
        # Apply mask if provided
        if mask is not None:
            mask_array = self.t2p(mask)
            if mask_array.mode != 'L':  # Convert to grayscale if not already
                mask_array = mask_array.convert('L')
            mask_np = np.array(mask_array, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            mask_np = mask_np[..., np.newaxis]  # Add channel dimension
            
            # Blend original and processed based on mask
            processed_array = np.array(processed_image)
            masked_array = original_array * (1 - mask_np) + processed_array * mask_np
            processed_image = Image.fromarray(np.clip(masked_array, 0, 255).astype(np.uint8))
        
        # Convert back to tensor
        processed_tensor = self.p2t(processed_image)
        
        return (processed_tensor,)

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
            
            # Ensure the center point is within image bounds
            center_x = min(max(0, int(center_x)), width-1)
            center_y = min(max(0, int(center_y)), height-1)
            
            color = image.getpixel((center_x, center_y))  # Sample color
            draw.polygon(coords, fill=color)

        return result

    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return p
        return None

    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
            return t
        return None


class ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                  }
            }
    
    INPUT_IS_LIST = True
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"
    
    CATEGORY = "SyntaxNodes/Conversion"
    
    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    import comfy.utils
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)


class ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"
    
    CATEGORY = "SyntaxNodes/Conversion"
    
    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )


NODE_CLASS_MAPPINGS = {
    "LowPolyNode": LowPolyNode,
    "ImageListToImageBatch": ImageListToImageBatch,
    "ImageBatchToImageList": ImageBatchToImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LowPolyNode": "Low Poly Image Processor",
    "ImageListToImageBatch": "Image List to Batch",
    "ImageBatchToImageList": "Image Batch to List"
}
