import numpy as np
import cv2
from PIL import Image
import torch
from comfy.utils import ProgressBar

class GhostingNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_buffer = []  # Buffer to store recent frames
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "decay_rate": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 5.0,
                    "step": 0.01
                }),
                "blend_opacity": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0,
                    "step": 0.01
                }),
                "buffer_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
                })
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "SyntaxNodes/Processing"
    
    def process_image(self, image, decay_rate, blend_opacity, buffer_size, mask=None):
        # Check if the input is a batch (has more than 1 in first dimension)
        if image.shape[0] > 1:
            # Convert batch to list using ImageBatchToImageList
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
                
                # Process the single image using the original method
                output, = self._process_single_image(single_image, decay_rate, blend_opacity, buffer_size, single_mask)
                processed_list.append(output)
                
                # Update progress
                progress.update(1)
            
            # Convert list back to batch using ImageListToImageBatch
            image_batch_converter = ImageListToImageBatch()
            output_batch, = image_batch_converter.doit(processed_list)
            
            return (output_batch,)
        else:
            # Process a single image using the original method
            return self._process_single_image(image, decay_rate, blend_opacity, buffer_size, mask)
    
    def _process_single_image(self, image, decay_rate, blend_opacity, buffer_size, mask=None):
        # Convert from ComfyUI image format to numpy array
        pil_image = self.t2p(image)
        frame = np.array(pil_image, dtype=np.float32)
        
        # Handle optional mask input
        if mask is not None:
            mask_pil = self.t2p(mask)
            if mask_pil.mode != 'L':  # Convert to grayscale if not already
                mask_pil = mask_pil.convert('L')
            mask_np = np.array(mask_pil, dtype=np.float32) / 255.0  # Normalize mask to [0, 1]
        else:
            mask_np = np.ones_like(frame[..., 0], dtype=np.float32)  # Default to full mask
        
        if self.frame_buffer:
            buffer_shape = self.frame_buffer[0].shape
            if frame.shape != buffer_shape:
                frame = cv2.resize(frame, (buffer_shape[1], buffer_shape[0]))
                if frame.shape[-1] != buffer_shape[-1]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if buffer_shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if len(self.frame_buffer) >= buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)
        
        ghost_frame = frame.copy()
        for i, previous_frame in enumerate(reversed(self.frame_buffer)):
            weight = blend_opacity * (decay_rate ** i)
            blended = cv2.addWeighted(ghost_frame, 1 - weight, previous_frame, weight, 0)
            ghost_frame = ghost_frame * (1 - mask_np[..., None]) + blended * mask_np[..., None]  # Apply mask
        
        ghost_frame = np.clip(ghost_frame, 0, 255).astype(np.uint8)
        ghost_pil_image = Image.fromarray(ghost_frame)
        output_tensor = self.p2t(ghost_pil_image)
        return (output_tensor,)
    
    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            p = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return p
    
    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            t = torch.from_numpy(i).unsqueeze(0)
        return t

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
    "GhostingNode": GhostingNode,
    "ImageListToImageBatch": ImageListToImageBatch,
    "ImageBatchToImageList": ImageBatchToImageList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GhostingNode": "Ghosting/Afterimage Effect",
    "ImageListToImageBatch": "Image List to Batch",
    "ImageBatchToImageList": "Image Batch to List"
}