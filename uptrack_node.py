import numpy as np
import cv2
from PIL import Image
import torch
from mediapipe.python.solutions import pose as mp_pose
from comfy.utils import ProgressBar

class UpTrackNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Motion tracking state
        self.current_angle = 0
        self.velocity = 0
        self.acceleration = 0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Tracking mode
                "tracking_mode": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "combo",
                    "labels": ["Upright Lock", "Follow Flip"]
                }),
                # Track points switches
                "use_shoulders": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1,
                    "step": 1
                }),
                "use_hips": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1
                }),
                "use_knees": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1
                }),
                # Physics parameters
                "inertia": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01
                }),
                "snap_force": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "damping": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                # Detection settings
                "confidence": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "angle_threshold": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 45.0,
                    "step": 1.0
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_tracking"
    CATEGORY = "SyntaxNodes/Processing"

    def get_orientation_from_points(self, landmarks, use_shoulders, use_hips, use_knees, tracking_mode):
        angles = []
        weights = []
        
        if use_shoulders:
            # Get shoulder angle
            left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
            shoulder_angle = np.degrees(np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ))
            angles.append(shoulder_angle)
            weights.append(1.0)  # Shoulders get full weight
            
        if use_hips:
            # Get hip angle
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
            hip_angle = np.degrees(np.arctan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            ))
            angles.append(hip_angle)
            weights.append(0.8)  # Hips slightly less weight than shoulders
            
        if use_knees:
            # Get knee angle
            left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y])
            right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y])
            knee_angle = np.degrees(np.arctan2(
                right_knee[1] - left_knee[1],
                right_knee[0] - left_knee[0]
            ))
            angles.append(knee_angle)
            weights.append(0.6)  # Knees get least weight
            
        if not angles:  # If no points selected
            return None
            
        # Weighted average of angles
        weights = np.array(weights) / sum(weights)
        final_angle = np.average(angles, weights=weights)
        
        if tracking_mode == 0:  # Upright Lock mode
            return final_angle + 90  # Adjust to make 0 degrees = upright
        else:  # Follow Flip mode
            return final_angle  # Don't adjust, follow actual orientation

    def normalize_angle(self, angle):
        """Normalize angle to -180 to 180 range"""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def rotate_image(self, image, angle, center=None):
        height, width = image.shape[:2]
        if center is None:
            center = (width // 2, height // 2)
            
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                flags=cv2.INTER_LINEAR)
        return rotated

    def process_single_image(self, image_tensor, tracking_mode, use_shoulders, use_hips, use_knees,
                           inertia, snap_force, damping, confidence, angle_threshold):
        # Convert tensor to PIL
        pil_image = self.t2p(image_tensor)
        image = np.array(pil_image)
        
        # Update pose detection confidence
        self.pose.min_detection_confidence = confidence
        
        # Process image with MediaPipe
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if results.pose_landmarks:
            target_angle = self.get_orientation_from_points(
                results.pose_landmarks.landmark,
                use_shoulders,
                use_hips,
                use_knees,
                tracking_mode
            )
            
            if target_angle is not None:
                # Normalize angles
                target_angle = self.normalize_angle(target_angle)
                current = self.normalize_angle(self.current_angle)
                
                # Only respond to significant changes
                angle_diff = self.normalize_angle(target_angle - current)
                if abs(angle_diff) > angle_threshold:
                    # Calculate forces
                    snap = angle_diff * snap_force
                    drag = -self.velocity * damping
                    
                    # Update motion
                    self.acceleration = snap + drag
                    self.velocity = (self.velocity + self.acceleration) * inertia
                    self.current_angle += self.velocity
        
        # Rotate image
        rotated_image = self.rotate_image(image, -self.current_angle)
        output_image = Image.fromarray(rotated_image)
            
        # Convert back to tensor
        return self.p2t(output_image)

    def process_tracking(self, image, tracking_mode, use_shoulders, use_hips, use_knees,
                        inertia, snap_force, damping, confidence, angle_threshold):
        batch_size = image.shape[0]
        
        # Initialize progress bar
        pbar = ProgressBar(batch_size)
        
        # Process each image in the batch
        processed_images = []
        for i in range(batch_size):
            processed = self.process_single_image(
                image[i:i+1],
                tracking_mode,
                use_shoulders,
                use_hips,
                use_knees,
                inertia,
                snap_force,
                damping,
                confidence,
                angle_threshold
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
    "UpTrackNode": UpTrackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpTrackNode": "Upright Person Tracking"
}