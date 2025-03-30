import os
import cv2
import numpy as np
import torch
import random
import librosa
import subprocess
import datetime
import json
from PIL import Image
# Fix the PngInfo import issue
try:
    from PIL.PngImagePlugin import PngInfo
except ImportError:
    # Create a fallback PngInfo class if it can't be imported
    class PngInfo:
        def __init__(self):
            self.text = {}
        def add_text(self, key, value):
            self.text[key] = value

from comfy.utils import ProgressBar
import folder_paths

class FrequencyBeatSyncNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {"default": "videos"}),
                "audio_path": ("STRING", {"default": "audio.mp3"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "max_beats": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "max_frames": ("INT", {"default": 150, "min": 0, "max": 1000, "step": 1}),
                "effect_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "output_mode": (["Frames for Editing", "Direct Video Output"],),
                "filename_prefix": ("STRING", {"default": "BeatSync"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "process_video"
    CATEGORY = "Video Processing"
    
    def t2p(self, t):
        """Convert tensor to PIL Image"""
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def p2t(self, p):
        """Convert PIL Image to tensor"""
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)
    
    def get_audio_data(self, audio_path):
        """Load audio file and return in the format expected by VHS"""
        try:
            # Use librosa to load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Convert to PyTorch tensor with shape [1, channels, samples]
            if len(y.shape) == 1:  # Mono
                waveform = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            else:  # Already has channels
                waveform = torch.from_numpy(y).unsqueeze(0)  # [1, channels, samples]
            
            # Create the audio dictionary in the format VHS expects
            audio_data = {
                'waveform': waveform,
                'sample_rate': sr,
                'path': audio_path
            }
            
            return audio_data
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Return a minimal valid audio dict as fallback
            return {
                'waveform': torch.zeros((1, 2, 44100)),  # 1 second of silence, stereo
                'sample_rate': 44100,
                'path': audio_path
            }
    
    def detect_frequency_beats(self, audio_path):
        """Detect beats in different frequency ranges of the audio file using optimized parameters"""
        print("Analyzing audio frequencies...")
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        try:
            # Split into harmonic (pitched) and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Try with optimized parameters for frequency separation
            try:
                # Use different approach based on librosa version
                lib_version = librosa.__version__
                
                if int(lib_version.split('.')[0]) >= 1:  # Version 1.0 or higher
                    # For newer librosa versions that might handle preemphasis differently
                    # Use filter bank approach
                    from scipy import signal
                    
                    # Design a low-pass filter for bass frequencies (below 150Hz)
                    sos_low = signal.butter(10, 150, 'lowpass', fs=sr, output='sos')
                    y_low = signal.sosfilt(sos_low, y_percussive)
                    
                    # High frequencies are the remainder
                    y_high = y_percussive - y_low
                    print("Using optimized filter bank for frequency separation")
                else:
                    # Use original approach from the script with fixed parameters
                    try:
                        # Try with return_zeropad parameter (older versions)
                        y_low = librosa.effects.preemphasis(y_percussive, coef=0.95, return_zeropad=False)
                        # High frequencies are the remainder
                        y_high = y_percussive - y_low
                        print("Using original preemphasis method for frequency separation")
                    except:
                        # Fall back to simple preemphasis
                        y_low = librosa.effects.preemphasis(y_percussive, coef=0.95)
                        y_high = y_percussive - y_low
                        print("Using preemphasis without return_zeropad")
            except Exception as e:
                print(f"Frequency separation failed: {str(e)}")
                # More robust fallback using FFT
                y_fft = np.fft.rfft(y_percussive)
                # Calculate cutoff frequency (around 150Hz)
                cutoff = int(len(y_fft) * 150 / (sr/2))
                
                # Create low and high frequency versions
                y_fft_low = y_fft.copy()
                y_fft_low[cutoff:] = 0
                
                y_fft_high = y_fft.copy()
                y_fft_high[:cutoff] = 0
                
                y_low = np.fft.irfft(y_fft_low, len(y_percussive))
                y_high = np.fft.irfft(y_fft_high, len(y_percussive))
                print("Using FFT-based frequency separation")
        except Exception as e:
            print(f"Error in frequency separation: {str(e)}")
            # Simple fallback
            half = len(y) // 2
            y_low = y[:half]
            y_high = y[half:]
        
        # Fixed optimal parameters for beat detection
        # Low frequencies (bass/kicks)
        # Use more conservative parameters to avoid too many scene changes
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr)
        beats_low = librosa.onset.onset_detect(
            onset_envelope=onset_env_low,
            sr=sr,
            delta=0.5 * 0.08,  # More selective threshold
            wait=4,            # Longer wait to avoid rapid scene changes
            pre_max=3,
            post_max=3, 
            pre_avg=3,
            post_avg=5
        )
        low_beat_times = librosa.frames_to_time(beats_low, sr=sr)
        
        # High frequencies (hi-hats/snares)
        onset_env_high = librosa.onset.onset_strength(y=y_high, sr=sr)
        beats_high = librosa.onset.onset_detect(
            onset_envelope=onset_env_high,
            sr=sr,
            delta=0.4 * 0.08,  # Standard threshold
            wait=1,            # Short wait for frequent effects
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=3
        )
        high_beat_times = librosa.frames_to_time(beats_high, sr=sr)
        
        # Additional cleanup to remove beats that are too close together
        # For low beats - ensure minimum spacing of 0.5 seconds between scene changes
        if len(low_beat_times) > 1:
            filtered_low_beats = [low_beat_times[0]]
            min_spacing = 0.5  # seconds
            
            for beat in low_beat_times[1:]:
                if beat - filtered_low_beats[-1] >= min_spacing:
                    filtered_low_beats.append(beat)
            
            low_beat_times = np.array(filtered_low_beats)
        
        print(f"Detected {len(low_beat_times)} low frequency beats (scene changes)")
        print(f"Detected {len(high_beat_times)} high frequency beats (effects)")
        
        return low_beat_times, high_beat_times, sr
    
    def get_audio_duration(self, audio_path, sr=None):
        """Get duration of audio file"""
        try:
            result = subprocess.run([
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                audio_path
            ], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            # Fallback to librosa
            if sr is None:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            else:
                return 0
    
    def get_frame_at_time(self, video_cap, time_pos):
        """Extract a frame at a specific time position"""
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Handle case where time_pos exceeds video duration
        if time_pos >= duration:
            time_pos = time_pos % duration
        
        # Calculate frame number
        frame_num = int(time_pos * fps)
        
        # Set position and read frame
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video_cap.read()
        
        if not ret:
            # If frame read failed, return a black frame
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        return frame
    
    def rgb_split_effect(self, frame, intensity=1.0):
        """Create RGB split effect like a playhead"""
        # Create copies for each channel
        height, width = frame.shape[:2]
        
        # Determine shift amount based on intensity
        shift_amount = int(max(3, min(20, 10 * intensity)))
        
        # Create output frame
        result = frame.copy()
        
        # Apply RGB channel shifts
        if shift_amount > 0:
            # Red channel shift right
            result[:, shift_amount:, 2] = frame[:, :-shift_amount, 2]
            # Blue channel shift left
            result[:, :-shift_amount, 0] = frame[:, shift_amount:, 0]
        
        return result
    
    def echo_effect(self, frame, prev_frame, intensity=0.5):
        """Create a trailing echo/ghost effect"""
        if prev_frame is None:
            return frame
        
        # Blend current frame with previous frame
        alpha = max(0.2, min(0.8, intensity * 0.5))
        return cv2.addWeighted(frame, 1.0, prev_frame, alpha, 0)
    
    def vhs_effect(self, frame, intensity=1.0):
        """Create VHS-style distortion"""
        # Add noise
        noise_intensity = min(50, max(10, int(intensity * 25)))
        noise = np.random.randint(0, noise_intensity, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add slight color shift
        frame = self.rgb_split_effect(frame, intensity=intensity*0.3)
        
        # Reduce color depth slightly
        frame = (frame // 16) * 16
        
        return frame
    
    def glitch_effect(self, frame, intensity=1.0):
        """Create digital glitch effect"""
        height, width = frame.shape[:2]
        result = frame.copy()
        
        # Number of glitch rectangles
        num_glitches = int(max(3, min(10, intensity * 7)))
        
        for _ in range(num_glitches):
            try:
                # Random small area
                x = random.randint(0, width - 30)
                y = random.randint(0, height - 10)
                w = random.randint(10, min(width - x, 100))
                h = random.randint(5, min(height - y, 40))
                
                # Get glitch area
                glitch_area = frame[y:y+h, x:x+w]
                
                # Random glitch type
                glitch_type = random.randint(0, 3)
                
                if glitch_type == 0:
                    # Shift horizontally
                    shift = random.randint(-20, 20)
                    if shift > 0 and x + w + shift < width:
                        result[y:y+h, x+shift:x+w+shift] = glitch_area
                    elif shift < 0 and x + shift >= 0:
                        result[y:y+h, x+shift:x+w+shift] = glitch_area
                elif glitch_type == 1:
                    # Color channel swap
                    glitch_area_mod = glitch_area.copy()
                    if glitch_area_mod.shape[2] >= 3:  # Check channels
                        temp = glitch_area_mod[:,:,0].copy()
                        glitch_area_mod[:,:,0] = glitch_area_mod[:,:,1]
                        glitch_area_mod[:,:,1] = temp
                        result[y:y+h, x:x+w] = glitch_area_mod
                elif glitch_type == 2:
                    # Pixelate
                    cell_size = random.randint(3, 10)
                    if h > cell_size and w > cell_size:
                        for i in range(0, h, cell_size):
                            for j in range(0, w, cell_size):
                                if i + cell_size <= h and j + cell_size <= w:
                                    block = glitch_area[i:i+cell_size, j:j+cell_size]
                                    if block.size > 0:
                                        block_color = block.mean(axis=(0,1))
                                        result[y+i:y+i+cell_size, x+j:x+j+cell_size] = block_color
                else:
                    # Invert colors
                    result[y:y+h, x:x+w] = 255 - glitch_area
            except Exception as e:
                # Ignore errors in glitch effects
                pass
        
        return result
    
    def zoom_pulse_effect(self, frame, intensity=1.0):
        """Create a zoom pulse effect"""
        height, width = frame.shape[:2]
        
        # Calculate zoom factor
        zoom_factor = 1.0 + (0.05 * intensity)
        
        # Calculate new dimensions
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize the frame
        zoomed = cv2.resize(frame, (new_width, new_height))
        
        # Crop to original size
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        
        if start_x < 0 or start_y < 0 or start_x + width > new_width or start_y + height > new_height:
            return frame  # Safety check
        
        result = zoomed[start_y:start_y+height, start_x:start_x+width]
        return result
    
    def write_video_directly(self, video_folder, audio_path, width, height, fps, max_beats, 
                           effect_intensity, filename_prefix, low_beat_times, high_beat_times):
        """Write video directly to file instead of returning frames - allows processing full songs"""
        try:
            # Get output path
            output_dir = folder_paths.get_output_directory()
            
            # Get timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_filename = f"{filename_prefix}_{timestamp}.mp4"
            temp_video_path = os.path.join(output_dir, "temp_" + output_filename)
            final_output_path = os.path.join(output_dir, output_filename)
            
            # Create temp folder if needed
            os.makedirs(output_dir, exist_ok=True)
            
            # Get audio duration for total frames
            audio_duration = self.get_audio_duration(audio_path)
            total_frames = int(audio_duration * fps)
            
            print(f"Writing {total_frames} frames directly to {temp_video_path}")
            
            # Get list of video files and open them
            video_files = []
            for file in os.listdir(video_folder):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    video_path = os.path.join(video_folder, file)
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            video_files.append(video_path)
                        cap.release()
                    except:
                        pass
            
            if not video_files:
                raise ValueError(f"No valid videos found in {video_folder}")
            
            # Shuffle and prepare video files
            random.shuffle(video_files)
            
            # Open video files and keep them ready
            video_caps = []
            for video_file in video_files:
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    video_caps.append({
                        'cap': cap,
                        'path': video_file,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                    })
            
            if not video_caps:
                raise ValueError("Could not open any of the videos")
            
            # Initialize output video writer - use same format as original script
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            # Process frames directly - more memory efficient
            pbar = ProgressBar(total_frames)
            current_video_idx = 0
            prev_frame = None
            
            # Process each frame and write directly to file
            for frame_idx in range(total_frames):
                # Update progress
                if frame_idx % 10 == 0:  # Update less frequently to speed up processing
                    pbar.update_absolute(frame_idx)
                
                # Current time in seconds
                current_time = frame_idx / fps
                
                # Check if we need to change videos based on low frequency beats
                for beat_time in low_beat_times:
                    if abs(current_time - beat_time) < 1.0/fps:
                        # Switch to the next video on a low-frequency beat
                        current_video_idx = (current_video_idx + 1) % len(video_caps)
                        break
                
                # Get current video
                video = video_caps[current_video_idx]
                
                # Get frame from current position - use a position that cycles through the video
                time_in_video = current_time % video['duration']
                frame = self.get_frame_at_time(video['cap'], time_in_video)
                
                # Resize frame to target size
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # Check for high frequency beats to apply effects
                for beat_time in high_beat_times:
                    if abs(current_time - beat_time) < 0.1:  # Within 100ms of a high-frequency beat
                        # Calculate effect strength based on proximity to the beat
                        beat_offset = abs(current_time - beat_time) / 0.1
                        effect_strength = effect_intensity * (1.0 - beat_offset)
                        
                        # Choose effect type based on beat pattern
                        beat_idx = np.where(high_beat_times == beat_time)[0][0]
                        beat_pattern = (beat_idx % 16) // 4
                        
                        # Apply the effect based on the pattern
                        if beat_pattern == 0:
                            # RGB split effect
                            frame = self.rgb_split_effect(frame, intensity=effect_strength)
                        elif beat_pattern == 1:
                            # Echo/ghost effect
                            if prev_frame is not None:
                                frame = self.echo_effect(frame, prev_frame, intensity=effect_strength)
                        elif beat_pattern == 2:
                            # Glitch effect
                            frame = self.glitch_effect(frame, intensity=effect_strength)
                        else:
                            # VHS effect
                            frame = self.vhs_effect(frame, intensity=effect_strength)
                        
                        # Add zoom pulse on high frequency beats
                        frame = self.zoom_pulse_effect(frame, intensity=effect_strength * 0.5)
                        break
                
                # Store frame for echo effects
                prev_frame = frame.copy()
                
                # Write frame directly to output
                out.write(frame)
            
            # Release resources
            out.release()
            for video in video_caps:
                video['cap'].release()
            
            pbar.update(total_frames)
            
            # Add audio using ffmpeg
            print("Adding audio to the final video...")
            
            # Add audio using ffmpeg
            final_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                final_output_path
            ]
            
            try:
                subprocess.run(final_cmd, check=True, capture_output=True)
                print(f"Video successfully saved to: {final_output_path}")
                
                # Try to remove temp file
                try:
                    os.remove(temp_video_path)
                except:
                    pass
                
                return final_output_path
                
            except Exception as e:
                print(f"Error adding audio: {e}")
                print("Using video without audio")
                return temp_video_path
                
        except Exception as e:
            print(f"Error in direct video writing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_video(self, video_folder, audio_path, width, height, fps, max_beats, max_frames, 
                     effect_intensity, output_mode, filename_prefix):
        # Initialize progress bar
        pbar = ProgressBar(100)
        pbar.update(0)
        
        try:
            print(f"Starting beat-sync video processing with {width}x{height} resolution")
            
            # Check if video folder exists
            if not os.path.exists(video_folder):
                raise ValueError(f"Video folder does not exist: {video_folder}")
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file does not exist: {audio_path}")
            
            # Get list of video files
            video_files = []
            for file in os.listdir(video_folder):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    video_path = os.path.join(video_folder, file)
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            video_files.append(video_path)
                        cap.release()
                    except:
                        pass
            
            if not video_files:
                print(f"No valid videos found in {video_folder}")
                # Return a dummy frame
                dummy = np.zeros((height, width, 3), dtype=np.uint8)
                dummy_pil = Image.fromarray(dummy)
                return (self.p2t(dummy_pil), self.get_audio_data(audio_path))
            
            print(f"Found {len(video_files)} videos in {video_folder}")
            pbar.update(5)
            
            # Detect beats in audio
            low_beat_times, high_beat_times, sr = self.detect_frequency_beats(audio_path)
            
            pbar.update(20)
            
            # Limit beats if specified
            if max_beats > 0:
                if len(low_beat_times) > max_beats:
                    print(f"Limiting to {max_beats} low frequency beats")
                    low_beat_times = low_beat_times[:max_beats]
                if len(high_beat_times) > max_beats * 2:  # Allow more high-frequency beats
                    print(f"Limiting to {max_beats * 2} high frequency beats")
                    high_beat_times = high_beat_times[:max_beats * 2]
            
            # Load audio for return
            audio_data = self.get_audio_data(audio_path)
            
            # Handle different output modes
            if output_mode == "Direct Video Output":
                # Write directly to file and return the path
                output_path = self.write_video_directly(
                    video_folder, audio_path, width, height, fps, max_beats,
                    effect_intensity, filename_prefix, low_beat_times, high_beat_times
                )
                
                if output_path:
                    # Create a dummy frame just to satisfy the output requirements
                    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    dummy_tensor = self.p2t(Image.fromarray(dummy_frame))
                    
                    # Create a UI notification
                    print(f"\n=== VIDEO SAVED ===\nThe full-length beat-synced video has been saved to:\n{output_path}\n")
                    
                    # Return the data
                    return (dummy_tensor, audio_data)
                else:
                    # Fall back to frame-based processing if direct writing fails
                    print("Direct video writing failed, falling back to frame-based processing")
                    output_mode = "Frames for Editing"
            
            # If we're here, we're doing frame-based processing
            if output_mode == "Frames for Editing":
                # Calculate how many frames to process (use max_frames as a limit)
                audio_duration = self.get_audio_duration(audio_path, sr)
                
                # Determine frames to generate based on max_frames
                if max_frames > 0:
                    # Use max_frames as an explicit limit
                    if len(low_beat_times) > 0 or len(high_beat_times) > 0:
                        max_beat_time = max(
                            max(low_beat_times) if len(low_beat_times) > 0 else 0,
                            max(high_beat_times) if len(high_beat_times) > 0 else 0
                        )
                        calculated_frames = int((max_beat_time + 5) * fps)
                    else:
                        calculated_frames = int(min(audio_duration, 60) * fps)  # Limit to 60 sec if no beats
                    
                    total_frames = min(max_frames, calculated_frames)
                else:
                    # For memory safety, use a reasonable limit when no explicit max is provided
                    memory_safe_limit = 300  # A reasonable number of frames for memory
                    total_frames = min(int(min(audio_duration, 30) * fps), memory_safe_limit)
                    print(f"Using {total_frames} frames for preview (set max_frames or use Direct Output for full song)")
                
                print(f"Generating {total_frames} frames at {fps} FPS")
                
                # Open video files
                random.shuffle(video_files)
                video_caps = []
                
                for video_file in video_files:
                    cap = cv2.VideoCapture(video_file)
                    if cap.isOpened():
                        video_caps.append({
                            'cap': cap,
                            'path': video_file,
                            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps': cap.get(cv2.CAP_PROP_FPS),
                            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                        })
                
                if not video_caps:
                    print("Could not open any of the videos")
                    # Return a dummy frame
                    dummy = np.zeros((height, width, 3), dtype=np.uint8)
                    dummy_pil = Image.fromarray(dummy)
                    return (self.p2t(dummy_pil), audio_data)
                
                pbar.update(30)
                
                # Process frames for return
                frames_pil = []
                prev_frame = None
                current_video_idx = 0  # Start with the first video
                
                # Process each frame
                for frame_idx in range(total_frames):
                    # Update progress
                    progress = int(30 + 60 * frame_idx / total_frames)
                    pbar.update_absolute(progress)
                    
                    # Current time in seconds
                    current_time = frame_idx / fps
                    
                    # Check if we need to change videos based on low frequency beats
                    # IMPORTANT: Only change videos on low frequency beats
                    for beat_time in low_beat_times:
                        if abs(current_time - beat_time) < 1.0/fps:
                            # Switch to the next video on a low-frequency beat
                            current_video_idx = (current_video_idx + 1) % len(video_caps)
                            break
                    
                    # Get current video
                    video = video_caps[current_video_idx]
                    
                    # Get frame from current position - use a position that cycles through the video
                    time_in_video = current_time % video['duration']
                    frame = self.get_frame_at_time(video['cap'], time_in_video)
                    
                    # Resize frame to target size
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    
                    # Check for high frequency beats to apply effects
                    for beat_time in high_beat_times:
                        if abs(current_time - beat_time) < 0.1:  # Within 100ms of a high-frequency beat
                            # Calculate effect strength based on proximity to the beat
                            beat_offset = abs(current_time - beat_time) / 0.1
                            effect_strength = effect_intensity * (1.0 - beat_offset)
                            
                            # Choose effect type based on beat pattern
                            beat_idx = np.where(high_beat_times == beat_time)[0][0]
                            beat_pattern = (beat_idx % 16) // 4
                            
                            # Apply the effect based on the pattern
                            if beat_pattern == 0:
                                # RGB split effect
                                frame = self.rgb_split_effect(frame, intensity=effect_strength)
                            elif beat_pattern == 1:
                                # Echo/ghost effect
                                if prev_frame is not None:
                                    frame = self.echo_effect(frame, prev_frame, intensity=effect_strength)
                            elif beat_pattern == 2:
                                # Glitch effect
                                frame = self.glitch_effect(frame, intensity=effect_strength)
                            else:
                                # VHS effect
                                frame = self.vhs_effect(frame, intensity=effect_strength)
                            
                            # Add zoom pulse on high frequency beats
                            frame = self.zoom_pulse_effect(frame, intensity=effect_strength * 0.5)
                            break
                    
                    # Store frame for echo effects
                    prev_frame = frame.copy()
                    
                    # Convert BGR to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL and add to list
                    pil_image = Image.fromarray(frame_rgb)
                    frames_pil.append(pil_image)
                
                # Clean up video captures
                for video in video_caps:
                    video['cap'].release()
                
                pbar.update(95)
                
                # Convert frames to tensor batch
                frames_tensor = []
                for pil_frame in frames_pil:
                    frame_tensor = self.p2t(pil_frame)
                    frames_tensor.append(frame_tensor)
                
                # Combine into a batch - with safety for memory issues
                if frames_tensor:
                    try:
                        result_tensor = torch.cat(frames_tensor, dim=0)
                        print(f"Final tensor shape: {result_tensor.shape}")
                    except RuntimeError as e:
                        # If we hit a memory error, try with fewer frames
                        if "memory" in str(e):
                            print("Memory error - reducing number of frames")
                            if len(frames_tensor) > 100:
                                # Take evenly spaced frames
                                step = len(frames_tensor) // 100
                                result_tensor = torch.cat(frames_tensor[::step][:100], dim=0)
                            else:
                                # Just take the first few
                                result_tensor = torch.cat(frames_tensor[:50], dim=0)
                        else:
                            raise
                else:
                    # Create a dummy frame
                    dummy = np.zeros((height, width, 3), dtype=np.uint8)
                    dummy_pil = Image.fromarray(dummy)
                    result_tensor = self.p2t(dummy_pil)
                
                pbar.update(100)
                return (result_tensor, audio_data)
            
        except Exception as e:
            print(f"Error in FrequencyBeatSyncNode: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a dummy frame
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            dummy_pil = Image.fromarray(dummy)
            
            # Create a minimal valid audio dict as fallback
            audio_data = {
                'waveform': torch.zeros((1, 2, 44100)),  # 1 second of silence, stereo
                'sample_rate': 44100,
                'path': audio_path
            }
            
            return (self.p2t(dummy_pil), audio_data)

# Register the node
NODE_CLASS_MAPPINGS = {
    "FrequencyBeatSyncNode": FrequencyBeatSyncNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencyBeatSyncNode": "Frequency-Based Beat Sync"
}