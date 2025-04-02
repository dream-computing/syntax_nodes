# -*- coding: utf-8 -*-
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
    class PngInfo:
        def __init__(self): self.text = {}
        def add_text(self, key, value): self.text[key] = value

from comfy.utils import ProgressBar
import folder_paths

class FrequencyBeatSyncNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Define ranges for clarity
        PERCENT = {"min": 0.0, "max": 1.0, "step": 0.05}
        MULTIPLIER = {"min": 0.0, "max": 5.0, "step": 0.1}
        STRENGTH = {"min": -2.0, "max": 2.0, "step": 0.05} # For bidirectional effects like distortion
        BOOLEAN_TOGGLE = {"default": True, "label_on": "Enabled", "label_off": "Disabled"}

        return {
            "required": {
                "video_folder": ("STRING", {"default": "videos"}),
                "audio_path": ("STRING", {"default": "audio.mp3"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "max_beats": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Max low-frequency (scene change) beats. 0=unlimited."}),
                "max_frames": ("INT", {"default": 150, "min": 0, "max": 10000, "step": 1, "tooltip": "Max frames for 'Frames for Editing' mode. 0=auto. Ignored for 'Direct Video Output'."}),
                "global_effect_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Global multiplier for beat-triggered effect strength."}),
                "output_mode": (["Frames for Editing", "Direct Video Output"],),
                "filename_prefix": ("STRING", {"default": "BeatSync"}),
                
                # --- Sync Adjustment ---
                "sync_offset_ms": ("INT", {"default": 0, "min": -500, "max": 500, "step": 10, "tooltip": "Adjust sync timing in milliseconds (+/- 500ms). Positive values shift effects later."}),
                
                # --- Beat Detection Parameters ---
                "low_freq_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Sensitivity for low-frequency (scene change) beat detection."}),
                "high_freq_sensitivity": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Sensitivity for high-frequency (effect) beat detection."}),
                
                # --- Video Sequence Control ---
                "random_video_order": ("BOOLEAN", {**BOOLEAN_TOGGLE, "default": True, "tooltip": "Use random video ordering? If disabled, videos will be used in alphabetical order."}),

                # --- Effect Enable Toggles ---
                "enable_rgb_split": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable RGB Split effect on high-frequency beats?"}),
                "enable_echo": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Echo effect on high-frequency beats?"}),
                "enable_zoom_pulse": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Zoom Pulse effect on high-frequency beats?"}),
                "enable_vhs": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable VHS effect on high-frequency beats?"}),
                "enable_glitch": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Glitch effect on high-frequency beats?"}),
                "enable_frame_stutter": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Frame Stutter effect on high-frequency beats?"}),
                "enable_hue_shift": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Hue Shift effect on high-frequency beats?"}),
                "enable_saturation_pulse": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Saturation Pulse effect on high-frequency beats?"}),
                "enable_brightness_flash": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Brightness Flash effect on high-frequency beats?"}),
                "enable_barrel_distortion": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Barrel/Pincushion Distortion effect on high-frequency beats?"}),
                "enable_scanlines": ("BOOLEAN", {**BOOLEAN_TOGGLE, "tooltip": "Enable Scanline effect on high-frequency beats?"}),

                # --- Effect Parameters (remain the same) ---
                # Basic Effects
                "rgb_shift_amount": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "tooltip": "Base pixel shift for RGB Split."}),
                "echo_alpha": ("FLOAT", {"default": 0.5, **PERCENT, "tooltip": "Base opacity for Echo's previous frame blend."}),
                "zoom_factor": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 3.0, "step": 0.01, "tooltip": "Base zoom factor for Zoom Pulse (>1 zoom in, <1 zoom out)."}),
                "zoom_pulse_on_low_freq": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Apply zoom pulse also on low frequency beats (scene changes)."}),

                # VHS Effects
                "vhs_noise_intensity": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1, "tooltip": "Base intensity for VHS static noise."}),
                "vhs_rgb_split_intensity": ("FLOAT", {"default": 0.3, **MULTIPLIER, "tooltip": "Base multiplier for RGB split within VHS effect."}),
                "vhs_quantization_level": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1, "tooltip": "Color depth reduction for VHS (higher=more posterized, 1=off)."}),

                # Glitch Effects
                "glitch_num_rectangles": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1, "tooltip": "Base number of glitch rectangles."}),
                # Frame stutter toggle moved to enable toggles above

                # Color Effects
                "hue_shift_degrees": ("INT", {"default": 30, "min": -180, "max": 180, "step": 5, "tooltip": "Base degrees to shift hue (-180 to 180)."}),
                "saturation_multiplier": ("FLOAT", {"default": 1.5, **MULTIPLIER, "tooltip": "Base multiplier for saturation (>1 increase, <1 decrease)."}),
                "brightness_add": ("INT", {"default": 50, "min": -255, "max": 255, "step": 5, "tooltip": "Base value added/subtracted to brightness."}),

                # Distortion Effects
                 "barrel_distortion_strength": ("FLOAT", {"default": -0.2, **STRENGTH, "tooltip": "Base strength for Barrel (>0) or Pincushion (<0) distortion."}),

                 # Overlay Effects
                 "scanline_intensity": ("FLOAT", {"default": 0.15, **PERCENT, "tooltip": "Base intensity (opacity) of scanlines."}),
                 "scanline_density": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Density of scanlines (higher = thinner lines closer together)."}),
            },
             "optional": {
                 # Optional inputs can go here if needed in the future
                 # We've removed the effect_selection string and replaced it with toggles
             }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "process_video"
    CATEGORY = "Video Processing/Beat Sync"

    # --- Helper Functions (t2p, p2t, get_audio_data, etc.) --- unchanged ---
    def t2p(self, t):
        if t is not None:
            i = 255.0 * t.cpu().numpy().squeeze()
            return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    def p2t(self, p):
        if p is not None:
            i = np.array(p).astype(np.float32) / 255.0
            return torch.from_numpy(i).unsqueeze(0)
    def get_audio_data(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            wf = torch.from_numpy(y).unsqueeze(0)
            if len(y.shape) == 1: wf = wf.unsqueeze(0)
            return {'waveform': wf, 'sample_rate': sr, 'path': audio_path}
        except Exception as e:
            print(f"Error loading audio: {e}")
            return {'waveform': torch.zeros((1, 2, 44100)), 'sample_rate': 44100, 'path': audio_path}

    # --- Beat Detection --- IMPROVED with adjustable sensitivity ---
    def detect_frequency_beats(self, audio_path, low_freq_sensitivity=0.5, high_freq_sensitivity=0.4):
        """Detect beats in different frequency ranges of the audio file using optimized parameters"""
        print("Analyzing audio frequencies...")
        y, sr = librosa.load(audio_path, sr=None)

        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)

            # Use filtering approach for low/high separation
            try:
                from scipy import signal
                sos_low = signal.butter(10, 150, 'lowpass', fs=sr, output='sos')
                y_low = signal.sosfilt(sos_low, y_percussive)
                y_high = y_percussive - y_low
                print("Using filter bank for frequency separation on percussive component.")
            except Exception as e:
                print(f"Filter bank failed ({e}), falling back to FFT separation")
                y_fft = np.fft.rfft(y_percussive)
                cutoff = int(len(y_fft) * 150 / (sr/2)) # ~150Hz cutoff
                y_fft_low = y_fft.copy(); y_fft_low[cutoff:] = 0
                y_fft_high = y_fft.copy(); y_fft_high[:cutoff] = 0
                y_low = np.fft.irfft(y_fft_low, len(y_percussive))
                y_high = np.fft.irfft(y_fft_high, len(y_percussive))

        except Exception as e:
            print(f"Error in HPSS/frequency separation: {str(e)}, using simple split")
            # Fallback if HPSS fails entirely
            half = len(y) // 2
            y_low = y[:half]
            y_high = y[half:]

        # Low frequencies (bass/kicks) - scene changes - using onset_detect with adjustable sensitivity
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr)
        beats_low = librosa.onset.onset_detect(
            onset_envelope=onset_env_low,
            sr=sr,
            delta=low_freq_sensitivity * 0.08,  # Adjustable threshold based on sensitivity parameter
            wait=4,            # Longer wait to avoid rapid scene changes
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            units='frames'     # Ensure units are frames
        )
        low_beat_times = librosa.frames_to_time(beats_low, sr=sr)

        # High frequencies (hi-hats/snares) - effects - using onset_detect with adjustable sensitivity
        onset_env_high = librosa.onset.onset_strength(y=y_high, sr=sr)
        beats_high = librosa.onset.onset_detect(
            onset_envelope=onset_env_high,
            sr=sr,
            delta=high_freq_sensitivity * 0.08,  # Adjustable threshold based on sensitivity parameter
            wait=1,            # Short wait for frequent effects
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=3,
            units='frames'     # Ensure units are frames
        )
        high_beat_times = librosa.frames_to_time(beats_high, sr=sr)

        # Cleanup low beats - ensure minimum spacing
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

    # --- Frame Extraction / Audio Duration --- (unchanged)
    def get_audio_duration(self, audio_path, sr=None):
        try:
            result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"ffprobe failed ({e}), using librosa for duration")
            try: return librosa.get_duration(path=audio_path)
            except Exception as le: print(f"Librosa duration failed ({le}), returning 0"); return 0
    def get_frame_at_time(self, video_cap, time_pos):
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            return np.zeros((h if h > 0 else 480, w if w > 0 else 640, 3), dtype=np.uint8)
        duration = frame_count / fps
        effective_time_pos = time_pos % duration if duration > 0 else 0
        frame_num = max(0, min(frame_count - 1, int(effective_time_pos * fps)))
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video_cap.read()
        if not ret:
            h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            return np.zeros((h if h > 0 else 480, w if w > 0 else 640, 3), dtype=np.uint8)
        return frame

    # --- Effect Functions (unchanged) ---
    def rgb_split_effect(self, frame, shift_amount):
        if shift_amount <= 0: return frame
        h, w = frame.shape[:2]; result = frame.copy(); shift = int(round(shift_amount))
        if shift > 0:
            result[:, shift:, 2] = frame[:, :-shift, 2]; result[:, :shift, 2] = frame[:, 0:1, 2]
            result[:, :-shift, 0] = frame[:, shift:, 0]; result[:, -shift:, 0] = frame[:, -1:, 0]
        return result
    def echo_effect(self, frame, prev_frame, alpha):
        if prev_frame is None or alpha <= 0.0: return frame
        if alpha >= 1.0: return prev_frame
        if frame.shape != prev_frame.shape: return frame
        return cv2.addWeighted(frame, 1.0 - alpha, prev_frame, alpha, 0)
    def vhs_effect(self, frame, noise_intensity, rgb_split_intensity, quantization_level):
        result = frame.copy()
        if noise_intensity > 0:
            noise_val = int(round(noise_intensity))
            noise = np.random.randint(-noise_val // 2, noise_val // 2 + 1, frame.shape, dtype=np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if rgb_split_intensity > 0:
             result = self.rgb_split_effect(result, 5 * rgb_split_intensity) # Keep internal dependency
        q_level = int(round(quantization_level))
        if q_level > 1: result = (result // q_level) * q_level
        return result
    def glitch_effect(self, frame, num_glitches):
        if num_glitches <= 0: return frame
        height, width = frame.shape[:2]; result = frame.copy(); glitches_applied = 0
        max_attempts = num_glitches * 5; attempts = 0
        while glitches_applied < num_glitches and attempts < max_attempts:
            attempts += 1;
            try:
                max_w=max(10,width//5); max_h=max(5,height//10); w=random.randint(10,max(11,max_w)); h=random.randint(5,max(6,max_h))
                x=random.randint(0,max(0,width-w-1)); y=random.randint(0,max(0,height-h-1))
                if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height: continue
                glitch_area=frame[y:y+h,x:x+w];
                if glitch_area.size==0: continue
                glitch_type=random.randint(0,3)
                if glitch_type==0:
                    max_shift=w//2; shift=random.randint(-max_shift,max_shift);
                    if shift==0: continue
                    target_x_start=x+shift; target_x_end=x+w+shift
                    if target_x_start>=0 and target_x_end<=width: result[y:y+h,target_x_start:target_x_end]=glitch_area; glitches_applied+=1
                elif glitch_type==1:
                    if glitch_area.shape[2]>=3:
                        glitch_area_mod=glitch_area.copy(); ch1,ch2=random.sample([0,1,2],2); temp=glitch_area_mod[:,:,ch1].copy(); glitch_area_mod[:,:,ch1]=glitch_area_mod[:,:,ch2]; glitch_area_mod[:,:,ch2]=temp
                        result[y:y+h,x:x+w]=glitch_area_mod; glitches_applied+=1
                elif glitch_type==2:
                    min_cell=3; max_cell=max(min_cell+1,min(w,h)//2); cell_size=random.randint(min_cell,max_cell)
                    if h>=cell_size and w>=cell_size:
                        temp_area=glitch_area.copy()
                        for i in range(0,h-h%cell_size,cell_size):
                            for j in range(0,w-w%cell_size,cell_size):
                                block=glitch_area[i:i+cell_size,j:j+cell_size];
                                if block.size>0: temp_area[i:i+cell_size,j:j+cell_size]=np.mean(block,axis=(0,1),dtype=int)
                        result[y:y+h,x:x+w]=temp_area; glitches_applied+=1
                else: result[y:y+h,x:x+w]=255-glitch_area; glitches_applied+=1
            except Exception: pass
        return result
    def zoom_pulse_effect(self, frame, zoom_factor):
        if abs(zoom_factor - 1.0) < 0.005: return frame
        h, w = frame.shape[:2];
        if zoom_factor <= 0: return frame
        try:
            if zoom_factor == 1.0: return frame
            new_w = int(round(w * zoom_factor)); new_h = int(round(h * zoom_factor))
            if new_w <= 0 or new_h <= 0: return frame
            zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if zoom_factor > 1.0:
                start_x=(new_w-w)//2; start_y=(new_h-h)//2; end_x=start_x+w; end_y=start_y+h
                if start_x<0 or start_y<0 or end_x>new_w or end_y>new_h: return frame
                return zoomed[start_y:end_y, start_x:end_x]
            else:
                result = np.zeros_like(frame); start_x=(w-new_w)//2; start_y=(h-new_h)//2; end_x=start_x+new_w; end_y=start_y+new_h
                if start_x<0 or start_y<0 or end_x>w or end_y>h: return frame
                result[start_y:end_y, start_x:end_x] = zoomed; return result
        except Exception as e: print(f"Warn: Zoom error (factor {zoom_factor}): {e}"); return frame
    def hue_shift_effect(self, frame, degrees):
        if abs(degrees) < 1: return frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue_shift = int(round(degrees / 2.0)) # OpenCV hue is 0-179
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e: print(f"Warn: Hue shift error: {e}"); return frame
    def saturation_pulse_effect(self, frame, multiplier):
        if abs(multiplier - 1.0) < 0.01: return frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32) * multiplier
            hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e: print(f"Warn: Saturation pulse error: {e}"); return frame
    def brightness_flash_effect(self, frame, amount):
        if abs(amount) < 1: return frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2].astype(np.int16) + int(round(amount))
            hsv[:, :, 2] = np.clip(value, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e: print(f"Warn: Brightness flash error: {e}"); return frame
    def barrel_distortion_effect(self, frame, strength):
        if abs(strength) < 0.005: return frame
        try:
            h, w = frame.shape[:2]; cx, cy = w / 2.0, h / 2.0
            map_x = np.zeros((h, w), dtype=np.float32); map_y = np.zeros((h, w), dtype=np.float32)
            max_rad_sq = cx*cx + cy*cy
            k = strength * 0.5 # Adjusted scaling factor for strength input
            for y_px in range(h):
                for x_px in range(w):
                    x_rel=x_px-cx; y_rel=y_px-cy;
                    r_sq_norm=(x_rel*x_rel+y_rel*y_rel)/max_rad_sq if max_rad_sq>0 else 0
                    dist_factor = (1 + k * r_sq_norm);
                    src_x = cx + x_rel * dist_factor
                    src_y = cy + y_rel * dist_factor
                    map_x[y_px, x_px] = src_x
                    map_y[y_px, x_px] = src_y
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        except Exception as e: print(f"Warn: Barrel distortion error: {e}"); return frame
    def scanline_effect(self, frame, intensity, density):
         if intensity <= 0.0 or density <= 0: return frame
         try:
             h, w = frame.shape[:2]; lines = np.zeros_like(frame, dtype=np.uint8)
             line_thickness = 1;
             spacing = max(1, int(round(h / (density * 5.0)))) # Adjusted scaling factor based on density input range
             for y in range(0, h, spacing + line_thickness):
                 cv2.line(lines, (0, y), (w, y), (0, 0, 0), thickness=line_thickness)
             return cv2.addWeighted(frame, 1.0 - intensity, lines, intensity, 0.0)
         except Exception as e: print(f"Warn: Scanline error: {e}"); return frame

    # --- Core Effect Application Logic --- IMPROVED with simpler beat detection but keeping toggles ---
    def _apply_effects(self, frame, prev_raw_frame, prev_processed_frame, current_time, low_beat_times, high_beat_times, effect_params, fps):
        modified_frame = frame.copy()
        effect_applied_this_frame = False
        
        # Apply sync offset (convert ms to seconds)
        sync_offset_sec = effect_params.get('sync_offset_ms', 0) / 1000.0
        adjusted_time = current_time - sync_offset_sec
        
        # Get effect window (using fixed 0.1 seconds like original implementation)
        beat_window = 0.1  # Fixed 0.1 second window as in original
        
        # Determine available effects based on toggles
        available_effects = []
        if effect_params.get('enable_rgb_split', False): available_effects.append('rgb')
        if effect_params.get('enable_echo', False): available_effects.append('echo')
        if effect_params.get('enable_zoom_pulse', False): available_effects.append('zoom')
        if effect_params.get('enable_vhs', False): available_effects.append('vhs')
        if effect_params.get('enable_glitch', False): available_effects.append('glitch')
        if effect_params.get('enable_frame_stutter', False): available_effects.append('stutter')
        if effect_params.get('enable_hue_shift', False): available_effects.append('hue')
        if effect_params.get('enable_saturation_pulse', False): available_effects.append('sat')
        if effect_params.get('enable_brightness_flash', False): available_effects.append('bright')
        if effect_params.get('enable_barrel_distortion', False): available_effects.append('barrel')
        if effect_params.get('enable_scanlines', False): available_effects.append('scan')

        if not available_effects:
            # Skip high-freq effects if none are enabled
            pass
        else:
            # Using original approach: check if within 0.1 seconds of any beat
            for i, beat_time in enumerate(high_beat_times):
                time_diff = abs(adjusted_time - beat_time)
                
                # Only process beats within the effect window (0.1 seconds)
                if time_diff < beat_window:
                    # Straightforward proximity-based effect strength
                    beat_offset = time_diff / beat_window
                    effect_strength = effect_params['global_effect_intensity'] * (1.0 - beat_offset)
                    
                    # Skip very weak effects
                    if effect_strength <= 0.01:
                        continue
                    
                    # Choose effect type based on the beat index
                    # Fixed: Using i instead of np.where which could cause errors
                    effect_choice = available_effects[i % len(available_effects)]
                    
                    # Apply the chosen effect with calculated strength
                    applied_now = True  # Assume effect will be applied unless check fails
                    
                    if effect_choice == 'rgb':
                        eff_shift = max(0, effect_params['rgb_shift_amount'] * effect_strength)
                        if eff_shift > 0.5:
                            modified_frame = self.rgb_split_effect(modified_frame, eff_shift)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'echo' and prev_raw_frame is not None:
                        eff_alpha = np.clip(effect_params['echo_alpha'] * effect_strength, 0.0, 1.0)
                        if eff_alpha > 0.01:
                            modified_frame = self.echo_effect(modified_frame, prev_raw_frame, eff_alpha)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'glitch':
                        eff_num = int(round(effect_params['glitch_num_rectangles'] * effect_strength))
                        if eff_num > 0:
                            modified_frame = self.glitch_effect(modified_frame, eff_num)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'vhs':
                        eff_noise = max(0, effect_params['vhs_noise_intensity'] * effect_strength)
                        eff_rgb_split = max(0.0, effect_params['vhs_rgb_split_intensity'] * effect_strength)
                        eff_quant = effect_params['vhs_quantization_level']
                        if eff_noise > 0.5 or eff_rgb_split > 0.01 or eff_quant > 1:
                            modified_frame = self.vhs_effect(modified_frame, eff_noise, eff_rgb_split, eff_quant)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'zoom':
                        base_zoom = effect_params['zoom_factor']
                        eff_zoom = 1.0 + (base_zoom - 1.0) * effect_strength
                        if abs(eff_zoom - 1.0) > 0.005:
                            modified_frame = self.zoom_pulse_effect(modified_frame, eff_zoom)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'hue':
                        eff_degrees = effect_params['hue_shift_degrees'] * effect_strength
                        if abs(eff_degrees) >= 1:
                            modified_frame = self.hue_shift_effect(modified_frame, eff_degrees)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'sat':
                        base_sat = effect_params['saturation_multiplier']
                        eff_sat = 1.0 + (base_sat - 1.0) * effect_strength
                        if abs(eff_sat - 1.0) > 0.01:
                            modified_frame = self.saturation_pulse_effect(modified_frame, eff_sat)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'bright':
                        eff_bright = effect_params['brightness_add'] * effect_strength
                        if abs(eff_bright) >= 1:
                            modified_frame = self.brightness_flash_effect(modified_frame, eff_bright)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'barrel':
                        eff_barrel = effect_params['barrel_distortion_strength'] * effect_strength
                        if abs(eff_barrel) > 0.005:
                            modified_frame = self.barrel_distortion_effect(modified_frame, eff_barrel)
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'stutter' and prev_processed_frame is not None:
                        if effect_strength > 0.5:
                            if modified_frame.shape == prev_processed_frame.shape:
                                modified_frame = prev_processed_frame.copy()
                            else:
                                applied_now = False
                        else:
                            applied_now = False
                            
                    elif effect_choice == 'scan':
                        eff_scan_intensity = np.clip(effect_params['scanline_intensity'] * effect_strength, 0.0, 1.0)
                        eff_scan_density = effect_params['scanline_density']
                        if eff_scan_intensity > 0.01 and eff_scan_density > 0:
                            modified_frame = self.scanline_effect(modified_frame, eff_scan_intensity, eff_scan_density)
                        else:
                            applied_now = False
                            
                    else:
                        applied_now = False

                    if applied_now:
                        effect_applied_this_frame = True
                        break  # Apply only one effect at a time

        # Optional Zoom on Low Freq
        if effect_params.get('zoom_pulse_on_low_freq', False) and not effect_applied_this_frame:
            for beat_time in low_beat_times:
                time_diff = abs(adjusted_time - beat_time)
                if time_diff < beat_window:  # Use same 0.1s window
                    # Simple proximity calculation as in original
                    beat_offset = time_diff / beat_window
                    effect_strength = effect_params['global_effect_intensity'] * (1.0 - beat_offset)
                    
                    if effect_strength <= 0.01:
                        continue
                        
                    base_zoom = effect_params['zoom_factor']
                    eff_zoom = 1.0 + (base_zoom - 1.0) * effect_strength
                    
                    if abs(eff_zoom - 1.0) > 0.005:
                        modified_frame = self.zoom_pulse_effect(modified_frame, eff_zoom)
                    
                    break # Apply zoom only once if multiple low beats are close

        return modified_frame

    # --- Main Processing Functions ---
    def write_video_directly(self, video_folder, audio_path, width, height, fps, max_beats,
                             filename_prefix, low_beat_times, high_beat_times, effect_params):
        try:
            output_dir = folder_paths.get_output_directory()
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_filename = f"{filename_prefix}_{timestamp}.mp4"
            temp_video_path = os.path.join(output_dir, "temp_" + output_filename)
            final_output_path = os.path.join(output_dir, output_filename)
            os.makedirs(output_dir, exist_ok=True)
            
            audio_duration = self.get_audio_duration(audio_path)
            if audio_duration <= 0:
                raise ValueError("Audio duration is zero or negative.")
                
            total_frames = int(audio_duration * fps)
            if total_frames <= 0:
                raise ValueError("Calculated total frames is zero or negative.")
                
            print(f"Writing {total_frames} frames directly to {temp_video_path}")
            
            # Get list of video files
            video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) 
                         if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm')) 
                         and os.path.isfile(os.path.join(video_folder, f))]
            
            if not video_files:
                raise ValueError(f"No valid videos found in {video_folder}")
            
            # Sort or randomize based on random_video_order parameter
            if effect_params.get('random_video_order', True):
                random.shuffle(video_files)
            else:
                # Sort alphabetically for sequential order
                video_files.sort()
                
            video_caps = []
            for vf in video_files:
                try:
                    cap = cv2.VideoCapture(vf)
                    if cap.isOpened():
                        v_fps = cap.get(cv2.CAP_PROP_FPS)
                        v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        v_dur = v_frames/v_fps if v_fps > 0 else 0
                        
                        if v_dur > 0.1: 
                            video_caps.append({
                                'cap': cap,
                                'path': vf,
                                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                'fps': v_fps,
                                'frame_count': v_frames,
                                'duration': v_dur
                            })
                        else:
                            cap.release()
                except Exception as e: 
                    print(f"Warn: Error opening {vf}: {e}")
                    
            if not video_caps:
                raise ValueError("Could not open any valid videos.")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise IOError(f"Failed to open video writer for {temp_video_path}")
            
            pbar = ProgressBar(total_frames)
            current_video_idx = 0
            last_low_beat_switch_time = -1.0
            prev_raw_frame = None
            prev_processed_frame = None

            # Calculate sync offset for scene changes (in seconds)
            sync_offset_sec = effect_params.get('sync_offset_ms', 0) / 1000.0

            for frame_idx in range(total_frames):
                pbar.update_absolute(frame_idx + 1)
                current_time = frame_idx / fps
                switched_video = False
                
                # Apply the sync offset to scene changes too
                adjusted_time = current_time - sync_offset_sec
                
                for beat_time in low_beat_times:
                    time_diff = abs(adjusted_time - beat_time)
                    frame_time_window = 1.0 / fps  # One frame duration
                    
                    # Using a smaller time window for more precise scene changes
                    if time_diff < frame_time_window and (adjusted_time - last_low_beat_switch_time) > 0.4:
                        current_video_idx = (current_video_idx + 1) % len(video_caps)
                        last_low_beat_switch_time = adjusted_time
                        switched_video = True
                        break
                        
                video = video_caps[current_video_idx]
                time_in_video = current_time % video['duration'] if video['duration'] > 0 else 0
                current_raw_frame = self.get_frame_at_time(video['cap'], time_in_video)
                
                if current_raw_frame.shape[1] != width or current_raw_frame.shape[0] != height: 
                    current_raw_frame = cv2.resize(current_raw_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                if frame_idx == 0: 
                    prev_raw_frame = current_raw_frame.copy()
                    prev_processed_frame = current_raw_frame.copy()

                modified_frame = self._apply_effects(
                    current_raw_frame,
                    prev_raw_frame,
                    prev_processed_frame,
                    current_time,
                    low_beat_times,
                    high_beat_times,
                    effect_params,
                    fps
                )
                
                out.write(modified_frame)
                prev_raw_frame = current_raw_frame.copy()
                prev_processed_frame = modified_frame.copy()

            out.release()
            [vc['cap'].release() for vc in video_caps]
            # No pbar.finish() call to avoid error
            
            print("Adding audio...")
            final_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                final_output_path
            ]
            
            try:
                process = subprocess.run(
                    final_cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8', 
                    errors='ignore'
                )
                print(f"Video saved: {final_output_path}")
                try:
                    os.remove(temp_video_path)
                except OSError as e:
                    print(f"Warn: Failed to remove temp file {temp_video_path}: {e}")
                return final_output_path
            except subprocess.CalledProcessError as e:
                print(f"ERROR during FFmpeg merge! Command: {' '.join(e.cmd)}\nReturn Code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}")
                print(f"Video *without* audio saved as {temp_video_path}")
                try:
                    os.rename(temp_video_path, final_output_path)
                    return final_output_path
                except OSError as re:
                    print(f"Error renaming temp file: {re}")
                    return temp_video_path
        except Exception as e:
            print(f"--- ERROR in write_video_directly ---")
            import traceback
            traceback.print_exc()
            if 'out' in locals() and out.isOpened():
                out.release()
            if 'video_caps' in locals():
                [vc['cap'].release() for vc in video_caps if vc.get('cap') and vc['cap'].isOpened()]
            return None

    def process_video(self, video_folder, audio_path, width, height, fps, max_beats, max_frames,
                     global_effect_intensity, output_mode, filename_prefix,
                     # Sync parameters - removed effect_window_duration
                     sync_offset_ms,
                     # Beat detection parameters
                     low_freq_sensitivity, high_freq_sensitivity,
                     # Video ordering parameter
                     random_video_order,
                     # Effect enable toggles
                     enable_rgb_split, enable_echo, enable_zoom_pulse, enable_vhs, enable_glitch,
                     enable_frame_stutter, enable_hue_shift, enable_saturation_pulse, enable_brightness_flash,
                     enable_barrel_distortion, enable_scanlines,
                     # Regular effect parameters
                     rgb_shift_amount, echo_alpha, zoom_factor, zoom_pulse_on_low_freq,
                     vhs_noise_intensity, vhs_rgb_split_intensity, vhs_quantization_level,
                     glitch_num_rectangles, 
                     hue_shift_degrees, saturation_multiplier, brightness_add,
                     barrel_distortion_strength,
                     scanline_intensity, scanline_density):

        pbar = ProgressBar(100)
        pbar.update(0)
        
        # Bundle all params into dictionary for easier passing
        effect_params = { 
            # Include sync parameters
            "sync_offset_ms": sync_offset_ms,
            # We don't use effect_window_duration anymore with simpler approach
            
            # Include the video order parameter
            "random_video_order": random_video_order,
            
            # Include enable toggles
            "enable_rgb_split": enable_rgb_split,
            "enable_echo": enable_echo,
            "enable_zoom_pulse": enable_zoom_pulse,
            "enable_vhs": enable_vhs,
            "enable_glitch": enable_glitch,
            "enable_frame_stutter": enable_frame_stutter,
            "enable_hue_shift": enable_hue_shift,
            "enable_saturation_pulse": enable_saturation_pulse,
            "enable_brightness_flash": enable_brightness_flash,
            "enable_barrel_distortion": enable_barrel_distortion,
            "enable_scanlines": enable_scanlines,
            
            # Include effect parameters
            "global_effect_intensity": global_effect_intensity, 
            "rgb_shift_amount": rgb_shift_amount, 
            "echo_alpha": echo_alpha,
            "zoom_factor": zoom_factor, 
            "zoom_pulse_on_low_freq": zoom_pulse_on_low_freq, 
            "vhs_noise_intensity": vhs_noise_intensity,
            "vhs_rgb_split_intensity": vhs_rgb_split_intensity, 
            "vhs_quantization_level": vhs_quantization_level, 
            "glitch_num_rectangles": glitch_num_rectangles,
            "hue_shift_degrees": hue_shift_degrees, 
            "saturation_multiplier": saturation_multiplier, 
            "brightness_add": brightness_add,
            "barrel_distortion_strength": barrel_distortion_strength, 
            "scanline_intensity": scanline_intensity,
            "scanline_density": scanline_density
        }
        
        try:
            print(f"Starting beat-sync: {width}x{height}@{fps}fps, Mode: {output_mode}")
            if not os.path.isdir(video_folder):
                raise ValueError(f"Video folder not found: {video_folder}")
            if not os.path.isfile(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")
            
            video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) 
                       if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm')) 
                       and os.path.isfile(os.path.join(video_folder, f))]
                       
            if not video_files: 
                print(f"No videos found in {video_folder}. Returning black frame.")
                dummy_f = Image.new('RGB', (width, height))
                return (self.p2t(dummy_f), self.get_audio_data(audio_path))
                
            print(f"Found {len(video_files)} videos.")
            pbar.update(5)
            
            # Use the improved beat detection with sensitivity parameters
            low_beat_times, high_beat_times, sr = self.detect_frequency_beats(
                audio_path,
                low_freq_sensitivity=low_freq_sensitivity,
                high_freq_sensitivity=high_freq_sensitivity
            )
            pbar.update(20)
            
            if max_beats > 0 and len(low_beat_times) > max_beats: 
                print(f"Limiting low freq beats from {len(low_beat_times)} to {max_beats}")
                low_beat_times = low_beat_times[:max_beats]
                
            audio_data = self.get_audio_data(audio_path)

            if output_mode == "Direct Video Output":
                output_path = self.write_video_directly(
                    video_folder, audio_path, width, height, fps, 
                    max_beats, filename_prefix, low_beat_times, high_beat_times, effect_params
                )
                
                if output_path: 
                    print(f"\n=== VIDEO SAVED ===\n{output_path}\n")
                    dummy_f = Image.new('RGB', (width, height))
                    return (self.p2t(dummy_f), audio_data)
                else: 
                    print("Direct video writing failed. Falling back to frame preview.")
                    output_mode = "Frames for Editing"

            if output_mode == "Frames for Editing":
                audio_duration = self.get_audio_duration(audio_path, sr)
                if audio_duration <= 0:
                    audio_duration = 10.0
                
                safe_default_frames = 300
                preview_max_secs = 15.0
                
                if max_frames > 0: 
                    total_frames = min(max_frames, int(audio_duration * fps))
                else: 
                    total_frames = min(safe_default_frames, int(min(audio_duration, preview_max_secs) * fps))
                    
                print(f"Generating {total_frames} frames for preview at {fps} FPS.")
                
                # Sort or randomize based on the random_video_order parameter
                if effect_params.get('random_video_order', True):
                    random.shuffle(video_files)
                else:
                    # Sort alphabetically for sequential order
                    video_files.sort()
                    
                video_caps = []
                for vf in video_files:
                    try:
                        cap = cv2.VideoCapture(vf)
                        if cap.isOpened():
                            v_fps = cap.get(cv2.CAP_PROP_FPS)
                            v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            v_dur = v_frames/v_fps if v_fps > 0 else 0
                            
                            if v_dur > 0.1: 
                                video_caps.append({
                                    'cap': cap,
                                    'path': vf,
                                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                    'fps': v_fps,
                                    'frame_count': v_frames,
                                    'duration': v_dur
                                })
                            else:
                                cap.release()
                    except Exception as e:
                        print(f"Warn: Error opening {vf} for preview: {e}")
                    
                if not video_caps: 
                    print("Could not open valid videos for preview.")
                    dummy_f = Image.new('RGB', (width, height))
                    return (self.p2t(dummy_f), audio_data)
                    
                pbar.update(30)
                frames_pil = []
                prev_raw_frame = None
                prev_processed_frame = None
                current_video_idx = 0
                last_low_beat_switch_time = -1.0
                
                # Calculate sync offset for scene changes (in seconds)
                sync_offset_sec = effect_params.get('sync_offset_ms', 0) / 1000.0

                for frame_idx in range(total_frames):
                    progress = int(30 + 60 * (frame_idx + 1) / total_frames)
                    pbar.update_absolute(progress)
                    current_time = frame_idx / fps
                    switched_video = False
                    
                    # Apply sync offset to scene changes too
                    adjusted_time = current_time - sync_offset_sec
                    
                    for beat_time in low_beat_times:
                        time_diff = abs(adjusted_time - beat_time)
                        frame_time_window = 1.0 / fps  # One frame duration
                        
                        if time_diff < frame_time_window and (adjusted_time - last_low_beat_switch_time) > 0.4:
                            current_video_idx = (current_video_idx + 1) % len(video_caps)
                            last_low_beat_switch_time = adjusted_time
                            switched_video = True
                            break
                            
                    video = video_caps[current_video_idx]
                    time_in_video = current_time % video['duration'] if video['duration'] > 0 else 0
                    current_raw_frame = self.get_frame_at_time(video['cap'], time_in_video)
                    
                    if current_raw_frame.shape[1] != width or current_raw_frame.shape[0] != height: 
                        current_raw_frame = cv2.resize(current_raw_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        
                    if frame_idx == 0: 
                        prev_raw_frame = current_raw_frame.copy()
                        prev_processed_frame = current_raw_frame.copy()

                    modified_frame = self._apply_effects(
                        current_raw_frame,
                        prev_raw_frame,
                        prev_processed_frame,
                        current_time,
                        low_beat_times,
                        high_beat_times,
                        effect_params,
                        fps
                    )
                    
                    frame_rgb = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames_pil.append(pil_image)
                    
                    prev_raw_frame = current_raw_frame.copy()
                    prev_processed_frame = modified_frame.copy()

                [vc['cap'].release() for vc in video_caps]
                pbar.update(95)
                
                if frames_pil:
                    try: 
                        frames_tensors = [self.p2t(f) for f in frames_pil]
                        result_tensor = torch.cat(frames_tensors, dim=0)
                        print(f"Final preview tensor shape: {result_tensor.shape}")
                    except RuntimeError as e: 
                        print(f"WARN: Memory error creating preview tensor batch: {e}. Returning first frame.")
                        result_tensor = self.p2t(frames_pil[0]) if frames_pil else self.p2t(Image.new('RGB', (width, height)))
                else: 
                    print("Warning: No frames generated for preview.")
                    result_tensor = self.p2t(Image.new('RGB', (width, height)))
                    
                pbar.update(100)
                return (result_tensor, audio_data)

        except Exception as e:
            print(f"\n--- ERROR in FrequencyBeatSyncNode ---")
            print(f"Error Type: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("-------------------------------------\n")
            dummy_f = Image.new('RGB', (width, height))
            audio_data = self.get_audio_data(audio_path)
            return (self.p2t(dummy_f), audio_data)


# Register the node
NODE_CLASS_MAPPINGS = { "FrequencyBeatSyncNode": FrequencyBeatSyncNode }
NODE_DISPLAY_NAME_MAPPINGS = { "FrequencyBeatSyncNode": "Beat Sync (Advanced)" }
