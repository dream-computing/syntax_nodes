import os
import sys
import subprocess

print('''
███████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█                                                                                                                 █
█ ___  _   _  _ __  | |_   __ _ __  __  __| |(_) / _| / _| _   _  ___ (_)  ___   _ __       ___   ___   _ __ ___  █
█/ __|| | | || '_ \ | __| / _` |\ \/ / / _` || || |_ | |_ | | | |/ __|| | / _ \ | '_ \     / __| / _ \ | '_ ` _ \ █
█\__ \| |_| || | | || |_ | (_| | >  < | (_| || ||  _||  _|| |_| |\__ \| || (_) || | | | _ | (__ | (_) || | | | | |█
█|___/ \__, ||_| |_| \__| \__,_|/_/\_\ \__,_||_||_|  |_|   \__,_||___/|_| \___/ |_| |_|(_) \___| \___/ |_| |_| |_|█
█      |___/                                                                                                      █
█                                                                                                                 █
███████████████████████████████████████████████████████████████████████████████████████████████████████████████████

''')


# Import node modules
from .jigsaw_puzzle_node import JigsawPuzzleNode
from .low_poly_node import LowPolyNode
from .region_boundary_node import RegionBoundaryNode
from .pointillism import PointillismNode
from .frequency_beat_sync import FrequencyBeatSyncNode
from .ghosting_afterimage_node import GhostingNode
from .depth_to_lidar_effect_node import DepthToLidarEffectNode
from .LuminanceParticleNode import LuminanceParticleNode # Assuming previous particle node file exists
from .edge_measurement_overlay_node import EdgeMeasurementOverlayNode
from .edge_tracing_node import EdgeTracingNode
from .variable_line_width_effect_node import VariableLineWidthEffectNode
from .cyberpunk_window_node import CyberpunkWindowNode
from .cyberpunk_magnify_node import CyberpunkMagnifyNode
from .rgb_streak_node import RGBStreakNode
from .voxel_node import VoxelNode
from .papercraftnode import PaperCraftNode
from .frequency_beat_sync_advanced import FrequencyBeatSyncNode as FrequencyBeatSyncNodeAdvanced

# Map the node classes for ComfyUI to recognize them
NODE_CLASS_MAPPINGS = {
    "JigsawPuzzleNode": JigsawPuzzleNode,
    "LowPolyNode": LowPolyNode,
    "RegionBoundaryNode": RegionBoundaryNode,
    "PointillismNode": PointillismNode,
    "FrequencyBeatSyncNode": FrequencyBeatSyncNode,
    "GhostingNode": GhostingNode,
    "DepthToLidarEffectNode": DepthToLidarEffectNode,
    "LuminanceParticleNode": LuminanceParticleNode,
    "EdgeMeasurementOverlayNode": EdgeMeasurementOverlayNode,
    "EdgeTracingNode": EdgeTracingNode,
    "VariableLineWidthEffectNode": VariableLineWidthEffectNode,
    "CyberpunkWindowNode": CyberpunkWindowNode,
    "CyberpunkMagnifyNode": CyberpunkMagnifyNode,
    "RGBStreakNode": RGBStreakNode,
    "VoxelNode": VoxelNode,
    "PaperCraftNode": PaperCraftNode,
    "FrequencyBeatSyncNodeAdvanced": FrequencyBeatSyncNodeAdvanced,
}

# Provide user-friendly display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawPuzzleNode": "Jigsaw Puzzle Effect",
    "LowPolyNode": "Low Poly Image Processor",
    "RegionBoundaryNode": "Region Boundary Node",
    "PointillismNode": "Pointillism Effect",
    "FrequencyBeatSyncNode": "Beat Sync",
    "GhostingNode": "Ghosting/Afterimage Effect",
    "DepthToLidarEffectNode": "Depth to LIDAR Effect",
    "LuminanceParticleNode": "Luminance Particle Effect", 
    "EdgeMeasurementOverlayNode": "Edge Measurement Overlay",
    "EdgeTracingNode": "Edge Tracing Animation",
    "VariableLineWidthEffectNode": "Variable Line Width Effect",
    "CyberpunkWindowNode": "Cyberpunk Window Effect",
    "CyberpunkMagnifyNode": "Cyberpunk Magnify Effect",
    "RGBStreakNode": "RGB Streak Effect",
    "VoxelNode": "Voxel Block Effect",
    "PaperCraftNode": "Paper Craft Effect",
    "FrequencyBeatSyncNodeAdvanced": "Beat Sync (Advanced)",
}




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- Syntax Nodes: Custom Nodes Loaded ---")
