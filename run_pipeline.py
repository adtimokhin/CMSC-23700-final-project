"""
Main entry point: builds the node pipeline, runs it, then launches Blender to render.

Usage:
    python run_pipeline.py

After rendering, run save_video.py to stitch frames into an MP4.
"""

import os

from nodes.audio import AudioInputNode, AudioAnalysisNode
from nodes.smoothing import SmoothingNode
from nodes.transform import ObjectTransformNode
from nodes.material import MaterialNode
from nodes.noise_displacement import NoiseDisplacementNode
from nodes.export import ExportNode
from pipeline.pipeline import Pipeline

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Configuration ---------- #
# Update these paths for your setup
BLENDER = "/Applications/Blender.app/Contents/MacOS/blender"
AUDIO_FILE = os.path.join(PROJECT_DIR, "audio", "short.mp3")  # put your MP3 here
FPS = 60  # low fps for fast iteration; increase for final render

################################################################################
#                                   Nodes
################################################################################

audio = AudioInputNode(filepath=AUDIO_FILE, fps=FPS, start=62, end=63)
analysis = AudioAnalysisNode()
smooth = SmoothingNode(
    fields=["audio_bass", "audio_volume"],
    n_control_points=30,
    degree=3,
)
sphere_transform = ObjectTransformNode(
    obj_name="sphere",
    obj_file=os.path.join(PROJECT_DIR, "meshes", "scene-sphere.obj"),
    mapping={},
    base_location=(0, 0, 0),
)
sphere_noise = NoiseDisplacementNode(
    obj_name="sphere",
    source="audio_volume",
    amplitude=0.9,
    noise_scale=2.0,
    time_speed=1.0,
    octaves=3,
)
sphere_material = MaterialNode(
    obj_name="sphere",
    source="audio_bass",
    color_low=(0.0, 1.0, 0.2),   # dark blue when quiet
    color_high=(1.0, 0.3, 0.1),   # bright cyan when bass hits
)

manifest_dir = os.path.join(PROJECT_DIR, "output", "manifests")
export = ExportNode(output_dir=manifest_dir)

################################################################################
#                                   Chain
################################################################################

audio.then(analysis).\
        then(smooth).\
        then(sphere_transform).\
        then(sphere_noise).\
        then(sphere_material).\
        then(export)

################################################################################
#                            Running the Pipeline
################################################################################

pipeline = Pipeline(head=audio)
data = pipeline.run()

print("Pipeline done. Run run_blender.py to render and stitch the video.")
