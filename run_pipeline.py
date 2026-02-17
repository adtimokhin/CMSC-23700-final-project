"""
Main entry point: builds the node pipeline, runs it, then launches Blender to render.

Usage:
    python run_pipeline.py

After rendering, run save_video.py to stitch frames into an MP4.
"""

import os
import subprocess

from nodes.audio import AudioInputNode, AudioAnalysisNode
from nodes.smoothing import SmoothingNode
from nodes.transform import ObjectTransformNode
from nodes.material import MaterialNode
from nodes.export import ExportNode
from pipeline.pipeline import Pipeline

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Configuration ---------- #
# Update these paths for your setup
BLENDER = "/Applications/Blender.app/Contents/MacOS/blender"
AUDIO_FILE = os.path.join(PROJECT_DIR, "audio", "short.mp3")  # put your MP3 here
FPS = 10  # low fps for fast iteration; increase for final render

# ---------- Build the pipeline ---------- #

audio = AudioInputNode(filepath=AUDIO_FILE, fps=FPS, start=60, end=64)

analysis = AudioAnalysisNode()

smooth = SmoothingNode(
    fields=["audio_bass", "audio_volume"],
    n_control_points=30,
    degree=3,
)

# Sphere moves up/down based on volume
sphere_transform = ObjectTransformNode(
    obj_name="sphere",
    obj_file=os.path.join(PROJECT_DIR, "meshes", "scene-sphere.obj"),
    mapping={
        "location_z": {"source": "audio_volume", "range": [0.0, 1.5]},
    },
    base_location=(0, 0, 0.75),
)

# Color driven by bass: blue when quiet, bright cyan/white when bass hits
sphere_material = MaterialNode(
    obj_name="sphere",
    source="audio_bass",
    color_low=(0.05, 0.1, 0.4),   # dark blue when quiet
    color_high=(0.2, 0.8, 1.0),   # bright cyan when bass hits
)

manifest_dir = os.path.join(PROJECT_DIR, "output", "manifests")
export = ExportNode(output_dir=manifest_dir)

# ---------- Chain nodes ---------- #
audio >> analysis >> smooth >> sphere_transform >> sphere_material >> export

# ---------- Run pipeline ---------- #
pipeline = Pipeline(head=audio)
data = pipeline.run()

print(f"\nPipeline complete: {data['n_frames']} frames at {data['fps']} fps")
print(f"Duration: {data['duration']:.1f}s")
print(f"Manifest saved to: {manifest_dir}")

# ---------- Launch Blender render ---------- #
# Clean old frames so save_video doesn't mix them with new ones
import glob
animation_renders_dir = os.path.join(PROJECT_DIR, "output", "animation_renders")
for old_frame in glob.glob(os.path.join(animation_renders_dir, "*.png")):
    os.remove(old_frame)

blend_file = os.path.join(PROJECT_DIR, "blank.blend")
render_script = os.path.join(PROJECT_DIR, "render_from_manifest.py")

print(f"\nLaunching Blender render...")
subprocess.run([
    BLENDER, blend_file, "--background", "--python", render_script, "--", manifest_dir
])

# ---------- Stitch video ---------- #
from save_video import save_video

animation_renders_dir = os.path.join(PROJECT_DIR, "output", "animation_renders")
os.makedirs(os.path.join(PROJECT_DIR, "output", "animations"), exist_ok=True)

print(f"\nStitching video...")
save_video(animation_renders_dir, fps=FPS, background_color=(0, 0, 0))
print("Done! Video saved to output/animations/animation.mp4")
