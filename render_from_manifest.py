"""
Blender-side render script. Reads a pipeline manifest and renders the animation.

Usage:
    /path/to/blender blank.blend --background --python render_from_manifest.py -- /path/to/manifest_dir
"""

import sys
import os
import json
import numpy as np
import bpy

# Add project root to path so we can import existing code
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from blender_sample import TriangleMesh, Scene, setup_animation_keyframes, select_obj


def load_manifest(manifest_dir):
    """Load the pipeline output manifest."""
    from pipeline.pipeline import Pipeline
    return Pipeline.load_manifest(manifest_dir)


def apply_vertex_displacement(blender_obj, base_vertices, offsets_per_frame, n_frames):
    """
    Apply per-frame vertex displacement using shape keys.
    Each frame gets a shape key that blends in at its frame and out at neighbors.
    """
    select_obj(blender_obj)

    # Add basis shape key (the original mesh)
    blender_obj.shape_key_add(name="Basis", from_mix=False)

    for frame_idx in range(n_frames):
        key = blender_obj.shape_key_add(name=f"frame_{frame_idx}", from_mix=False)
        offsets = offsets_per_frame[frame_idx]

        for vi, vert in enumerate(key.data):
            vert.co.x = base_vertices[vi][0] + offsets[vi][0]
            vert.co.y = base_vertices[vi][1] + offsets[vi][1]
            vert.co.z = base_vertices[vi][2] + offsets[vi][2]

        # Keyframe: this shape key is 1.0 at its frame, 0.0 at neighbors
        frame = frame_idx + 1  # Blender frames are 1-indexed
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=max(1, frame - 1))
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=min(n_frames, frame + 1))


def apply_material_animation(blender_obj, colors_per_frame, n_frames):
    """Keyframe material base color changes per frame."""
    mat = blender_obj.data.materials[0]
    principled = mat.node_tree.nodes["Principled BSDF"]

    for frame_idx in range(n_frames):
        r, g, b = colors_per_frame[frame_idx]
        principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
        principled.inputs["Base Color"].keyframe_insert(
            data_path="default_value", frame=frame_idx + 1
        )


def main():
    # Parse manifest directory from command line args (after --)
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender blank.blend --background --python render_from_manifest.py -- /path/to/manifest_dir")
        sys.exit(1)

    manifest_dir = argv[argv.index("--") + 1]
    print(f"[Render] Loading manifest from: {manifest_dir}")

    data = load_manifest(manifest_dir)
    n_frames = data["n_frames"]
    fps = data.get("fps", 24)

    # Scene setup (reuse existing)
    scene = Scene()
    lights = scene.add_lights()
    plane = scene.add_plane()

    # Set frame range and fps
    bpy.data.scenes["Scene"].frame_start = 1
    bpy.data.scenes["Scene"].frame_end = n_frames
    bpy.data.scenes["Scene"].render.fps = fps

    # Load and animate each object
    for obj_name, obj_data in data["objects"].items():
        print(f"[Render] Setting up object: {obj_name}")

        # Load mesh
        tri_mesh = TriangleMesh(obj_data["obj_file"])
        blender_obj = tri_mesh.mesh

        # Apply transform animation (reuses existing function)
        locations = obj_data["locations"]
        rotations = obj_data["rotations"]
        scales = obj_data["scales"]
        setup_animation_keyframes(blender_obj, locations, rotations, scales, n_frames)

        # Apply vertex displacement if present
        if "vertex_offsets" in obj_data and obj_data["vertex_offsets"] is not None:
            print(f"[Render] Applying vertex displacement to {obj_name}")
            apply_vertex_displacement(
                blender_obj,
                obj_data["base_vertices"],
                obj_data["vertex_offsets"],
                n_frames,
            )

        # Apply material animation if present
        if "material_colors" in obj_data:
            print(f"[Render] Applying material animation to {obj_name}")
            # Override the default red color from TriangleMesh with first frame's color
            mat = blender_obj.data.materials[0]
            principled = mat.node_tree.nodes["Principled BSDF"]
            r, g, b = obj_data["material_colors"][0]
            principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
            apply_material_animation(
                blender_obj,
                obj_data["material_colors"],
                n_frames,
            )

    # Render animation
    output_path = os.path.join(project_dir, "output", "animation_renders", "frame.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bpy.data.scenes["Scene"].render.filepath = output_path
    print(f"[Render] Rendering {n_frames} frames to {os.path.dirname(output_path)}")
    bpy.ops.render.render(animation=True)


main()
