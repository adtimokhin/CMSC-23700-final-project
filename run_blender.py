"""
Blender render script.

Reads the pipeline manifest and renders the animation.
Run run_pipeline.py first to populate output/manifests/.
After this finishes, run save_video.py to stitch frames into an MP4.
"""

import os
import sys
import glob
import bpy

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from pipeline.pipeline import Pipeline

################################################################################
#                              Config
################################################################################

FPS = 60
manifest_dir = os.path.join(project_dir, "output", "manifests")
animation_renders_dir = os.path.join(project_dir, "output", "animation_renders")

################################################################################
#                         Load manifest
################################################################################

data = Pipeline.load_manifest(manifest_dir)
n_frames = data["n_frames"]
fps = data.get("fps", FPS)

################################################################################
#                         Scene / render setup
################################################################################

# Render engine
bpy.context.scene.cycles.device = "CPU"
bpy.context.scene.cycles.samples = 10
bpy.context.scene.render.resolution_percentage = 50
bpy.context.scene.cycles.use_denoising = True

# Camera
cam = bpy.data.scenes["Scene"].objects["Camera"]
cam.location.x = 8
cam.location.y = 0
cam.location.z = 1
cam.rotation_euler.x = 1.5208
cam.rotation_euler.y = 0
cam.rotation_euler.z = 1.5708

# Frame range
bpy.data.scenes["Scene"].frame_start = 1
bpy.data.scenes["Scene"].frame_end = n_frames
bpy.data.scenes["Scene"].render.fps = fps

################################################################################
#                         Helpers
################################################################################

def load_mesh(obj_file):
    bpy.ops.wm.obj_import(filepath=obj_file)
    obj = bpy.context.selected_objects[0]

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Scale
    obj.scale = [1, 1, 1]

    # Rotation
    obj.rotation_euler.x = 1.57
    obj.rotation_euler.y = 0
    obj.rotation_euler.z = -1.65

    # Shift bottom vertex to z = 0
    obj.data.update()
    bpy.context.view_layer.update()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    vertices = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
    min_z = min(v.z for v in vertices)
    obj.location.z -= min_z

    obj.location.x += 0
    obj.location.y += 0
    obj.location.z += 0

    # Material
    mat = bpy.data.materials.new(name="obj_material")
    mat.use_nodes = True
    obj.data.materials.clear()
    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (1, 0, 0, 1)
    obj.data.materials.append(mat)

    return obj


def select_obj(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def setup_animation_keyframes(obj, locations, rotations, scales, n_frames):
    select_obj(obj)
    bpy.data.scenes["Scene"].frame_end = int(n_frames - 1)
    f = 1
    for loc, rot, scale in zip(locations, rotations, scales):
        obj.location[0] = loc[0]
        obj.location[1] = loc[1]
        obj.location[2] = loc[2]

        obj.rotation_euler[0] = rot[0]
        obj.rotation_euler[1] = rot[1]
        obj.rotation_euler[2] = rot[2]

        obj.scale[0] = scale[0]
        obj.scale[1] = scale[1]
        obj.scale[2] = scale[2]

        bpy.data.scenes["Scene"].frame_current = f
        obj.keyframe_insert(data_path="location", frame=f)
        obj.keyframe_insert(data_path="rotation_euler", frame=f)
        obj.keyframe_insert(data_path="scale", frame=f)
        f += 1


def apply_vertex_displacement(obj, base_vertices, offsets_per_frame, n_frames):
    select_obj(obj)
    obj.shape_key_add(name="Basis", from_mix=False)

    for frame_idx in range(n_frames):
        key = obj.shape_key_add(name=f"frame_{frame_idx}", from_mix=False)
        offsets = offsets_per_frame[frame_idx]

        for vi, vert in enumerate(key.data):
            vert.co.x = base_vertices[vi][0] + offsets[vi][0]
            vert.co.y = base_vertices[vi][1] + offsets[vi][1]
            vert.co.z = base_vertices[vi][2] + offsets[vi][2]

        frame = frame_idx + 1
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=max(1, frame - 1))
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=min(n_frames, frame + 1))


def apply_material_animation(obj, colors_per_frame, n_frames):
    mat = obj.data.materials[0]
    principled = mat.node_tree.nodes["Principled BSDF"]

    for frame_idx in range(n_frames):
        r, g, b = colors_per_frame[frame_idx]
        principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
        principled.inputs["Base Color"].keyframe_insert(
            data_path="default_value", frame=frame_idx + 1
        )

################################################################################
#                         Per-object animation
################################################################################

for _, obj_data in data["objects"].items():
    blender_obj = load_mesh(obj_data["obj_file"])

    setup_animation_keyframes(
        blender_obj,
        obj_data["locations"],
        obj_data["rotations"],
        obj_data["scales"],
        n_frames,
    )

    if "vertex_offsets" in obj_data and obj_data["vertex_offsets"] is not None:
        apply_vertex_displacement(
            blender_obj,
            obj_data["base_vertices"],
            obj_data["vertex_offsets"],
            n_frames,
        )

    if "material_colors" in obj_data:
        mat = blender_obj.data.materials[0]
        principled = mat.node_tree.nodes["Principled BSDF"]
        r, g, b = obj_data["material_colors"][0]
        principled.inputs["Base Color"].default_value = (r, g, b, 1.0)
        apply_material_animation(blender_obj, obj_data["material_colors"], n_frames)

################################################################################
#                              Render
################################################################################

# Clean old frames so save_video doesn't mix them with new ones
os.makedirs(animation_renders_dir, exist_ok=True)
for old_frame in glob.glob(os.path.join(animation_renders_dir, "*.png")):
    os.remove(old_frame)

output_path = os.path.join(animation_renders_dir, "frame.png")
bpy.data.scenes["Scene"].render.filepath = output_path
bpy.ops.render.render(animation=True)
