import os
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler
import sys
from pathlib import Path

# Example calling this file from the command line:
# Replace with your path to blender and the path to this file
# On Mac, it will be something like:
# /Applications/Blender.app/Contents/MacOS/blender /Users/your-user-name/Desktop/CMSC/CMSC23700/graphics-2025/final-proj/blank.blend --background --python blender_sample.py

# And on windows, it will be something like:
# C:\Program Files\Blender Foundation\Blender\blender.exe C:\Users\your-user-name\Desktop\CMSC\CMSC23700\graphics-2025\final-proj\blank.blend --background --python blender_sample.py


# NOTE:
# If you run this code from same directory that blank.blend is in, you only need to run:
# /Applications/Blender.app/Contents/MacOS/blender blank.blend --background --python blender_sample.py


class TriangleMesh:
    def __init__(self, obj_file):
        # render mesh with/without edges (True for with edges)
        self.use_freestyle = False
        bpy.data.scenes["Scene"].render.use_freestyle = self.use_freestyle
        if self.use_freestyle:
            bpy.data.linestyles["LineStyle"].thickness = 0.8  # line thickness
        bpy.data.scenes["Scene"].use_nodes = True

        # load in the mesh object
        self.mesh = self.load_obj(obj_file)

        # initialize mesh postion, rotation and material
        self.mesh_settings(self.mesh)

    def load_obj(self, obj_file):
        bpy.ops.wm.obj_import(filepath=obj_file)
        ob = bpy.context.selected_objects[0]
        if self.use_freestyle:
            self.__mark_all_edges_freestyle(ob)
        return ob

    def __mark_all_edges_freestyle(self, mesh):
        for edge in mesh.data.edges:
            edge.use_freestyle_mark = True

    def mesh_settings(self, mesh):
        # smooth shading, comment out if you have a low poly mesh
        # or want to see the faces more clearly
        bpy.ops.object.shade_smooth()

        # apply scale
        mesh.scale = [1, 1, 1]

        # apply rotations in radians
        mesh.rotation_euler.x = 1.57
        mesh.rotation_euler.y = 0
        mesh.rotation_euler.z = -1.65

        # shift bottom vertex to sit on zero (puts the mesh on the ground plane, you may remove if you want)
        mesh.data.update()
        bpy.context.view_layer.update()
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
        vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])
        mesh.location = Vector((0, 0, mesh.location.z - min(vertices[:, 2])))

        # further translate mesh position
        mesh.location.x += 0
        mesh.location.y += 0
        mesh.location.z += 0

        # hard code color and alpha values
        r, g, b, a = 1, 0, 0, 1

        # now add material
        mat = bpy.data.materials.new(name="obj_material")
        mat.use_nodes = True

        # this wipes out any materials that come with the .obj
        mesh.data.materials.clear()

        # the principled shader is the default shader in blender
        # and the most customizable
        principled = mat.node_tree.nodes["Principled BSDF"]
        # here, we set just the base color
        principled.inputs["Base Color"].default_value = (r, g, b, a)

        # you can also set other properties like roughness, metallic, etc.
        # principled.inputs["Roughness"].default_value = 0.5
        # principled.inputs["Metallic"].default_value = 0.5

        # There are also default shaders for emission, glass, etc.
        # NOTE: you cannot use the principled shader and these shaders on the same mesh
        # comment out or delete the principled definition

        # GLOWING/EMISSION MATERIAL
        # emission = mat.node_tree.nodes["Emission"]
        # emission.inputs["Color"].default_value = (r, g, b, a)
        # emission.inputs["Strength"].default_value = 1.0

        # GLASS MATERIAL
        # glass = mat.node_tree.nodes["Glass BSDF"]
        # glass.inputs["Roughness"].default_value = 0
        # glass.inputs["IOR"].default_value = 1.5

        # update material
        mesh.data.materials.append(mat)


class Scene:
    def __init__(self):
        self.setup_render_engine()
        self.cam = self.setup_camera()

    def setup_render_engine(self):
        # CPU is faster on most laptops, but if you have a GPU, change this to GPU
        bpy.context.scene.cycles.device = "CPU"

        # we default to 10 samples at 50% resolution for faster rendering
        # feel free to increase this for higher quality, though it will take much longer per frame
        bpy.context.scene.cycles.samples = 10
        bpy.context.scene.render.resolution_percentage = 50

        # denoising is inexpensive and improves quality
        # feel free to remove if you want to see the noise
        bpy.context.scene.cycles.use_denoising = True

    def setup_camera(self):
        # camera settings
        # this camera is the only thing in the blank.blend file
        cam = bpy.data.scenes["Scene"].objects["Camera"]
        cam.rotation_euler.x = 1.36
        cam.rotation_euler.y = 0
        cam.rotation_euler.z = 1.157
        cam.location.x = 6.6
        cam.location.y = -3
        cam.location.z = 2.49

        return cam

    def add_lights(self):
        # lighting
        # feel free to adjust the light settings or add more lights
        # we will be using an area light
        lights = []
        bpy.ops.object.light_add(
            type="AREA", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        # but you can also add other types of lights, like point lights
        # bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        light = bpy.context.object
        light.data.energy = 1000
        light.data.color = (0.8, 0.77, 0.8)
        light.data.size = 10
        light.location.x = 6
        light.location.z = 4.5
        light.rotation_euler.y = 0.436

        lights.append(light)
        return lights

    def add_plane(self):
        # add the ground plane
        bpy.ops.mesh.primitive_plane_add(
            enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        plane = bpy.context.object
        plane.scale.x = 100
        plane.scale.y = 100

        # this makes the plane invisible, but it will still catch shadows
        # feel free to remove if you want to see the plane, or even assign a material to it
        plane.is_shadow_catcher = True

        # subdivide the plane (this fixes the freestyle lines)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=1000)
        bpy.ops.object.mode_set(mode="OBJECT")
        plane.data.update()

        # if you want to add a material to the plane, uncomment the following lines
        # and set the shadow catcher to false
        # mat = bpy.data.materials.new(name="plane_material")
        # mat.use_nodes = True
        # principled = mat.node_tree.nodes["Principled BSDF"]
        # r, g, b, a = 0, 0, 1, 1
        # principled.inputs["Base Color"].default_value = (r, g, b, a)
        # plane.data.materials.append(mat)

        return plane


def select_obj(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def setup_animation_keyframes(object_to_animate, locations, rotations, scales, n_frames):
    """
    insert keyframes along path
    """
    select_obj(object_to_animate)
    bpy.data.scenes["Scene"].frame_end = int(n_frames - 1)
    f = 1

    for loc, rot, scale in zip(locations, rotations, scales):
        print(loc, rot, scale)
        # put object at location
        object_to_animate.location[0] = loc[0]
        object_to_animate.location[1] = loc[1]
        object_to_animate.location[2] = loc[2]

        # rotations
        object_to_animate.rotation_euler[0] = rot[0]
        object_to_animate.rotation_euler[1] = rot[1]
        object_to_animate.rotation_euler[2] = rot[2]

        # scale
        object_to_animate.scale[0] = scale[0]
        object_to_animate.scale[1] = scale[1]
        object_to_animate.scale[2] = scale[2]

        bpy.data.scenes["Scene"].frame_current = f
        object_to_animate.keyframe_insert(data_path="location", frame=f)
        object_to_animate.keyframe_insert(data_path="rotation_euler", frame=f)
        object_to_animate.keyframe_insert(data_path="scale", frame=f)
        f += 1


# ---------- Load objs ---------- #
# set obj file to spot.obj in the meshes directory
# we've also included some other .obj files, but feel free to download or create your own
obj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "scene-sphere.obj")
print(f"obj file: {obj_file}")


# ---------- Scene Setup ---------- #
# set scene parameters and camera
scene = Scene()
cam = scene.cam

# set up lights and plane
lights = scene.add_lights()
light = lights[0]
plane = scene.add_plane()

# load in obj file and get mesh object
mesh = TriangleMesh(obj_file)
mesh = mesh.mesh

# ---------- Render Mesh ---------- #
# setup output file path
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "renders", "sphere.png"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"output path: {output_path}")
# edit output path as needed
bpy.data.scenes["Scene"].render.filepath = output_path

# debug flag
debug = False


# ---------- Animation ---------- #
# animate flag
animate = True

# hard code locations
n_frames = 20

# example animations (moving either cam or mesh)
# example camera animation
# locations = zip([6.6]*n_frames, np.linspace(-6, 3, n_frames), [2.49]*n_frames)
# rotations = zip([1.36]*n_frames, [0]*n_frames, [1.157]*n_frames)

# example mesh animation
# this uses np.linspace to generate linear animations
# but you can replace this with your splines!
rotations = zip([1.36] * n_frames, [0] * n_frames, np.linspace(1.157, 5, n_frames))
locations = zip(
    [0] * n_frames, np.linspace(0, -2, n_frames), np.linspace(0.7521, 1.5, n_frames)
)
scales = zip(
    np.linspace(0.5, 1.5, n_frames),
    np.linspace(0.5, 1.5, n_frames),
    np.linspace(0.5, 1.5, n_frames),
)


if animate:
    # setup_animation_keyframes(cam, locations, rotations, scales, n_frames) # for camera
    setup_animation_keyframes(mesh, locations, rotations, scales, n_frames)  # for mesh
    Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "animation_renders"
        )
    ).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        "animation_renders",
        "sphere.png",
    )  # edit output path as needed
    bpy.data.scenes["Scene"].render.filepath = output_path

# render
if not debug:
    bpy.ops.render.render(write_still=not animate, animation=animate)
else:
    print("debugging...")
    bpy.ops.wm.save_as_mainfile(filepath="/tmp/debug.blend")
    # To debug, run the following code. This will open up the debug file in blender.
    # path/to/blender /tmp/debug.blend
    # For me this looks like:
    # /Applications/Blender.app/Contents/MacOS/blender /tmp/debug.blend
