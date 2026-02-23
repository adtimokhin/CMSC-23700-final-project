# Final Project — Audio-Driven Animation Pipeline

Generates an audio-reactive 3D animation using a node-based pipeline and Blender.
The process runs in three steps.

## Step 1 — Run the pipeline (`run_pipeline.py`)

```bash
python run_pipeline.py
```

Run with **regular Python**. Builds the node graph, processes the audio file, and
writes per-frame animation data (locations, rotations, vertex offsets, material
colors, etc.) to `output/manifests/`.

Nothing is rendered here — this step only produces the instructions that Blender
will consume.


## Step 2 — Render frames in Blender (`run_blender.py`)

```bash
/Applications/Blender.app/Contents/MacOS/blender blank.blend --background --python run_blender.py
```

Run with **Blender's Python**. Loads the manifest from `output/manifests/`,
builds the scene, applies per-frame animation keyframes and vertex shape keys,
and renders every frame as a PNG into `output/animation_renders/`.



## Step 3 — Stitch the video (`save_video.py`)

```bash
python save_video.py
```

Run with **regular Python**. Reads all PNGs from `output/animation_renders/`,
composites them onto a background colour, and writes the final video to
`output/animations/animation.mp4`.
