import os
import numpy as np
from pathlib import Path

# pip install imageio
# pip install imageio-ffmpeg
import imageio
from PIL import Image


def save_video(path, fps, background_color):
    # save a video using of all images in a folder
    file_paths = Path(path).glob("*.png")
    # imgs is a list of numpy array images
    imgs = []
    for file_path in sorted(file_paths):
        print(file_path)
        img = Image.open(file_path)
        img_arr = np.asarray(img)

        if background_color:
            background = Image.new("RGB", img.size, background_color)
            background.paste(img, (0, 0), img)
            img_arr = np.asarray(background)
        imgs.append(img_arr)

    write_to = "output/animations/{}.mp4".format(
        "animation"
    )  # have a folder of output where output files could be stored.
    writer = imageio.get_writer(write_to, format="mp4", mode="I", fps=fps)

    for img in imgs:
        writer.append_data(img[:])
    writer.close()


# Here is an example which saves images in the animation_renders folder to a video file at 10 fps with a white background
fps = 10
background_color = (255, 255, 255)

save_video(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/animation_renders"),
    fps,
    background_color,
)
