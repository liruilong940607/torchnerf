import os
import functools

from absl import app
from absl import flags

import numpy as np
import torch

from torchnerf.nerf import models
from torchnerf.nerf import utils

import imageio

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_float(
    "elevation",
    -30.0,
    "Elevation angle (negative is above)",
)
flags.DEFINE_integer(
    "num_views",
    40,
    "The number of views to generate.",
)
flags.DEFINE_integer(
    "height",
    800,
    "The size of images to generate.",
)
flags.DEFINE_integer(
    "width",
    800,
    "The size of images to generate.",
)
flags.DEFINE_float(
    "camera_angle_x",
    0.7,
    "The camera angle in rad in x direction (used to get focal length).",
    short_name='A',
)
flags.DEFINE_float(
    "radius",
    4.0,
    "Radius to origin of camera path.",
)
flags.DEFINE_integer(
    "fps",
    20,
    "FPS of generated video",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

def main(unused_argv):
    utils.set_random_seed(20210222)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_data=False)

    print('* Generating poses')
    render_poses = np.stack(
        [
            utils.pose_spherical(angle, FLAGS.elevation, FLAGS.radius)
            for angle in np.linspace(-180, 180, FLAGS.num_views + 1)[:-1]
        ],
        0,
    )  # (NV, 4, 4)

    print('* Generating rays')
    focal = 0.5 * FLAGS.width / np.tan(0.5 * FLAGS.camera_angle_x)
    rays = utils.generate_rays(FLAGS.width, FLAGS.height, focal, render_poses)

    print('* Creating model')
    model, state = models.get_model_state(FLAGS, device=device, restore=False)
    render_pfn = utils.get_render_pfn(model, randomized=False)

    print('* Rendering')

    vid_name = "e{:03}".format(int(-FLAGS.elevation * 10))
    video_dir = os.path.join(FLAGS.train_dir, 'video', vid_name)
    frames_dir = os.path.join(video_dir, 'frames')
    print(' Saving to', video_dir)
    utils.makedirs(frames_dir)

    frames = []
    for i in range(FLAGS.num_views):
        print(f'** View {i+1}/{FLAGS.num_views} = {i / FLAGS.num_views * 100}%')
        with torch.no_grad():
            rays_this_view = utils.namedtuple_map(
                lambda z: torch.from_numpy(z[i].copy()).to(device), rays)
            pred_color, pred_disp, pred_acc = utils.render_image(
                render_pfn,
                rays_this_view,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk,
            )
            pred_color = pred_color.cpu().numpy()
        utils.save_img(pred_color, os.path.join(frames_dir, f'{i:04}.png'))
        frames.append(np.array(pred_color))

    frames = np.stack(frames)
    vid_path = os.path.join(video_dir, "video.mp4")
    print('* Writing video', vid_path)
    imageio.mimwrite(
        vid_path, (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8),
        fps=FLAGS.fps, quality=8
    )
    print('* Done')

if __name__ == "__main__":
    app.run(main)