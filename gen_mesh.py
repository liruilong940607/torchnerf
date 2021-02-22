import os
import functools

from absl import app
from absl import flags
import mcubes

import numpy as np
import torch

from torchnerf.nerf import models
from torchnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "reso",
    "300 300 300",
    "Marching cube resolution in each dimension: x y z",
)
flags.DEFINE_string(
    "c1",
    "-2 -2 -2",
    "Marching cubes bounds lower corner 1 in x y z OR single number",
)
flags.DEFINE_string(
    "c2",
    "2 2 2",
    "Marching cubes bounds upper corner in x y z OR single number",
)
flags.DEFINE_float(
    "iso", 6.0, "Marching cubes isosurface"
)
flags.DEFINE_bool(
    "coarse",
    False,
    "Force use corase network (else depends on renderer n_fine in conf)",
)
flags.DEFINE_integer(
    "point_chunk",
    720720,
    "Chunk (batch) size of points for evaluation. NOTE: --chunk will be ignored",
)
# TODO: implement color
#  flags.DEFINE_bool(
#      "color",
#      False,
#      "Generate colored mesh."
#  )

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

def marching_cubes(
    fn,
    c1,
    c2,
    reso,
    isosurface,
    chunk,
):
    """
    Run marching cubes on network. Uses PyMCubes.
    Args:
      fn main NeRF type network
      c1: list corner 1 of marching cube bounds x,y,z
      c2: list corner 2 of marching cube bounds x,y,z (all > c1)
      reso: list resolutions of marching cubes x,y,z
      isosurface: float sigma-isosurface of marching cubes
    """
    grid = np.vstack(
        np.meshgrid(
            *(np.linspace(lo, hi, sz, dtype=np.float32)
                for lo, hi, sz in zip(c1, c2, reso)),
            indexing="ij"
        )
    ).reshape(3, -1).T
    grid = torch.from_numpy(grid).to(device)

    print("* Evaluating sigma @", grid.shape[0], "points")
    rgbs, sigmas = utils.eval_points(
        fn,
        grid,
        chunk,
    )
    sigmas = sigmas.reshape(*reso)
    del rgbs

    print("* Running marching cubes")
    vertices, triangles = mcubes.marching_cubes(sigmas, isosurface)
    # Scale
    c1, c2 = np.array(c1), np.array(c2)
    vertices *= (c2 - c1) / np.array(reso)

    return vertices + c1, triangles


def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for idx, v in enumerate(vertices):
            c = vert_rgb[idx]
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], c[0], c[1], c[2])
            )
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def main(unused_argv):
    utils.set_random_seed(20210222)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_data=False)

    reso = list(map(int, FLAGS.reso.split()))
    if len(reso) == 1:
        reso *= 3
    c1 = list(map(float, FLAGS.c1.split()))
    if len(c1) == 1:
        c1 *= 3
    c2 = list(map(float, FLAGS.c2.split()))
    if len(c2) == 1:
        c2 *= 3

    print('* Creating model')
    model, state = models.get_model_state(FLAGS, device=device, restore=True)
    print('* Eval reso', FLAGS.reso, 'coarse?', FLAGS.coarse)

    eval_points_pfn = utils.get_eval_points_pfn(model, raw_rgb=True,
            coarse=FLAGS.coarse)

    verts, faces = marching_cubes(
        eval_points_pfn,
        c1=c1, c2=c2, reso=reso, isosurface=FLAGS.iso, chunk=FLAGS.point_chunk
    )

    mesh_path = os.path.join(FLAGS.train_dir, 'mesh.obj')
    print(' Saving to', mesh_path)
    save_obj(verts, faces, mesh_path)


if __name__ == "__main__":
    app.run(main)