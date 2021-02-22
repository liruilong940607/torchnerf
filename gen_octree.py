import os
import functools
from tqdm import tqdm
from absl import app
from absl import flags

import numpy as np
import torch

from torchnerf.nerf import models
from torchnerf.nerf import utils

from torchnerf.nerf.svox import N3Tree

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "output",
    "torchnerf/out.npz",
    "Output file",
)

flags.DEFINE_string(
    "center",
    "0 0 0",
    "Center of volume in x y z OR single number",
)
# TODO implement different radius for each dimension
flags.DEFINE_float(
    "radius",
    1.2,
    #  1.5,
    "1/2 side length of volume",
)
flags.DEFINE_float(
    "refine_thresh",
    0.01,
    #  0.05,
    "Absolute threshold to consider refining a leaf voxel",
)
flags.DEFINE_float(
    "step2_refine_thresh",
    0.4,
    "Sigma difference threshold to consider refining a leaf voxel",
)
flags.DEFINE_integer(
    "step2_n_iter",
    1,
    "Number of step2 iterations",
)
flags.DEFINE_float(
    "max_refine_prop",
    0.5,
    "Max proportion of cells to refine",
)
flags.DEFINE_integer(
    "tree_branch_n",
    2,
    "Tree branch factor (2=octree)",
)
flags.DEFINE_integer(
    "init_grid_depth",
    8,
    #  7,
    "Initial evaluation grid (2^{x+1} voxel grid)",
)

flags.DEFINE_bool(
    "coarse",
    False,
    "Force use corase network (else depends on renderer n_fine in conf)",
)
flags.DEFINE_integer(
    "point_chunk",
    720720,
    "Chunk (batch) size of points for evaluation",
)
flags.DEFINE_integer(
    "samples_per_cell",
    8,
    "Samples per cell",
    short_name='S',
)
# TODO: implement color
#  flags.DEFINE_bool(
#      "color",
#      False,
#      "Generate colored mesh."
#  )

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
def step1(tree, eval_points_fpn):
    print('* Step 1: Grid eval')
    reso = 2 ** (FLAGS.init_grid_depth + 1)
    arr = (np.arange(0, reso, dtype=np.float32) + 0.5) / reso
    grid = np.vstack(
        np.meshgrid(
            *[(arr - tree.offset[i]) / tree.invradius for i in range(3)],
            indexing="ij")
    ).reshape(3, -1).T

    rgb, sigma = utils.eval_points(
        eval_points_fpn, 
        torch.from_numpy(grid).float().to(device), 
        FLAGS.point_chunk, 
        to_cpu=True)
    sigma = np.nan_to_num(np.maximum(sigma, 0.0))
    approx_alpha = 1.0 - np.exp(- 2.0 * sigma.reshape(-1) / reso)
    mask = approx_alpha >= FLAGS.refine_thresh
    grid, rgb, sigma = grid[mask], rgb[mask], sigma[mask]

    print(' Building octree')
    for i in tqdm(range(FLAGS.init_grid_depth)):
        tree[grid].refine()

    if FLAGS.step2_n_iter == 0:
        tree.set(grid, np.concatenate([rgb, sigma], axis=-1))
    assert tree.max_depth == FLAGS.init_grid_depth

@torch.no_grad()
def step2(tree, eval_points_pfn):
    print('* Step 2: AA')

    for iterid in range(FLAGS.step2_n_iter):
        print(' iter', iterid + 1, 'of', FLAGS.step2_n_iter)
        n_samples = FLAGS.samples_per_cell
        corners = tree.corners()
        n_all_leaves = corners.shape[0]
        tdepth = tree.max_depth
        side_len = (1.0 / tree.N) ** (tdepth + 1)

        depths = tree.depth
        #  if iterid < FLAGS.step2_n_iter - 1:
        leaf_mask = depths==tdepth
        #  else:
        #      leaf_mask = depths >= tdepth - 1
        corners = corners[leaf_mask]
        n_leaves = corners.shape[0]

        #  side_len = np.power(side_len,
        #          (depths[leaf_mask] + 1).astype(
        #              np.float32)).reshape(n_leaves, 1, 1)

        points = corners[:, None] + np.random.uniform(
                size=(n_leaves, n_samples, 3)) * side_len  # (n_cells, n_samples, 3)
        points = (points - tree.offset) / tree.invradius
        points = points.reshape(-1, 3)  # (n_cells * n_samples, 3)

        rgb, sigma = utils.eval_points(
            eval_points_pfn, 
            torch.from_numpy(points).float().to(device), 
            FLAGS.point_chunk,
            to_cpu=True)
        sigma = sigma.reshape(n_leaves, n_samples, 1)
        rgb = rgb.reshape(n_leaves, n_samples, -1)

        tree[leaf_mask, -1:] = sigma.mean(axis=1)  #  (n_cells, 1)
        tree[leaf_mask, :-1] = rgb.mean(axis=1)    #  (n_cells, rgb size)

        if iterid < FLAGS.step2_n_iter - 1:
            sigma = sigma[..., 0]
            sigma_max = np.max(sigma, axis=1)
            sigma_min = np.min(sigma, axis=1)
            approx_alpha_max = 1.0 - np.exp(- 2.0 * sigma_max / side_len)
            approx_alpha_min = 1.0 - np.exp(- 2.0 * sigma_min / side_len)
            neg_dsigma = approx_alpha_min - approx_alpha_max

            vals_sor = -np.sort(neg_dsigma)
            indices_sor = np.argsort(neg_dsigma)

            max_idx = int(np.ceil(n_leaves * FLAGS.max_refine_prop))
            thresh_idx = ((vals_sor > FLAGS.step2_refine_thresh).astype(
                np.long) * np.arange(
                n_leaves, dtype=np.long)).max().item() + 1
            max_idx = min(max_idx, thresh_idx)
            indices_sor = indices_sor[:max_idx]

            mask_tmp = np.zeros(n_leaves, dtype=np.bool)
            mask_tmp[indices_sor] = True

            mask = np.zeros(n_all_leaves, dtype=np.bool)
            mask[depths == tdepth] = mask_tmp
            print('REFINE', max_idx, 'leaves, of', thresh_idx,  '> thresh')
            tree.refine(mask=mask)
            vals_sor = None



def main(unused_argv):
    utils.set_random_seed(20210222)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_data=False)

    center = list(map(float, FLAGS.center.split()))
    if len(center) == 1:
        center *= 3

    data_dim =  1 + 3 * ((FLAGS.sh_order + 1) ** 2 if FLAGS.sh_order >= 0 else 1)
    tree = N3Tree(N=FLAGS.tree_branch_n,
                  data_dim=data_dim,
                  init_refine=0,
                  depth_limit=FLAGS.init_grid_depth + FLAGS.step2_n_iter - 1,
                  radius=FLAGS.radius,
                  center=center)

    print('* Creating model')
    model, state = models.get_model_state(FLAGS, device=device, restore=True)
    eval_points_pfn = utils.get_eval_points_pfn(model, raw_rgb=True,
            coarse=FLAGS.coarse)

    step1(tree, eval_points_pfn)
    step2(tree, eval_points_pfn)
    #  tree[:, -1] = np.maximum(tree[:, -1].value(), 0.0)
    tree.shrink_to_fit()
    tree._push_to_leaf()
    print(tree)
    print('* Saving', FLAGS.output)
    tree.savez(FLAGS.output)


if __name__ == "__main__":
    app.run(main)