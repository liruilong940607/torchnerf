# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Evaluation script for Nerf."""

import os

import functools
from os import path

from absl import app
from absl import flags

import torch
import numpy as np

from torchnerf.nerf import datasets
from torchnerf.nerf import models
from torchnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

def main(unused_argv):
    utils.set_random_seed(20210222)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)
    model, state = models.get_model_state(FLAGS, device=device, restore=False)

    # Rendering is forced to be deterministic even if training was randomized, as
    # this eliminates "speckle" artifacts.
    render_pfn = utils.get_render_pfn(model, randomized=False)

    # Compiling to the CPU because it's faster and more accurate.
    ssim_fn = functools.partial(utils.compute_ssim, max_val=1.0)

    last_step = 0
    out_dir = path.join(
        FLAGS.train_dir, "path_renders" if FLAGS.render_path else "test_preds"
    )
    if not FLAGS.eval_once:
        summary_writer = torch.utils.tensorboard.SummaryWriter(
            path.join(FLAGS.train_dir, "eval"))
    while True:
        model, state = models.restore_model_state(FLAGS, model, state)
        step = int(state.step)
        if step <= last_step:
            continue
        if FLAGS.save_output and (not utils.isdir(out_dir)):
            utils.makedirs(out_dir)
        psnrs = []
        ssims = []
        if not FLAGS.eval_once:
            showcase_index = np.random.randint(0, dataset.size)
        for idx in range(dataset.size):
            print(f"Evaluating {idx+1}/{dataset.size}")
            batch = next(dataset)
            rays = utils.namedtuple_map(
                lambda z: torch.from_numpy(z.copy()).to(device), batch["rays"])
            with torch.no_grad():
                pred_color, pred_disp, pred_acc = utils.render_image(
                    render_pfn,
                    rays,
                    FLAGS.dataset == "llff",
                    chunk=FLAGS.chunk,
                )
                if not FLAGS.render_path:
                    gt_color = torch.from_numpy(batch["pixels"]).to(device)
                    psnr = utils.compute_psnr(((pred_color - gt_color) ** 2).mean()).cpu().item()
                    ssim = ssim_fn(pred_color, gt_color).mean().cpu().item()
                    print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
                    psnrs.append(float(psnr))
                    ssims.append(float(ssim))
            pred_color = pred_color.cpu().numpy()
            pred_disp = pred_disp.cpu().numpy()
            pred_acc = pred_acc.cpu().numpy()
            if not FLAGS.eval_once and idx == showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                if not FLAGS.render_path:
                    showcase_gt = batch["pixels"]
            if FLAGS.save_output:
                utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
                utils.save_img(
                    pred_disp[Ellipsis, 0],
                    path.join(out_dir, "disp_{:03d}.png".format(idx)),
                )
        if not FLAGS.eval_once:
            summary_writer.add_image("pred_color", showcase_color, step)
            summary_writer.add_image("pred_disp", showcase_disp, step)
            summary_writer.add_image("pred_acc", showcase_acc, step)
            if not FLAGS.render_path:
                summary_writer.add_scalar("psnr", np.mean(np.array(psnrs)), step)
                summary_writer.add_scalar("ssim", np.mean(np.array(ssims)), step)
                summary_writer.add_image("target", showcase_gt, step)
        if FLAGS.save_output and (not FLAGS.render_path):
            with utils.open_file(path.join(out_dir, "psnr.txt"), "w") as pout:
                pout.write("{}".format(np.mean(np.array(psnrs))))
            with utils.open_file(path.join(out_dir, "ssim.txt"), "w") as pout:
                pout.write("{}".format(np.mean(np.array(ssims))))
        if FLAGS.eval_once:
            break
        if int(step) >= FLAGS.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)