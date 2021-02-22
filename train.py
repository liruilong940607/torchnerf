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
"""Training script for Nerf."""

import os

import functools
import gc
import time
from absl import app
from absl import flags

import numpy as np
import torch

from torchnerf.nerf import datasets
from torchnerf.nerf import models
from torchnerf.nerf import utils

FLAGS = flags.FLAGS

utils.define_flags()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

def train_step(model, state, batch, lr):
    """One optimization step.
    Args:
      model: The linen model.
      state: TrainState contains optimizer/step.
      batch: dict, a mini-batch of data for training.
      lr: float, real-time learning rate.
    Returns:
      state: new state
      stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
    """
    def loss_fn():
        batch["pixels"] = torch.from_numpy(batch["pixels"]).to(device)
        rays = utils.namedtuple_map(
            lambda z: torch.from_numpy(z).to(device), batch["rays"])
        ret = model(rays, FLAGS.randomized)
        if len(ret) not in (1, 2):
            raise ValueError(
                "ret should contain either 1 set of output (coarse only), or 2 sets"
                "of output (coarse as ret[0] and fine as ret[1])."
            )
        # The main prediction is always at the end of the ret list.
        rgb, unused_disp, unused_acc = ret[-1]
        loss = ((rgb - batch["pixels"][Ellipsis, :3]) ** 2).mean()
        psnr = utils.compute_psnr(loss)
        if len(ret) > 1:
            # If there are both coarse and fine predictions, we compute the loss for
            # the coarse prediction (ret[0]) as well.
            rgb_c, unused_disp_c, unused_acc_c = ret[0]
            loss_c = ((rgb_c - batch["pixels"][Ellipsis, :3]) ** 2).mean()
            psnr_c = utils.compute_psnr(loss_c)
        else:
            loss_c = 0.0
            psnr_c = 0.0

        stats = utils.Stats(
            loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c
        )
        return loss + loss_c, stats

    loss, stats = loss_fn()

    for param in state.optimizer.param_groups:
        param['lr'] = lr
    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()
    state.step += 1

    return state, stats 


def main(unused_argv):
    utils.set_random_seed(20210222)

    utils.update_flags(FLAGS)
    utils.check_flags(FLAGS, require_batch_size_div=False)

    utils.makedirs(FLAGS.train_dir)
    render_dir = os.path.join(FLAGS.train_dir, 'render')
    utils.makedirs(render_dir)

    print('* Load train data')
    dataset = datasets.get_dataset("train", FLAGS)
    print('* Load test data')
    test_dataset = datasets.get_dataset("test", FLAGS)
    print('* Load model')
    model, state = models.get_model_state(FLAGS, device=device, restore=True)

    learning_rate_fn = functools.partial(
        utils.learning_rate_decay,
        lr_init=FLAGS.lr_init,
        lr_final=FLAGS.lr_final,
        max_steps=FLAGS.max_steps,
        lr_delay_steps=FLAGS.lr_delay_steps,
        lr_delay_mult=FLAGS.lr_delay_mult,
    )

    render_pfn = utils.get_render_pfn(model, randomized=FLAGS.randomized)
    ssim_fn = functools.partial(utils.compute_ssim, max_val=1.0)

    # Resume training step of the last checkpoint.
    init_step = state.step + 1
    summary_writer = torch.utils.tensorboard.SummaryWriter(FLAGS.train_dir)

    stats_trace = []
    reset_timer = True
    for step, batch in zip(range(init_step, FLAGS.max_steps + 1), dataset):
        model.train()
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False
        lr = learning_rate_fn(step)
        state, stats = train_step(model, state, batch, lr)
        stats_trace.append(stats)

        # Log training summaries.
        if step % FLAGS.print_every == 0:
            summary_writer.add_scalar("train_loss", stats.loss.item(), step)
            summary_writer.add_scalar("train_psnr", stats.psnr.item(), step)
            summary_writer.add_scalar("train_loss_coarse", stats.loss_c.item(), step)
            summary_writer.add_scalar("train_psnr_coarse", stats.psnr_c.item(), step)
            avg_loss = sum([s.loss for s in stats_trace]) / len(stats_trace)
            avg_psnr = sum([s.psnr for s in stats_trace]) / len(stats_trace)
            stats_trace = []
            summary_writer.add_scalar("train_avg_loss", avg_loss.item(), step)
            summary_writer.add_scalar("train_avg_psnr", avg_psnr.item(), step)
            summary_writer.add_scalar("learning_rate", lr, step)
            steps_per_sec = FLAGS.print_every / (time.time() - t_loop_start)
            reset_timer = True
            rays_per_sec = FLAGS.batch_size * steps_per_sec
            summary_writer.add_scalar("train_steps_per_sec", steps_per_sec, step)
            summary_writer.add_scalar("train_rays_per_sec", rays_per_sec, step)
            precision = int(np.ceil(np.log10(FLAGS.max_steps))) + 1
            print(
                ("{:" + "{:d}".format(precision) + "d}").format(step)
                + f"/{FLAGS.max_steps:d}: "
                + f"i_loss={stats.loss.item():0.4f}, "
                + f"avg_loss={avg_loss.item():0.4f}, "
                + f"lr={lr:0.2e}, "
                + f"{rays_per_sec:0.0f} rays/sec"
            )
        if step % FLAGS.save_every == 0:
            print('* Saving')
            torch.save({
                'step': state.step,
                'model': model.state_dict(),
                'optimizer': state.optimizer.state_dict(),
            }, os.path.join(FLAGS.train_dir, f"step-{step:09d}.ckpt"))

        # Test-set evaluation.
        if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
            model.eval()
            print('\n* Rendering')
            t_eval_start = time.time()
            test_case = next(test_dataset)
            gt_color = torch.from_numpy(test_case["pixels"]).to(device)
            rays = utils.namedtuple_map(
                lambda z: torch.from_numpy(z.copy()).to(device), test_case["rays"])
            
            with torch.no_grad():
                pred_color, pred_disp, pred_acc = utils.render_image(
                    render_pfn,
                    rays,
                    FLAGS.dataset == "llff",
                    chunk=FLAGS.chunk,
                )
                psnr = utils.compute_psnr(
                    ((pred_color - gt_color) ** 2).mean()
                )
                ssim = ssim_fn(pred_color, gt_color).mean()

            eval_time = time.time() - t_eval_start
            num_rays = np.prod(np.array(rays.directions.size()[:-1]))
            rays_per_sec = num_rays / eval_time
            summary_writer.add_scalar("test_rays_per_sec", rays_per_sec, step)
            print(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
            summary_writer.add_scalar("test_psnr", psnr.item(), step)
            summary_writer.add_scalar("test_ssim", ssim.item(), step)

            gt_color = gt_color.cpu().numpy()
            pred_color = pred_color.cpu().numpy()
            pred_disp = pred_disp.cpu().numpy()
            pred_acc = pred_acc.cpu().numpy()
            vis_list= [gt_color,
                        pred_color,
                        np.repeat(pred_disp, 3, axis=-1),
                        np.repeat(pred_acc, 3, axis=-1)]
            out_path = os.path.join(render_dir, '{:010}.png'.format(step))
            utils.save_img(np.hstack(vis_list), out_path)
            print(' Rendering saved to ', out_path)

            # I am saving rendering to disk instead of Tensorboard
            # Since Tensorboard begins to load very slowly when it has many images
            #  summary_writer.image("test_pred_color", pred_color, step)
            #  summary_writer.image("test_pred_disp", pred_disp, step)
            #  summary_writer.image("test_pred_acc", pred_acc, step)
            #  summary_writer.image("test_target", test_case["pixels"], step)

    if FLAGS.max_steps % FLAGS.save_every != 0:
        print('* Saving')
        torch.save({
            'step': state.step,
            'model': model.state_dict(),
            'optimizer': state.optimizer.state_dict(),
        }, os.path.join(FLAGS.train_dir, f"step-{step:09d}.ckpt"))



if __name__ == "__main__":
    app.run(main)