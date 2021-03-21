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

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torchnerf.nerf import datasets
from torchnerf.nerf import models
from torchnerf.nerf import utils


def train_step(model, state, batch, lr, device, args):
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
        ret = model(rays, args.randomized)
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


def main(local_rank, args):
    def print0(*strs):
        print(*strs) if local_rank is 0 else None

    print ("local_rank:", local_rank, "world_size:", args.world_size)
    dist.init_process_group(
    	backend='nccl',
        init_method='tcp://127.0.0.1:8686',
    	world_size=args.world_size,
    	rank=local_rank
    )

    device = f"cuda:{local_rank}"
    utils.set_random_seed(20210222 + local_rank)

    print0('* Load train data')
    dataset = datasets.get_dataset("train", args)
    print0('* Load test data')
    test_dataset = datasets.get_dataset("test", args)
    print0('* Load model')
    model, state = models.get_model_state(args, device=device, restore=True)
    model = DDP(model, device_ids=[local_rank])
    print0('* Done loading model')

    learning_rate_fn = functools.partial(
        utils.learning_rate_decay,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        max_steps=args.max_steps,
        lr_delay_steps=args.lr_delay_steps,
        lr_delay_mult=args.lr_delay_mult,
    )

    render_pfn = utils.get_render_pfn(model, randomized=args.randomized)
    ssim_fn = functools.partial(utils.compute_ssim, max_val=1.0)

    # Resume training step of the last checkpoint.
    init_step = state.step + 1
    if local_rank == 0:
        summary_writer = SummaryWriter(args.train_dir)

    stats_trace = []
    reset_timer = True
    for step, batch in zip(range(init_step, args.max_steps + 1), dataset):
        model.train()
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False
        lr = learning_rate_fn(step)
        state, stats = train_step(model, state, batch, lr, device, args)
        if local_rank == 0:
            stats_trace.append(stats)

        # Log training summaries.
        if local_rank == 0 and step % args.print_every == 0:
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
            steps_per_sec = args.print_every / (time.time() - t_loop_start)
            reset_timer = True
            rays_per_sec = args.batch_size * steps_per_sec * args.world_size
            summary_writer.add_scalar("train_steps_per_sec", steps_per_sec, step)
            summary_writer.add_scalar("train_rays_per_sec", rays_per_sec, step)
            precision = int(np.ceil(np.log10(args.max_steps))) + 1
            print0(
                ("{:" + "{:d}".format(precision) + "d}").format(step)
                + f"/{args.max_steps:d}: "
                + f"i_loss={stats.loss.item():0.4f}, "
                + f"avg_loss={avg_loss.item():0.4f}, "
                + f"lr={lr:0.2e}, "
                + f"{rays_per_sec:0.0f} rays/sec"
            )
        if local_rank == 0 and step % args.save_every == 0:
            print0('* Saving')
            torch.save({
                'step': state.step,
                'model': model.state_dict(),
                'optimizer': state.optimizer.state_dict(),
            }, os.path.join(args.train_dir, f"step-{step:09d}.ckpt"))

        # Test-set evaluation.
        if local_rank == 0 and args.render_every > 0 and step % args.render_every == 0:
            model.eval()
            print0('\n* Rendering')
            t_eval_start = time.time()
            test_case = next(test_dataset)
            gt_color = torch.from_numpy(test_case["pixels"]).to(device)
            rays = utils.namedtuple_map(
                lambda z: torch.from_numpy(z.copy()).to(device), test_case["rays"])

            with torch.no_grad():
                pred_color, pred_disp, pred_acc = utils.render_image(
                    render_pfn,
                    rays,
                    args.dataset == "llff",
                    chunk=args.chunk,
                )
                psnr = utils.compute_psnr(
                    ((pred_color - gt_color) ** 2).mean()
                )
                ssim = ssim_fn(pred_color, gt_color).mean()

            eval_time = time.time() - t_eval_start
            num_rays = np.prod(np.array(rays.directions.size()[:-1]))
            rays_per_sec = num_rays / eval_time
            summary_writer.add_scalar("test_rays_per_sec", rays_per_sec, step)
            print0(f"Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec")
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
            print0(' Rendering saved to ', out_path)

            # I am saving rendering to disk instead of Tensorboard
            # Since Tensorboard begins to load very slowly when it has many images
            #  summary_writer.image("test_pred_color", pred_color, step)
            #  summary_writer.image("test_pred_disp", pred_disp, step)
            #  summary_writer.image("test_pred_acc", pred_acc, step)
            #  summary_writer.image("test_target", test_case["pixels"], step)

    if local_rank == 0 and args.max_steps % args.save_every != 0:
        print0('* Saving')
        torch.save({
            'step': state.step,
            'model': model.state_dict(),
            'optimizer': state.optimizer.state_dict(),
        }, os.path.join(args.train_dir, f"step-{step:09d}.ckpt"))



if __name__ == "__main__":
    args = utils.define_flags()
    args.render_dir = os.path.join(args.train_dir, 'render')
    args.world_size = torch.cuda.device_count()

    utils.update_flags(args)
    utils.check_flags(args, require_batch_size_div=False)

    utils.makedirs(args.train_dir)
    utils.makedirs(args.render_dir)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12321'
    mp.spawn(main, nprocs=args.world_size, args=(args,), daemon=False)


