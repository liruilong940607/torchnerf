""""Different datasets implementation plus a general port for all the datasets."""
import json
import os
import threading
import queue

import cv2  # pylint: disable=g-import-not-at-top
import numpy as np
from PIL import Image

from torchnerf.nerf import utils
from torchnerf.nerf.extra_data import dryice1


def get_dataset(split, args):
    return dataset_dict[args.dataset](split, args)


class DatasetDyn(threading.Thread):
    """Dataset Base Class."""

    def __init__(self, split, args):
        super(Dataset, self).__init__()
        self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
        self.daemon = True
        self.split = split
        if split == "train":
            self._train_init(args)
        elif split == "test":
            self._test_init(args)
        else:
            raise ValueError(
                'the split argument should be either "train" or "test", set'
                "to {} here.".format(split)
            )
        self.batch_size = args.batch_size # // jax.host_count()
        self.image_batching = args.image_batching
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next training batch or test example.
        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.get()
        return x

    def peek(self):
        """Peek at the next training batch or test example without dequeuing it.
        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
        return x

    def run(self):
        if self.split == "train":
            next_func = self._next_train
        else:
            next_func = self._next_test
        while True:
            self.queue.put(next_func())

    @property
    def size(self):
        return self.n_examples

    def _train_init(self, args):
        """Initialize training."""
        self._load_renderings(args)
        self._generate_rays()

        if args.image_batching:
            # flatten the ray and image dimension together.
            self.images = self.images.reshape([-1, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, r.shape[-1]]), self.rays
            )
        else:
            self.images = self.images.reshape([-1, self.resolution, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays
            )

    def _test_init(self, args):
        self._load_renderings(args)
        self._generate_rays()
        self.it = 0

    def _next_train(self):
        """Sample next training batch."""

        if self.image_batching:
            ray_indices = np.random.randint(
                0, self.rays[0].shape[0], (self.batch_size,)
            )
            batch_pixels = self.images[ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
        else:
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(
                0, self.rays[0][0].shape[0], (self.batch_size,)
            )
            batch_pixels = self.images[image_index][ray_indices]
            batch_rays = utils.namedtuple_map(
                lambda r: r[image_index][ray_indices], self.rays
            )
        return {"pixels": batch_pixels, "rays": batch_rays}

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.render_path:
            return {"rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
        else:
            return {
                "pixels": self.images[idx],
                "rays": utils.namedtuple_map(lambda r: r[idx], self.rays),
            }

    # TODO(bydeng): Swap this function with a more flexible camera model.
    def _generate_rays(self):
        """Generating rays for all images."""
        # print(' Generating rays')
        self.rays = utils.generate_rays(self.w, self.h, self.focal, self.camtoworlds)


class Dryice1(Dataset):
    """Dryice1 Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        loader = dryice1.get_dataset()
        assert "camera" in loader.keyfilter
        assert "image" in loader.keyfilter
        assert "bg" in loader.keyfilter
        assert args.factor == 0

        images = []
        cams = []
        for i in range(len(loader)):
            frame, cam = loader.framecamlist[i]
            assert cam is not None
            # image
            imagepath = (
                dryice1._DATA_ROOT + "experiments/dryice1/data/cam{}/image{:04}.jpg"
                .format(cam, int(frame)))
            image = np.array(Image.open(imagepath), dtype=np.float32) / 255.0
            height, width = image.shape[1:3]
            valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
            assert valid
            # image bg
            bgpath = (
                dryice1._DATA_ROOT + "experiments/dryice1/data/cam{}/bg.jpg".format(cam))
            bg = np.array(Image.open(bgpath), dtype=np.float32) / 255.0
            image = image - bg
            # camera



        with utils.open_file(
            os.path.join(args.data_dir, "transforms_{}.json".format(self.split)), "r"
        ) as fp:
            meta = json.load(fp)
        images = []
        cams = []
        # print(' Load Blender', args.data_dir, 'split', self.split)
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "Blender dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            cams.append(frame["transform_matrix"])
            if args.white_bkgd:
                mask = image[..., -1:]
                image = image[..., :3] * mask + (1.0 - mask)
            else:
                image = image[..., :3]
            images.append(image)
        self.images = np.stack(images, axis=0)
        self.h, self.w = self.images.shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)
        self.n_examples = self.images.shape[0]


dataset_dict = {
    "dryice1": Dryice1,
}
