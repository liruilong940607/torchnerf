from tqdm import tqdm
from absl import flags
from absl.testing import absltest

from torchnerf.nerf.datasets import *
from torchnerf.nerf.utils import *

FLAGS = flags.FLAGS
define_flags()

class TestDatasets(absltest.TestCase):

    def test_get_dataset(self): 
        update_flags(FLAGS)
        dataset = get_dataset(split="train", args=FLAGS)
        for data in tqdm(dataset):
            pass

if __name__ == '__main__':
    absltest.main()