from absl import flags
from absl.testing import absltest

from torchnerf.nerf.models import *
from torchnerf.nerf.utils import *

FLAGS = flags.FLAGS
define_flags()

class TestModels(absltest.TestCase):

    input_dim = 63
    condition_dim = 27
    batch_size = 2
    num_samples = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_eval_points_raw(self):
        model = NerfModel().to(self.device)
        points = torch.randn((self.num_samples, 3)).to(self.device)
        viewdirs = torch.randn((self.num_samples, 3)).to(self.device)
        coarse = False
        raw_rgb, raw_sigma = model.eval_points_raw(points, viewdirs, coarse)
        assert raw_rgb.size() == (self.num_samples, 3), raw_rgb.size()
        assert raw_sigma.size() == (self.num_samples, 1), raw_sigma.size()

    def test_eval_points(self):
        model = NerfModel().to(self.device)
        points = torch.randn((self.num_samples, 3)).to(self.device)
        viewdirs = torch.randn((self.num_samples, 3)).to(self.device)
        coarse = False
        raw_rgb, raw_sigma = model.eval_points(points, viewdirs, coarse)
        assert raw_rgb.size() == (self.num_samples, 3), raw_rgb.size()
        assert raw_sigma.size() == (self.num_samples, 1), raw_sigma.size()

    def test_get_model_state(self):
        restore = False
        model, state = get_model_state(FLAGS, restore)


if __name__ == '__main__':
    absltest.main()