import unittest
from torchnerf.nerf.model_utils import *

class TestModelUtils(unittest.TestCase):

    input_dim = 63
    condition_dim = 27
    batch_size = 2
    num_samples = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_MLP(self):
        model = MLP(input_dim=self.input_dim,
                    condition_dim=self.condition_dim).to(self.device)
        x = torch.randn((self.batch_size, self.num_samples, self.input_dim)).to(self.device)
        condition = torch.randn([self.batch_size, self.condition_dim]).to(self.device)
        raw_rgb, raw_sigma = model(x, condition)
        assert raw_rgb.size() == (self.batch_size, self.num_samples, 3)
        assert raw_sigma.size() == (self.batch_size, self.num_samples, 1)

    def test_sample_along_rays(self):
        origins = torch.randn([self.batch_size, 3]).to(self.device)
        directions = torch.randn([self.batch_size, 3]).to(self.device)
        num_samples = self.num_samples
        near = 0.1
        far = 1.0
        randomized = False # True
        lindisp = True # False
        z_vals, coords = sample_along_rays(
            origins, directions, num_samples, near, far, randomized, lindisp)
        assert z_vals.size() == (self.batch_size, self.num_samples), z_vals.size()
        assert coords.size() == (self.batch_size, self.num_samples, 3), coords.size()

    def test_posenc(self):
        x = torch.randn([self.batch_size, 3]).to(self.device)
        min_deg = 0
        max_deg = 10
        encoded = posenc(x, min_deg, max_deg, legacy_posenc_order=False)
        assert encoded.size() == (self.batch_size, 63)

    def test_volumetric_rendering(self):
        rgb = torch.randn([self.batch_size, self.num_samples, 3]).to(self.device)
        sigma = torch.randn([self.batch_size, self.num_samples, 1]).to(self.device)
        z_vals = torch.randn([self.batch_size, self.num_samples]).to(self.device)
        dirs = torch.randn([self.batch_size, 3]).to(self.device)
        white_bkgd = True 
        comp_rgb, disp, acc, weights = volumetric_rendering(
            rgb, sigma, z_vals, dirs, white_bkgd)
        assert comp_rgb.size() == (self.batch_size, 3)
        assert disp.size() == (self.batch_size,)
        assert acc.size() == (self.batch_size,)
        assert weights.size() == (self.batch_size, self.num_samples)

    def test_piecewise_constant_pdf(self):
        num_bins = 64
        bins = torch.randn([self.batch_size, num_bins + 1]).to(self.device)
        weights = torch.randn([self.batch_size, num_bins]).to(self.device)
        num_samples = self.num_samples
        randomized = True
        z_samples = piecewise_constant_pdf(
            bins, weights, num_samples, randomized)
        assert z_samples.size() == (self.batch_size, self.num_samples)

    def test_sample_pdf(self):
        num_bins = 64
        bins = torch.randn([self.batch_size, num_bins + 1]).to(self.device)
        weights = torch.randn([self.batch_size, num_bins]).to(self.device)
        origins = torch.randn([self.batch_size, 3]).to(self.device)
        directions = torch.randn([self.batch_size, 3]).to(self.device)
        z_vals = torch.randn([self.batch_size, self.num_samples]).to(self.device)
        num_samples = self.num_samples
        randomized = True
        z_vals, coords = sample_pdf(
            bins, weights, origins, directions, z_vals, num_samples, randomized
        )
        assert z_vals.size() == (self.batch_size, self.num_samples*2), z_vals.size()
        assert coords.size() == (self.batch_size, self.num_samples*2, 3), coords.size()
    
    def test_add_gaussian_noise(self):
        raw = torch.randn([self.batch_size, 3]).to(self.device)
        noise_std = 1.0
        randomized = True
        output = add_gaussian_noise(raw, noise_std, randomized)
        assert output.size() == raw.size()


if __name__ == '__main__':
    unittest.main()