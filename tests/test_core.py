import unittest

import torch

from causalattack.backends import FixedSmokeFeatures, SyntheticPC
from causalattack.core import (AttackSettings, biased_mmd_squared, display_command,
                               lab_ab_loss, optimize, shared_indices)
from run_demo import synthetic_pair


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(2)

    def test_biased_mmd_and_gradient(self):
        a = torch.randn(12, 5)
        self.assertLess(abs(float(biased_mmd_squared(a, a))), 1e-6)
        b = torch.randn(12, 5, requires_grad=True)
        loss = biased_mmd_squared(a, b)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(b.grad.norm()), 0)

    def test_fixed_with_replacement_indices(self):
        feature = torch.zeros(1, 5, 8, 8)
        first = shared_indices(feature, 1024, 2026)
        second = shared_indices(feature, 1024, 2026)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(first), 1024)
        self.assertLess(int(first.unique().numel()), 1024)

    def test_lab_and_final_command(self):
        texture = torch.tensor([[[[-0.2, 1.2]], [[0.5, 0.1]], [[0.2, 0.7]]]],
                               requires_grad=True)
        loss = lab_ab_loss(texture)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(texture.grad).all())
        command = display_command(texture)
        self.assertEqual(command.dtype, torch.uint8)
        self.assertEqual(int(command.min()), 0)
        self.assertEqual(int(command.max()), 255)

    def test_completed_update_changes_continuous_texture(self):
        scene, reference = synthetic_pair(64, torch.device('cpu'))
        texture, rows, result = optimize(scene, reference, SyntheticPC(),
                                         FixedSmokeFeatures(),
                                         AttackSettings(outer_steps=3, learning_rate=12,
                                                        alpha=1/(64*64)))
        self.assertEqual(len(rows), 3)
        self.assertTrue(torch.isfinite(texture).all())
        self.assertGreater(float(texture.abs().max()), 0)
        self.assertGreater(rows[0]['gradient_norm'], 0)
        self.assertLess(result['final_mmd'], result['initial_mmd'])


if __name__ == '__main__':
    unittest.main()
