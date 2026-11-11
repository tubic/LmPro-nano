import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from train_table3 import train_accumulated_update


class ToyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        prediction = self.weight * batch["x"]
        loss = ((prediction - batch["y"]) ** 2).mean()
        return {"total_loss": loss}, {"ddg": prediction}


class AggregateToyRegressor(torch.nn.Module):
    aggregate_loss_across_accumulation = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.effective_batch_size = None

    def forward(self, batch):
        prediction = self.weight * batch["x"]
        target = batch["y"]
        loss = ((prediction - target) ** 2).mean()
        return {
            "total_loss": loss,
        }, {
            "ddg": prediction,
            "_loss_fwd": prediction,
            "_loss_rev": -prediction,
            "_loss_target": target,
        }

    def loss_from_branches(self, fwd, _rev, target):
        self.effective_batch_size = int(fwd.numel())
        return (fwd.mean() - target.mean()) ** 2


class Table3AccumulationTest(unittest.TestCase):
    def test_microbatch_gradients_are_averaged_before_step(self):
        model = ToyRegressor()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = iter(
            [
                {"x": torch.tensor([1.0]), "y": torch.tensor([0.0])},
                {"x": torch.tensor([2.0]), "y": torch.tensor([0.0])},
            ]
        )

        loss, grad_norm = train_accumulated_update(
            model,
            optimizer,
            batches,
            torch.device("cpu"),
            accumulation_steps=2,
            max_grad_norm=100.0,
            iteration=1,
        )

        self.assertAlmostEqual(loss, 2.5, places=6)
        self.assertAlmostEqual(float(grad_norm), 5.0, places=6)
        self.assertAlmostEqual(float(model.weight), 0.5, places=6)

    def test_non_additive_loss_sees_the_complete_effective_batch(self):
        model = AggregateToyRegressor()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = iter(
            [
                {"x": torch.tensor([1.0]), "y": torch.tensor([0.0])},
                {"x": torch.tensor([3.0]), "y": torch.tensor([0.0])},
            ]
        )

        loss, grad_norm = train_accumulated_update(
            model,
            optimizer,
            batches,
            torch.device("cpu"),
            accumulation_steps=2,
            max_grad_norm=100.0,
            iteration=1,
        )

        self.assertEqual(model.effective_batch_size, 2)
        self.assertAlmostEqual(loss, 4.0, places=6)
        self.assertAlmostEqual(float(grad_norm), 8.0, places=6)
        self.assertAlmostEqual(float(model.weight), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
