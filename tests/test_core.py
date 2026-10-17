import numpy as np
import torch

from wfr_mfm.inference import rollout
from wfr_mfm.paths import traveling_dirac_path
from wfr_mfm.time_sampling import sample_time_pairs
from wfr_mfm.transport import solve_wfr_plan


def test_time_pairs_are_ordered_and_mixed() -> None:
    torch.manual_seed(3)
    t, terminal = sample_time_pairs(20_000, p_diff=0.6, device="cpu")
    assert torch.all(t <= terminal)
    different = (terminal - t).abs() > 1e-8
    assert abs(float(different.float().mean()) - 0.6) < 0.02


def test_traveling_dirac_path_reaches_both_endpoints() -> None:
    source = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 0.5], [1.0, 0.5]])
    mass0 = torch.tensor([[0.8], [0.8]])
    mass1 = torch.tensor([[1.2], [1.2]])
    relative_time = torch.tensor([[0.0], [1.0]])
    path = traveling_dirac_path(relative_time, 1.0, source, target, mass0, mass1, 2.0)
    torch.testing.assert_close(path.position[0], source[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(path.position[1], target[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(path.relative_mass[0], torch.tensor([1.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        path.relative_mass[1], mass1[1] / mass0[1], atol=1e-5, rtol=1e-5
    )


def test_wfr_plan_has_normalized_marginal_maps() -> None:
    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([[0.2], [1.8]], dtype=np.float32)
    plan = solve_wfr_plan(source, target, delta=2.0)
    block = plan.blocks[0]
    np.testing.assert_allclose(block.gamma0.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(block.gamma1.sum(axis=0), 1.0, atol=1e-5)


class ConstantMeanFlow(torch.nn.Module):
    def forward(self, x, t, terminal_time, condition=None):
        velocity = torch.ones_like(x)
        growth = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        return velocity, growth


def test_one_step_rollout() -> None:
    initial = torch.zeros((4, 2))
    result = rollout(ConstantMeanFlow(), initial, [0.0, 1.0], steps_per_interval=1)
    torch.testing.assert_close(result[1.0][0], torch.ones_like(initial))
    torch.testing.assert_close(result[1.0][1], torch.ones((4, 1)))
