import pytest
import torch

from data import Batch, BatchSampler, Dataset, DatasetTraverser, Episode
from data.utils import collate_segments_to_batch
from models.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_denoiser(num_conditioning=2):
    torch.manual_seed(7)
    cfg = DenoiserConfig(
        inner_model=InnerModelConfig(
            img_channels=1,
            num_steps_conditioning=num_conditioning,
            cond_channels=2 * num_conditioning,
            depths=[1],
            channels=[8],
            attn_depths=[False],
            num_actions=3,
        ),
        sigma_data=0.5,
        sigma_offset_noise=0.0,
    )
    denoiser = Denoiser(cfg)
    denoiser.setup_training(SigmaDistributionConfig(0.0, 0.0, 1.0, 1.0))
    return denoiser


def make_batch(mask):
    mask = torch.tensor(mask, dtype=torch.bool)
    b, t = mask.shape
    return Batch(
        obs=torch.linspace(-0.5, 0.5, b * t * 16).reshape(b, t, 1, 4, 4),
        act=torch.zeros(b, t, dtype=torch.long),
        rew=torch.zeros(b, t),
        end=torch.zeros(b, t, dtype=torch.long),
        trunc=torch.zeros(b, t, dtype=torch.long),
        mask_padding=mask,
        info=[{} for _ in range(b)],
        segment_ids=[],
    )


@pytest.mark.parametrize(
    "mask",
    [
        [[False, False, False, True]],
        [[True, True, True, False]],
        [[True, True, False, False]],
        [[True, True, True, True]],
        [[False, True, True, True], [True, True, False, True]],
    ],
)
def test_masked_loss_and_gradients(mask):
    denoiser = make_denoiser()
    batch = make_batch(mask)
    original_obs = batch.obs.clone()
    calls = []
    handle = denoiser.inner_model.register_forward_hook(
        lambda module, args, output: calls.append((args[0].detach().clone(), output))
    )
    try:
        loss, metrics = denoiser(batch)
    finally:
        handle.remove()

    # The hook receives c_in * noisy_obs; sigma=1 gives c_in=1/sqrt(1.25).
    expected = loss.new_zeros(())
    for i, (noisy_obs, prediction) in enumerate(calls):
        valid = batch.mask_padding[:, 2 + i]
        noisy_obs = noisy_obs * (1.25**0.5)
        target = (batch.obs[:, 2 + i] - 0.2 * noisy_obs) / (0.2**0.5)
        if valid.any():
            expected += (prediction[valid] - target[valid]).square().mean()
    expected /= len(calls)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["loss_denoising"], loss.detach())
    torch.testing.assert_close(batch.obs, original_obs, rtol=0, atol=0)
    loss.backward()
    gradients = [p.grad for p in denoiser.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    if not batch.mask_padding[:, 2:].any():
        assert loss.item() == 0.0
        assert all(torch.count_nonzero(g).item() == 0 for g in gradients)


def make_episode(length):
    return Episode(
        obs=torch.zeros(length, 1, 4, 4),
        act=torch.zeros(length, dtype=torch.long),
        rew=torch.zeros(length),
        end=torch.zeros(length, dtype=torch.long),
        trunc=torch.zeros(length, dtype=torch.long),
        info={},
    )


def test_short_evaluation_tail_has_zero_loss(tmp_path):
    dataset = Dataset(tmp_path, cache_in_ram=True, save_on_disk=False)
    dataset.add_episode(make_episode(3))
    # The native traverser keeps length-three chunks; default conditioning is four.
    batch, = list(DatasetTraverser(dataset, batch_num_samples=1, chunk_size=6))
    assert batch.mask_padding.tolist() == [[True, True, True, False, False, False]]
    loss, _ = make_denoiser(num_conditioning=4)(batch)
    assert loss.item() == 0.0
    loss.backward()


def test_short_training_sample_has_finite_loss(tmp_path):
    dataset = Dataset(tmp_path, cache_in_ram=True, save_on_disk=False)
    dataset.add_episode(make_episode(1))
    sampler = BatchSampler(dataset, rank=0, world_size=1, batch_size=2, seq_length=6)
    batch = collate_segments_to_batch([dataset[s] for s in sampler.sample()])
    assert batch.mask_padding.tolist() == [[False] * 5 + [True]] * 2
    loss, _ = make_denoiser(num_conditioning=4)(batch)
    assert torch.isfinite(loss)
    loss.backward()
