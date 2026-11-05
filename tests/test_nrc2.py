import copy
import numpy as np
import pytest
import torch
from agent.nrc2 import nrc2_loss
from agent.sf_simple import SFSimpleAgent
from agent.sf_nrc2 import SFNRC2Agent

torch.set_num_threads(1)


def test_geometry():
    w = torch.tensor([[1., 0., 0.]])
    h = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
    assert nrc2_loss(h, w).item() == 0
    assert nrc2_loss(h.roll(1, 1), w).item() == 1
    assert torch.allclose(nrc2_loss(h + 20, w), nrc2_loss(h * 3, w))
    assert nrc2_loss(h, torch.eye(3)).item() == 0
    assert nrc2_loss(h, torch.cat([w,w])).item() < 1e-12
    assert torch.isfinite(nrc2_loss(torch.ones(4, 3), w))


def test_gradients():
    torch.manual_seed(5)
    h = torch.randn(6, 5, dtype=torch.double, requires_grad=True)
    w = torch.randn(2, 5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(nrc2_loss, (h,w), atol=1e-4)
    nrc2_loss(h,w).backward()
    assert h.grad.norm() > 0 and w.grad.norm() > 0


def make(cls, **extra):
    return cls(update_task_every_step=10, sf_dim=4, num_init_steps=0,
        lr_task=1e-5, normalize_basis_features=True, normalize_task_params=True,
        name='test', reward_free=False, obs_type='states', obs_shape=(5,),
        action_shape=(2,), device='cpu', lr=1e-4, feature_dim=16, hidden_dim=16,
        critic_target_tau=.01, num_expl_steps=0, update_every_steps=1,
        stddev_schedule=.2, nstep=1, batch_size=8, stddev_clip=.3,
        init_critic=True, use_tb=False, use_wandb=False, update_encoder=True,
        consolidation=False, **extra)


def batch():
    return (np.random.randn(8,5).astype('float32'),
        np.random.uniform(-1,1,(8,2)).astype('float32'),
        np.random.rand(8,1).astype('float32'),
        np.full((8,1), .99, dtype='float32'),
        np.random.randn(8,5).astype('float32'),
        np.random.randn(8,4).astype('float32'))


def test_zero_matches_original_full_update():
    torch.manual_seed(7)
    a = make(SFSimpleAgent)
    torch.manual_seed(7)
    b = make(SFNRC2Agent, nrc2_coef=0., geometry_every=1)
    data = batch()
    for step in (1,10):
        torch.manual_seed(step)
        a.update(iter([data]), step)
        torch.manual_seed(step)
        b.update(iter([data]), step)
        for name in ('actor','critic','featureNet','critic_target'):
            for p,q in zip(getattr(a,name).parameters(),getattr(b,name).parameters()):
                assert torch.equal(p,q), name
        assert torch.equal(a.task_params,b.task_params)


@pytest.mark.parametrize('target', ['successor','feature','both'])
def test_regularized_updates(target):
    a = make(SFNRC2Agent, nrc2_coef=.1, nrc2_target=target, geometry_every=1)
    data = batch()
    for step in range(1,12):
        metrics = a.update(iter([data]), step)
        assert 0 <= metrics['nrc2_loss'] <= 1.000001
        assert all(np.isfinite(x) for x in metrics.values())
    assert all(len(m._forward_pre_hooks)==0 for m in a.critic.modules())
