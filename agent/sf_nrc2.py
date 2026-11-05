"""Minimal Simple SF intervention; original agent remains available unchanged."""
import torch
import torch.nn.functional as F
import utils
from agent.sf_simple import SFSimpleAgent
from agent.nrc2 import nrc2_loss, geometry


class SFNRC2Agent(SFSimpleAgent):
    def __init__(self, nrc2_coef=0., nrc2_target='successor', geometry_every=1000, **kwargs):
        if nrc2_coef < 0 or nrc2_target not in ('successor', 'feature', 'both'):
            raise ValueError('Invalid NRC2 coefficient or target')
        self.nrc2_coef = nrc2_coef
        self.nrc2_target = nrc2_target
        self.geometry_every = geometry_every
        super().__init__(**kwargs)

    def update_critic(self, obs, action, reward, discount, next_obs, task, step):
        # Transient readout-input hooks preserve upstream forward and loss code.
        captures, handles = {}, []
        heads = {'feature': self.featureNet.trunk[-1],
                 'sf1': self.critic.Q1[-1], 'sf2': self.critic.Q2[-1]}
        measure = self.geometry_every > 0 and step % self.geometry_every == 0
        if self.nrc2_coef == 0 and not measure:
            return super().update_critic(obs, action, reward, discount, next_obs, task, step)
        def capture(name):
            def hook(module, inputs):
                # Ignore the target computation under no_grad.
                if torch.is_grad_enabled():
                    captures[name] = inputs[0]
            return hook
        for name, head in heads.items():
            handles.append(head.register_forward_pre_hook(capture(name)))
        try:
            if self.nrc2_coef == 0:
                # Hooks collect detached diagnostics only; exact original update.
                result = super().update_critic(obs, action, reward, discount, next_obs, task, step)
                if measure:
                    for name, h in captures.items():
                        for key, value in geometry(h, heads[name].weight).items():
                            result[f'geometry_{name}_{key}'] = value
                return result
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                next_action = self.actor(next_obs, stddev).sample(clip=self.stddev_clip)
                next_h = self.featureNet(next_obs, next_action)[0]
                q1, q2 = self.critic_target(next_h, next_action, task)
                target = reward + discount * torch.min(q1, q2)
            h = self.featureNet(obs, action)[0]
            q1, q2 = self.critic(h, action, task)
            td = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            names = ('sf1', 'sf2') if self.nrc2_target == 'successor' else ('feature',)
            if self.nrc2_target == 'both':
                names = ('feature', 'sf1', 'sf2')
            penalties = [nrc2_loss(captures[name], heads[name].weight) for name in names]
            reg = torch.stack(penalties).mean()
            loss = td + self.nrc2_coef * reg
            result = dict(critic_loss=td.item(), critic_total_loss=loss.item(),
                          nrc2_loss=reg.item(), critic_target_q=target.mean().item(),
                          critic_q1=q1.mean().item(), critic_q2=q2.mean().item())
            if measure:
                for name, h in captures.items():
                    for key, value in geometry(h, heads[name].weight).items():
                        result[f'geometry_{name}_{key}'] = value
            if self.encoder_opt is not None:
                self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.feature_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_opt.step()
            self.feature_opt.step()
            if self.encoder_opt is not None:
                self.encoder_opt.step()
            return result
        finally:
            for handle in handles:
                handle.remove()
