import torch
from tqdm import tqdm
import numpy as np

class RewardLearner:

    def __init__(
        self,
        net,
        optim,
        encoder,
        steps_per_epoch=5,
        batch_size=64,
        regularization_coeff=1,
        is_constraint=False,
        lr_scheduler=None,
        loss_transform=None,
    ):
        self.net = net
        self.optim = optim
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.regularization_coeff = regularization_coeff
        self.lr_scheduler = lr_scheduler
        self.is_constraint = is_constraint
        self.loss_transform = loss_transform
        self.encoder = encoder if encoder is not None else torch.nn.Identity()
        self.device = next(self.net.parameters()).device

    def update(self, expert_replay_iter, learner_replay_iter):
        self.net.train()

        pbar = tqdm(range(self.steps_per_epoch), desc='Reward Learning')
        for _ in (pbar):

            expert_obs, expert_action, _, _, _, _ = next(expert_replay_iter)
            with torch.no_grad():
                expert_obs = self.encoder(expert_obs.to(self.device))
            expert_action = expert_action.to(self.device)  
            learner_obs, learner_action, _, _, _, _ = next(learner_replay_iter)
            learner_action = learner_action.to(self.device)
            with torch.no_grad():
                learner_obs = self.encoder(learner_obs.to(self.device))

            input_size = next(self.net.parameters()).size()
            concat = False
            if expert_obs.shape[-1] < input_size[-1]:
                concat = True
            expert_input = torch._cat(
                [expert_obs, expert_action if concat else []],
                axis=1,
            )
            learner_input = torch.cat(
                [learner_obs, learner_action if concat else []],
                axis=1,
            )

            learner = self.net(learner_input)
            expert = self.net(expert_input)


            if self.loss_transform is not None:
                loss_learner = self.loss_transform(learner)
                loss_expert = self.loss_transform(expert)
            else:
                loss_learner = learner
                loss_expert = expert
            loss = loss_learner.mean() - loss_expert.mean()
            if self.is_constraint:
                loss = -loss
            regularization = torch.sum(expert**2)/expert.shape[0]
            loss += self.regularization_coeff * regularization
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pbar.set_postfix({'loss': loss.item(), 'agent_rew': learner.mean().item(), 'expert_rew': expert.mean().item()})
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.net.eval()
        return {'loss': loss.item(),
                'learner_reward': learner.mean().item(),
                'expert_reward': expert.mean().item()}
