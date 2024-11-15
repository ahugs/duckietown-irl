import torch
from tqdm import tqdm
import numpy as np

class RewardLearner:

    def __init__(
        self,
        net,
        optim,
        expert_buffer,
        steps_per_epoch=5,
        batch_size=64,
        regularization_coeff=1,
        is_constraint=False,
        lr_scheduler=None,
        loss_transform=None,
    ):
        self.net = net
        self.optim = optim
        self.expert_buffer = expert_buffer
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.regularization_coeff = regularization_coeff
        self.lr_scheduler = lr_scheduler
        self.is_constraint = is_constraint
        self.loss_transform = loss_transform

    def update(self, learner_buffer):
        self.net.train()

        for _ in (pbar := tqdm(range(self.steps_per_epoch), desc='Reward Learning')):

            expert_batch, _ = self.expert_buffer.sample(self.batch_size)
            learner_batch, _ = learner_buffer.sample(self.batch_size)

            input_size = next(self.net.parameters()).size()
            concat = False
            if expert_batch.obs.shape[-1] < input_size[-1]:
                concat = True
            expert_input = np.concatenate(
                [expert_batch.obs, expert_batch.act if concat else []],
                axis=1,
            )
            learner_input = np.concatenate(
                [learner_batch.obs, learner_batch.act if concat else []],
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