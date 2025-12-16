# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from tqdm import tqdm

from pqf.training.training import ModelTrainer
from pqf.training.training_types import FinalSummary, TQDMState


class DiTAccumulator(object):
    def __init__(self):
        self._total_loss = 0.
        self._count = 0

    def reset(self):
        self._total_loss = 0.
        self._count = 0

    def accumulate(self, loss: torch.Tensor):
        loss = loss.detach().cpu()
        self._count += 1
        self._total_loss += loss.item()
        self._latest_state = {
            "loss": loss.item(),
        }

    def get_latest_state(self):
        return self._latest_state

    def get_average_state(self):
        return {
            "loss": self._total_loss / self._count,
        }

class DiTTrainer(ModelTrainer):
    def __init__(self, model, model_t, dit_generator, optimizer, lr_scheduler, batch_size):
        self.model = model
        self.model_t = model_t
        self.dit_generator = dit_generator

        self.accumulator = DiTAccumulator()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size

    def reset(self):
        self.accumulator.reset()

    def pass_to_model(self, data):
        loss_per_t = self.dit_generator.forward_train(self.model.forward_with_cfg, self.model_t.forward_with_cfg, data)
        total_loss = sum(sum(block) for block in loss_per_t)
        num_losses = sum(len(block) for block in loss_per_t)
        loss = total_loss / num_losses
        return loss

    def update_state(self, loss: torch.Tensor):
        self.accumulator.accumulate(loss)
        state = self.accumulator.get_latest_state()
        self.latest_state = {"loss": state["loss"]}

    def get_final_summary(self):
        state = self.accumulator.get_average_state()
        return FinalSummary(
            {"loss": f'{state["loss"]:.2f}'}
        )

    def handle_data(self, epoch, logger, verbose):
        total_data = torch.cat([torch.randperm(1000) for _ in range(10)], dim=0)
        n_batches = total_data.shape[0] // self.batch_size
        progress_data = tqdm(range(n_batches), desc=logger.get_desc("Epoch", epoch), disable=not verbose)
        for batch_idx in progress_data:
            idx = (epoch - 1) * n_batches + batch_idx
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            data = total_data[start_idx:end_idx]
            self.update(data)
            if verbose:
                progress_data.set_postfix(self.get_tqdm_state())
                logger.log_intermediate_summary(idx, self.get_intermediate_summary())
        if verbose:
            logger.log_final_summary(epoch, self.get_final_summary())
        return self.get_final_metric()

    def update(self, data):
        data = data.tolist()
        self.optimizer.zero_grad()
        loss = self.pass_to_model(data)
        self.update_state(loss)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler.step_batch():
            self.lr_scheduler.step()