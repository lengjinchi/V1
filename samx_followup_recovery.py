"""Explicit recovery of an interrupted follow-up; no model architecture changes."""
from pathlib import Path

import torch

from vajra.models.vajra.segment.train import SegmentationTrainer


def assert_same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same_state(left, right)
    else:
        assert actual == expected, (actual, expected)


class FollowupResumeTrainer(SegmentationTrainer):
    def setup_model(self):
        # The standard inference loader prefers EMA. Resume the student with
        # its own optimizer, and restore EMA separately in resume_training.
        checkpoint = torch.load(self.args.resume, map_location='cpu', weights_only=False)
        assert checkpoint['optimizer'] is not None and checkpoint['epoch'] >= 0
        self.model = self.get_model(weights=checkpoint['model'].float(), verbose=False)
        self._recovery_checkpoint = checkpoint
        return checkpoint

    def resume_training(self, checkpoint):
        super().resume_training(checkpoint)
        # Avoid default names such as last-last-vajra-v1-nano-seg.pt.
        architecture = checkpoint['model'].model_name
        self.last = Path(self.weights_dir) / f'last-{architecture}.pt'
        self.best = Path(self.weights_dir) / f'best-{architecture}.pt'
        best = torch.load(self.best, map_location='cpu', weights_only=False)
        assert best['epoch'] <= checkpoint['epoch']
        assert best['best_fitness'] == checkpoint['best_fitness']
        self.stopper.best_epoch = best['epoch'] + 1
        self.stopper.best_fitness = checkpoint['best_fitness']
        delta = self.start_epoch - self.stopper.best_epoch
        self.stopper.possible_stop = delta >= self.stopper.patience - 1

    def verify_recovery(self, context):
        checkpoint = self._recovery_checkpoint
        assert self.start_epoch == context['completed_epochs'] == checkpoint['epoch'] + 1
        assert self.epochs == 200 and self.args.patience == 30
        assert self.stopper.best_epoch == context['best_epoch']
        assert self.best_fitness == self.stopper.best_fitness == context['best_fitness']
        assert self.ema.updates == checkpoint['updates']
        assert_same_state(self.model.state_dict(), checkpoint['model'].float().state_dict())
        assert_same_state(self.ema.ema.state_dict(), checkpoint['ema'].float().state_dict())
        assert_same_state(self.optimizer.state_dict(), checkpoint['optimizer'])
        # Saves precede scheduler.step() in this repository. Restore the LR
        # that would follow that step, rather than repeating the previous LR.
        self.scheduler.last_epoch = self.start_epoch
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] * self.lf(self.start_epoch)
        self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]
        del self._recovery_checkpoint
