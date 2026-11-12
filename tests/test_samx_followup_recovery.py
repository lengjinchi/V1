import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from run_samx_experiments import write_json
from samx_followup_recovery import FollowupResumeTrainer, assert_same_state


class RecoveryTests(unittest.TestCase):
    def test_transient_windows_replace_lock_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'progress.json'
            write_json(target,{'epoch':18})
            original=Path.replace
            calls=[]
            def locked(source,destination):
                calls.append(source)
                if len(calls)<3:
                    self.assertEqual(json.loads(target.read_text()),{'epoch':18})
                    raise PermissionError('simulated Windows reader lock')
                return original(source,destination)
            with patch.object(Path,'replace',locked),patch('run_samx_experiments.time.sleep'):
                self.assertTrue(write_json(target,{'epoch':19}))
            self.assertEqual(len(calls),3)
            self.assertEqual(json.loads(target.read_text()),{'epoch':19})
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_persistent_lock_progress_skips_but_results_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'progress.json'
            write_json(target,{'epoch':19})
            with patch.object(Path,'replace',side_effect=PermissionError('locked')) as replace, \
                 patch('run_samx_experiments.time.sleep'):
                with self.assertWarns(RuntimeWarning):
                    self.assertFalse(write_json(target,{'epoch':20},best_effort=True))
                self.assertEqual(replace.call_count,8)
                with self.assertRaises(PermissionError):
                    write_json(target,{'epoch':20})
            self.assertEqual(json.loads(target.read_text()),{'epoch':19})
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_resume_uses_student_instead_of_ema(self):
        student=nn.Linear(2,1);ema=nn.Linear(2,1)
        with torch.no_grad():student.weight.fill_(1);ema.weight.fill_(7)
        trainer=object.__new__(FollowupResumeTrainer)
        trainer.args=SimpleNamespace(resume='last.pt')
        trainer.get_model=lambda weights,verbose:weights
        checkpoint=dict(model=student,ema=ema,optimizer={},epoch=18)
        with patch('samx_followup_recovery.torch.load',return_value=checkpoint):
            trainer.setup_model()
        torch.testing.assert_close(trainer.model.weight,torch.ones_like(student.weight))

    def test_verify_restores_post_save_lr_and_preserves_momentum(self):
        model=nn.Linear(2,1);ema=nn.Linear(2,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        model(torch.ones(1,2)).sum().backward();optimizer.step()
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x:1-x/200)
        trainer=object.__new__(FollowupResumeTrainer)
        trainer.model=model;trainer.ema=SimpleNamespace(ema=ema,updates=1968)
        trainer.optimizer=optimizer;trainer.scheduler=scheduler
        trainer.lf=lambda x:1-x/200
        trainer.start_epoch=19;trainer.epochs=200;trainer.args=SimpleNamespace(patience=30)
        trainer.stopper=SimpleNamespace(best_epoch=17,best_fitness=.39774)
        trainer.best_fitness=.39774
        trainer._recovery_checkpoint=dict(epoch=18,updates=1968,model=model,ema=ema,optimizer=optimizer.state_dict())
        context=dict(completed_epochs=19,best_epoch=17,best_fitness=.39774)
        momentum=[v['momentum_buffer'].clone() for v in optimizer.state.values()]
        trainer.verify_recovery(context)
        self.assertEqual(scheduler.last_epoch,19)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.01*(1-19/200))
        for old,state in zip(momentum,optimizer.state.values()):
            torch.testing.assert_close(state['momentum_buffer'],old,atol=0,rtol=0)
        scheduler.step()
        self.assertEqual(scheduler.last_epoch,20)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.009)

    def test_resume_rejects_wrong_optimizer_or_weights(self):
        with self.assertRaises(AssertionError):
            assert_same_state({'state':{1:torch.ones(2)}},{'state':{1:torch.zeros(2)}})

    def test_best_epoch_and_canonical_paths_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            model=nn.Linear(2,1);model.model_name='vajra-v1-nano-seg'
            ema=nn.Linear(2,1)
            optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
            trainer=object.__new__(FollowupResumeTrainer)
            trainer.model=model;trainer.ema=SimpleNamespace(ema=ema,updates=0)
            trainer.optimizer=optimizer;trainer.weights_dir=Path(directory)
            trainer.resume=True;trainer.epochs=200
            trainer.args=SimpleNamespace(model='last-vajra-v1-nano-seg.pt',close_mosaic=10)
            trainer.stopper=SimpleNamespace(best_epoch=0,best_fitness=0,patience=30,possible_stop=False)
            torch.save(dict(epoch=16,best_fitness=.39774),Path(directory)/'best-vajra-v1-nano-seg.pt')
            checkpoint=dict(epoch=18,model=model,ema=ema,updates=1968,
                            optimizer=optimizer.state_dict(),best_fitness=.39774)
            trainer.resume_training(checkpoint)
            self.assertEqual(trainer.start_epoch,19)
            self.assertEqual(trainer.stopper.best_epoch,17)
            self.assertEqual(trainer.stopper.best_fitness,.39774)
            self.assertEqual(trainer.ema.updates,1968)
            self.assertEqual(trainer.last.name,'last-vajra-v1-nano-seg.pt')
            self.assertFalse(trainer.stopper.possible_stop)


if __name__=='__main__':
    unittest.main()
