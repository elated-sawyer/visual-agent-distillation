"""Exercise the actual container worker with fake environment/model I/O."""
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import native_async_turns as adapter
import slime.rollout.fully_async_rollout as native


class NativeAsyncContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text('{}')
        (self.root / 'online-catalogue.json').write_text('{}')
        self.env = patch.dict(os.environ, {'MM_OPD_RUN': str(self.root)})
        self.env.start()
        self.args = SimpleNamespace(num_rollout=3, rollout_batch_size=64, n_samples_per_prompt=1,
            rollout_data_transport='object-store', seed=20261019, sglang_server_concurrency=16,
            dynamic_sampling_filter_path=None, rollout_sample_filter_path='native_async_turns.accept_batch')
        self.source = adapter.TurnDataSource(self.args)
        self.version = 1
        self.sequences = [0] * 16
        self.scored = []

    def tearDown(self):
        self.source.close()
        self.env.stop()
        self.temp.cleanup()

    def ready(self, root):
        result = []
        for lane, sequence in enumerate(self.sequences):
            p = self.root / f'lane-{lane}' / str(sequence)
            p.mkdir(parents=True, exist_ok=True)
            result.append((p, {'lane_id': lane}))
        return result

    async def generate(self, args, group, sampling_params, evaluation=False):
        sample = group[0]
        path = Path(sample.metadata['decision'])
        lane = int(path.parent.name.split('-')[1])
        version = str(self.version)
        await asyncio.sleep(.004 + sample.index % 4 * .001)
        sample.weight_versions = [version]
        sample.rollout_log_probs = [-.5, -.4]
        sample.response_length = 2
        sample.metadata = {'sampling_weight_version': version, 'teacher_exact_response_ids_verified': True}
        self.sequences[lane] += 1
        with self.source.lock:
            self.source.busy_lanes.remove(lane)
        self.scored.append(sample.index)
        return group

    def test_native_worker_overlap_budget_and_drain(self):
        with patch('turn_stream_rollout.ready_decisions', self.ready), \
             patch.object(adapter, 'validate_request'), \
             patch.object(native, 'GenerateState', return_value=SimpleNamespace(sampling_params={})), \
             patch.object(native, 'generate_and_rm_group', self.generate), \
             patch('slime.utils.http_utils.get_rollout_num_engines', return_value=4):
            for batch in range(3):
                out = adapter.generate_rollout(self.args, batch, self.source)
                self.assertEqual(len(out.samples), 64)
                worker = self.source.consumers['fully_async']
                time.sleep(.25)  # Simulated training; real native producer keeps running.
                self.assertLessEqual(self.source.issued - self.source.accepted, 64)
                if batch < 2:
                    self.assertEqual(len(self.scored), (batch + 2) * 64)
                worker.pause(drain=False)  # Adapter must force environment-safe drain.
                self.assertEqual(worker.active, 0)
                self.version += 1
                worker.resume()
            self.assertEqual(sorted(self.scored), list(range(192)))
            self.assertEqual(self.source.issued, 192)
            self.assertEqual(self.source.accepted, 192)
            self.assertEqual(self.source.get_samples(1), [])

    def test_busy_lane_cannot_be_admitted_twice(self):
        with patch('turn_stream_rollout.ready_decisions', self.ready), patch.object(adapter, 'validate_request'):
            for _ in range(16):
                self.assertEqual(len(self.source.get_samples(1)), 1)
            self.assertEqual(self.source.get_samples(1), [])

    def test_bad_version_is_rejected_without_new_credit(self):
        self.source.batches = 2
        samples = [SimpleNamespace(index=128+i, metadata={'sampling_weight_version': '1'},
                                  weight_versions=['1']) for i in range(64)]
        with self.assertRaises(AssertionError):
            adapter.accept_batch(self.args, [[s] for s in samples])
        self.assertEqual(self.source.accepted, 0)

    def test_native_importance_sampling_is_active(self):
        import torch
        from slime.backends.megatron_utils.loss import vanilla_tis_function
        args = SimpleNamespace(tis_clip=2., tis_clip_low=0.)
        loss, masks, metrics = vanilla_tis_function(args, pg_loss=torch.ones(3),
            train_log_probs=[torch.tensor([0., 0., 0.])],
            rollout_log_probs=[-torch.log(torch.tensor([.5, 1., 3.]))], loss_masks=[torch.ones(3)])
        torch.testing.assert_close(loss, torch.tensor([.5, 1., 2.]))
        self.assertTrue(torch.isfinite(metrics['tis']).all())


if __name__ == '__main__':
    unittest.main()
