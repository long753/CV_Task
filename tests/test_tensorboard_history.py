"""History import checks against actual TensorBoard event files, without GPU training."""

import json
import tempfile
import unittest
from pathlib import Path

from script.backfill_tensorboard import backfill_tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorBoardHistoryTests(unittest.TestCase):
    def test_committed_epochs_keep_native_steps_and_values_without_duplicate_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "training_state.json").write_text(json.dumps({"completed_epochs": 2}))
            (run / "results.csv").write_text(
                "epoch,time,train/box_loss,val/box_loss\n"
                "1,10,2.0,3.0\n2,20,1.5,2.5\n3,30,9.0,9.0\n"
            )
            backfill_tensorboard(run)
            events = EventAccumulator(str(run)).Reload()
            self.assertEqual([item.step for item in events.Scalars("train/box_loss")], [1, 2])
            self.assertEqual([item.value for item in events.Scalars("train/box_loss")], [2.0, 1.5])
            self.assertEqual([item.value for item in events.Scalars("val/box_loss")], [3.0, 2.5])
            with self.assertRaises(FileExistsError):
                backfill_tensorboard(run)
            events.Reload()
            self.assertEqual([item.step for item in events.Scalars("train/box_loss")], [1, 2])

    def test_missing_completed_epoch_is_rejected_before_publishing_a_partial_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "training_state.json").write_text(json.dumps({"completed_epochs": 3}))
            (run / "results.csv").write_text("epoch,time,train/box_loss\n1,10,2\n3,30,1\n")
            with self.assertRaisesRegex(ValueError, "every completed epoch"):
                backfill_tensorboard(run)
            self.assertEqual(list(run.glob("events.out.tfevents.*")), [])


if __name__ == "__main__":
    unittest.main()
