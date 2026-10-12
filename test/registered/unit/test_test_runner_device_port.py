"""CPU regression for UUID, ordinal and hidden-device test port selection."""

import unittest

from sglang.test.test_utils import _test_runner_device_slot


class TestRunnerDevicePort(unittest.TestCase):
    def test_numeric_masks_preserve_normal_offsets(self):
        for first in range(8):
            self.assertEqual(_test_runner_device_slot(f"{first},7"), first)

    def test_uuid_masks_are_stable_and_bounded(self):
        for uuid in (
            "GPU-be55ff73-c362-7f9a-b8d2-58fcdd82b9ca",
            "MIG-GPU-be55ff73-c362-7f9a-b8d2-58fcdd82b9ca/1/0",
        ):
            slot = _test_runner_device_slot(uuid)
            self.assertEqual(slot, _test_runner_device_slot(uuid + ",GPU-another"))
            self.assertGreaterEqual(slot, 0)
            self.assertLess(slot, 16)

    def test_hidden_devices_and_large_ordinals_stay_bounded(self):
        for mask in ("", "-1", "   "):
            self.assertEqual(_test_runner_device_slot(mask), 0)
        self.assertLess(_test_runner_device_slot("999999"), 16)


if __name__ == "__main__":
    unittest.main()
