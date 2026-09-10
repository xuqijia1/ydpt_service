import importlib
import sys
from pathlib import Path
import unittest

import numpy as np


SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))
video_processor = importlib.import_module("video_processor")


class SelectStepFrameTests(unittest.TestCase):
    def test_aipp_uses_the_matching_bgr_frame_for_step_images(self):
        black_visualization = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr_frame = np.full((2, 2, 3), 127, dtype=np.uint8)

        self.assertTrue(hasattr(video_processor, "select_step_frame"))
        selected = video_processor.select_step_frame(
            black_visualization, bgr_frame, use_aipp=True
        )

        self.assertIs(selected, bgr_frame)

    def test_aipp_without_a_bgr_frame_does_not_use_the_black_visualization(self):
        black_visualization = np.zeros((2, 2, 3), dtype=np.uint8)

        selected = video_processor.select_step_frame(
            black_visualization, None, use_aipp=True
        )

        self.assertIsNone(selected)

    def test_legacy_inference_result_has_no_evidence_frame(self):
        detections = [{"class": "gantry"}]
        vis_frame = np.full((2, 2, 3), 127, dtype=np.uint8)

        self.assertTrue(hasattr(video_processor, "unpack_inference_result"))
        unpacked = video_processor.unpack_inference_result((detections, vis_frame))

        self.assertEqual(unpacked, (detections, vis_frame, None))


if __name__ == "__main__":
    unittest.main()
