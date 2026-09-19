from fractions import Fraction
import unittest

import numpy as np

from dlss_engine.frame_interpolation.images import NearestTimestampCollector
from dlss_engine.frame_interpolation.models import resolve_source_rate
from dlss_engine.frame_interpolation.processor import TimedFrame


def frame(timestamp, provenance, source_index=None):
    return TimedFrame(
        np.zeros((2, 2, 4), dtype=np.uint8),
        timestamp,
        0,
        provenance,
        source_index,
    )


class ImageSequenceTimelineTests(unittest.TestCase):
    def test_common_image_source_rates_are_supported(self):
        self.assertEqual(resolve_source_rate("8"), Fraction(8))
        self.assertEqual(resolve_source_rate("12"), Fraction(12))
        self.assertEqual(resolve_source_rate("16"), Fraction(16))
        self.assertEqual(resolve_source_rate("20"), Fraction(20))

    def test_selects_a_half_open_image_timeline(self):
        collector = NearestTimestampCollector(Fraction(60), 4)
        collector.push(frame(Fraction(0), "Source", 0))
        collector.push(frame(Fraction(1, 60), "DLSSG"))
        collector.push(frame(Fraction(1, 30), "Source", 1))
        collector.finish()

        self.assertEqual(len(collector.frames), 4)
        self.assertEqual(collector.generated, 1)
        self.assertEqual(collector.copied, 3)
        self.assertEqual(collector.selected_real_ids, {0, 1})
        self.assertLessEqual(collector.max_error, Fraction(1, 60))

    def test_finish_requires_at_least_one_frame(self):
        with self.assertRaises(ValueError):
            NearestTimestampCollector(Fraction(60), 1).finish()


if __name__ == "__main__":
    unittest.main()
