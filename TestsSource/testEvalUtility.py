import unittest
import numpy as np
from numpy.testing import assert_array_equal

from Utilities.Eval import MetricsLogger


class Metrics:
    def __init__(self):
        self.cat_count = 0
        self.dog_count = 0


class TestMetricsCollector(unittest.TestCase):

    def test_sanity(self):
        """Perform simple sanity checks"""
        metrics = Metrics()
        collector = MetricsLogger(metrics)

        self.assertEqual(2, len(collector.data))
        self.assertEqual([], collector.data["cat_count"])
        self.assertEqual([], collector.data["dog_count"])

        # Add 5
        metrics.cat_count = 5
        metrics.dog_count = 7
        collector.append(metrics)

        self.assertEqual(2, len(collector.data))
        self.assertEqual([5], collector.data["cat_count"])
        self.assertEqual(5, collector.min["cat_count"])
        self.assertEqual(5, collector.max["cat_count"])
        self.assertEqual([7], collector.data["dog_count"])
        self.assertEqual(7, collector.min["dog_count"])
        self.assertEqual(7, collector.max["dog_count"])

        # Add 10
        metrics.cat_count = 10
        metrics.dog_count = 2
        collector.append(metrics)

        self.assertEqual(2, len(collector.data))
        assert_array_equal([5, 10], collector.data["cat_count"])
        self.assertEqual(5, collector.min["cat_count"])
        self.assertEqual(10, collector.max["cat_count"])
        assert_array_equal([7, 2], collector.data["dog_count"])
        self.assertEqual(2, collector.min["dog_count"])
        self.assertEqual(7, collector.max["dog_count"])

    def test_rollover(self):
        """Test that rollover works correctly when the maximum size of the buffer is reached"""
        metrics = Metrics()
        max_length = 20
        collector = MetricsLogger(metrics, max_length)

        # Fill collector to maximum length
        added_count = 0
        for i in range(max_length):
            metrics.cat_count = i
            metrics.dog_count = i + 100
            collector.append(metrics)

            # Check invariants
            added_count += 1
            self.assertEqual(max(0, added_count - max_length), collector.lower_bound)
            self.assertEqual(added_count-1, collector.upper_bound)
            self.assertEqual(min(added_count, max_length), collector.count)
        assert_array_equal(np.arange(max_length), collector.data["cat_count"])
        assert_array_equal(np.arange(100, 100 + max_length), collector.data["dog_count"])

        # Fill collector to maximum length one more time
        for i in range(max_length):
            metrics.cat_count = i + max_length
            metrics.dog_count = i + 100 + max_length
            collector.append(metrics)

            # Check invariants
            added_count += 1
            self.assertEqual(max(0, added_count - max_length), collector.lower_bound)
            self.assertEqual(added_count - 1, collector.upper_bound)
            self.assertEqual(min(added_count, max_length), collector.count)

        assert_array_equal(collector.data["cat_count"], np.arange(max_length, 2 * max_length))
        assert_array_equal(collector.data["dog_count"], np.arange(max_length + 100, 2 * max_length + 100))

        # Fill collector to maximum length a third time more time
        for i in range(max_length):
            metrics.cat_count = i + 2 * max_length
            metrics.dog_count = i + 100 + 2 * max_length
            collector.append(metrics)

            # Check invariants
            added_count += 1
            self.assertEqual(max(0, added_count - max_length), collector.lower_bound)
            self.assertEqual(added_count - 1, collector.upper_bound)
            self.assertEqual(min(added_count, max_length), collector.count)

        assert_array_equal(collector.data["cat_count"], np.arange(2 * max_length, 3 * max_length))
        assert_array_equal(collector.data["dog_count"], np.arange(2 * max_length + 100, 3 * max_length + 100))


if __name__ == '__main__':
    unittest.main()
