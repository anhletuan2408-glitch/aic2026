from __future__ import annotations

import unittest

from diagnose_qa_signals import round_robin_rankings


class QASignalDiagnosticsTests(unittest.TestCase):
    def test_round_robin_prevents_first_variant_from_consuming_budget(self) -> None:
        self.assertEqual(
            round_robin_rankings([[1, 2, 3, 4], [10, 11], [1, 20]], limit=7),
            [1, 10, 2, 11, 20, 3, 4],
        )

    def test_round_robin_handles_empty_rankings_and_deduplicates(self) -> None:
        self.assertEqual(round_robin_rankings([[], [5, 5, 6]], 10), [5, 6])


if __name__ == "__main__":
    unittest.main()
