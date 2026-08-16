from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from benchmark_kis_live import call_search_api


class BenchmarkKisLiveTests(unittest.TestCase):
    @patch("benchmark_kis_live.urllib.request.urlopen")
    def test_search_payload_controls_each_ablation_signal(self, urlopen) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"results":[{"video_id":"V","frame_idx":1}]}'
        urlopen.return_value = response

        rows = call_search_api(
            "http://127.0.0.1:7860", "red car",
            quality=False, use_ocr=False, use_hybrid=True,
        )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(rows[0]["frame_idx"], 1)
        self.assertEqual(
            (payload["quality"], payload["use_ocr"], payload["use_hybrid"]),
            (False, False, True),
        )
        self.assertEqual(payload["top_k"], 100)


if __name__ == "__main__":
    unittest.main()
