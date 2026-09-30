import unittest

try:
    import torch
    from force_processor import ForcedHistoryProcessor
except ModuleNotFoundError:
    torch = None
    ForcedHistoryProcessor = None


class Req:
    def __init__(self, rid="req", ids=()):
        self.rid = rid
        self.output_ids = list(ids)


@unittest.skipIf(torch is None, "torch unavailable")
class ProcessorTests(unittest.TestCase):
    def test_duplicate_empty_output_history_advances_by_invocation(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        param = [{"forced_tokens": [1, 2, 3], "__req__": req}]
        first = processor(torch.zeros(1, 5), param)
        second = processor(torch.zeros(1, 5), param)
        self.assertEqual(int(first.argmax()), 1)
        self.assertEqual(int(second.argmax()), 2)
        self.assertEqual(req.output_ids, [])

    def test_state_is_independent_per_request(self):
        processor = ForcedHistoryProcessor()
        left, right = Req("left"), Req("right")
        params = [
            {"forced_tokens": [3, 4], "__req__": left},
            {"forced_tokens": [1, 2], "__req__": right},
        ]
        first = processor(torch.zeros(2, 6), params)
        second = processor(torch.zeros(2, 6), params)
        self.assertEqual(first.argmax(-1).tolist(), [3, 1])
        self.assertEqual(second.argmax(-1).tolist(), [4, 2])

    def test_logits_are_untouched_after_forced_prefix(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        params = [{"forced_tokens": [2], "__req__": req}]
        processor(torch.zeros(1, 4), params)
        original = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
        result = processor(original.clone(), params)
        self.assertTrue(torch.equal(result, original))

    def test_rejects_changed_sequence_for_live_request(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        processor(torch.zeros(1, 5), [{"forced_tokens": [1, 2], "__req__": req}])
        with self.assertRaisesRegex(RuntimeError, "sequence changed"):
            processor(torch.zeros(1, 5), [{"forced_tokens": [1, 3], "__req__": req}])

    def test_rejects_scheduler_lag_beyond_one_step(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        params = [{"forced_tokens": [1, 2, 3], "__req__": req}]
        processor(torch.zeros(1, 5), params)
        processor(torch.zeros(1, 5), params)
        with self.assertRaisesRegex(RuntimeError, "position contract changed"):
            processor(torch.zeros(1, 5), params)

    def test_accepts_one_step_lag_then_commit(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        params = [{"forced_tokens": [1, 2, 3], "__req__": req}]
        processor(torch.zeros(1, 5), params)
        processor(torch.zeros(1, 5), params)
        req.output_ids.append(1)
        third = processor(torch.zeros(1, 5), params)
        self.assertEqual(int(third.argmax()), 3)

    def test_rejects_repeated_request_rows(self):
        processor = ForcedHistoryProcessor()
        req = Req()
        params = [
            {"forced_tokens": [1], "__req__": req},
            {"forced_tokens": [1], "__req__": req},
        ]
        with self.assertRaisesRegex(RuntimeError, "repeated request rows"):
            processor(torch.zeros(2, 4), params)

    def test_rejects_missing_state_and_out_of_range_token(self):
        processor = ForcedHistoryProcessor()
        with self.assertRaisesRegex(RuntimeError, "state missing"):
            processor(torch.zeros(1, 3), [{}])
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            processor(torch.zeros(1, 3), [{"forced_tokens": [4], "__req__": Req()}])


if __name__ == "__main__":
    unittest.main()
