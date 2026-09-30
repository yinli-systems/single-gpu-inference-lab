from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class ForcedHistoryProcessor(CustomLogitProcessor):
    """Force a frozen token history using scheduler-call order, not lagging output_ids.

    SGLang overlap scheduling can invoke a custom processor for the next step before
    ``Req.output_ids`` reflects the token selected by the preceding invocation.  The
    diagnostic therefore stores an invocation counter on the scheduler-owned ``Req``.
    This processor is intentionally unsupported for multi-token speculative rows.
    """

    _COUNT_ATTR = "_sgi_forced_history_call_count"
    _TOKENS_ATTR = "_sgi_forced_history_tokens"
    _VISIBLE_ATTR = "_sgi_forced_history_last_visible"

    def __call__(self, logits, custom_param_list):
        if len(custom_param_list) != logits.shape[0]:
            raise RuntimeError("processor row mismatch")
        seen = set()
        for row, param in enumerate(custom_param_list):
            req = param.get("__req__")
            forced = param.get("forced_tokens")
            if req is None or not isinstance(forced, list):
                raise RuntimeError("forced-history state missing")
            req_id = id(req)
            if req_id in seen:
                raise RuntimeError("forced-history processor does not support repeated request rows")
            seen.add(req_id)
            tokens = tuple(int(token) for token in forced)
            previous = getattr(req, self._TOKENS_ATTR, None)
            if previous is None:
                setattr(req, self._TOKENS_ATTR, tokens)
                setattr(req, self._COUNT_ATTR, 0)
            elif previous != tokens:
                raise RuntimeError("forced-history token sequence changed for live request")
            position = int(getattr(req, self._COUNT_ATTR))
            visible = len(req.output_ids)
            last_visible = int(getattr(req, self._VISIBLE_ATTR, visible))
            if visible < last_visible or visible > position or position - visible > 1:
                raise RuntimeError("forced-history scheduler position contract changed")
            setattr(req, self._VISIBLE_ATTR, visible)
            setattr(req, self._COUNT_ATTR, position + 1)
            if position >= len(tokens):
                continue
            token = tokens[position]
            if token < 0 or token >= logits.shape[-1]:
                raise RuntimeError("forced token out of range")
            logits[row, :] = -float("inf")
            logits[row, token] = 0.0
        return logits
