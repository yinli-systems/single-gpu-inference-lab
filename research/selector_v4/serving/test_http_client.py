"""Actual localhost SSE transport; no model/GPU/serving qualification."""

import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from research.selector_v4.serving.http_client import collect_block


def test_actual_localhost_complete_error_and_truncated_responses_are_all_retained():
    async def run():
        seen = []

        async def generate(request):
            payload = await request.json()
            seen.append(payload["rid"])
            marker = payload["input_ids"][0]
            if marker == 500:
                return web.Response(status=500, text="intentional worker failure")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for index, token in enumerate((7, 9)):
                await asyncio.sleep(0.01)
                meta = {"completion_tokens": index + 1, "cached_tokens": 8192}
                if index == 1:
                    meta["finish_reason"] = {"type": "length", "length": 2}
                raw = b"data: " + json.dumps({"output_ids": [token], "meta_info": meta}).encode()
                await response.write(raw[:9])
                await response.write(raw[9:] + b"\n\n")
            if marker != 499:
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"

        def work(markers):
            return {
                "name": "local-transport",
                "concurrency": 3,
                "cells": [
                    {"id": str(m), "input_ids": [m], "output_tokens": 2, "expect_cached": 8192}
                    for m in markers
                ],
            }

        try:
            async with aiohttp.ClientSession() as session:
                complete = await collect_block(session, url, work((1, 2)), "complete")
                failed = await collect_block(session, url, work((1, 499, 500)), "failed")
            assert complete["complete"] and complete["output_tokens"] == 4
            assert all(r["tokens"] == [7, 9] and r["tpot"] > 0 for r in complete["requests"])
            assert complete["cached_min"] == 8192 and complete["TTFT-p99"] > 0
            assert len(seen) == len(set(seen)) == 5
            assert not failed["complete"] and len(failed["requests"]) == 3
            assert len(failed["errors"]) == 2 and "output_tokens_per_second" not in failed
            records = {r["id"]: r for r in failed["requests"]}
            assert records["1"]["complete"]
            assert records["499"]["tokens"] == [7, 9]
            assert len(records["499"]["raw_events"]) == 2
            assert records["500"]["error_body"] == "intentional worker failure"
            assert records["500"]["http_status"] == 500
            assert "Incomplete" in records["499"]["error"]["message"]
        finally:
            await runner.cleanup()

    asyncio.run(run())
