"""End-to-end smoke test: a mock Ollama server + the real Brain pipeline.

Starts a tiny HTTP server that mimics Ollama's /api/version, /api/tags and
/api/chat, then drives harvey.brain.Brain (provider=ollama) through
preflight, think, and think_json against it — including the usage ledger.
"""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from harvey.brain import Brain
from harvey.config import BrainConfig
from harvey.state import StateManager

PORT = 11499
CALLS = {"chat": 0}


class MockOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.9.9-mock"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "qwen2.5:14b"}, {"name": "llama3.1:8b"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/api/chat":
            self._send({"error": "not found"}, 404)
            return
        CALLS["chat"] += 1
        prompt = req["messages"][-1]["content"]
        if "HARVEY_READY" in prompt:
            reply = "HARVEY_READY"
        elif "ONLY with valid JSON" in prompt:
            reply = '{"score": 8, "reasons": ["hiring a dispatcher", "posted on LinkedIn"]}'
        else:
            reply = "Hello from the mock Ollama model."
        self._send({
            "model": req.get("model", "qwen2.5:14b"),
            "message": {"role": "assistant", "content": reply},
            "prompt_eval_count": 120,
            "eval_count": 25,
            "done": True,
        })


def main():
    server = HTTPServer(("127.0.0.1", PORT), MockOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    import tempfile, os
    db = os.path.join(tempfile.mkdtemp(), "smoke.db")
    state = StateManager(db_path=db)

    async def run():
        await state.init_db()
        brain = Brain(state, BrainConfig(provider="ollama", model="qwen2.5:14b",
                                         base_url=f"http://127.0.0.1:{PORT}"))

        # 1) preflight: server answers, model pulled
        problem = await brain.preflight()
        assert problem == "", f"preflight failed: {problem}"
        print("PASS  preflight (server + model checked, no prompt sent)")

        # 2) plain think
        out = await brain.think("Say hello", session_id="harvey-smoke",
                                agent="writer", task="smoke")
        assert out == "Hello from the mock Ollama model.", out
        print("PASS  think (plain text round-trip)")

        # 3) think_json
        data = await brain.think_json("Score this", session_id="harvey-smoke-score",
                                      agent="scout", task="score")
        assert isinstance(data, dict) and data["score"] == 8, data
        print("PASS  think_json (structured JSON parsed)")

        # 4) usage ledger recorded under provider=ollama
        calls = await state.get_usage_today(provider="ollama")
        assert calls == 2, f"expected 2 ollama calls, got {calls}"
        print("PASS  usage ledger (2 calls counted under provider=ollama)")

        # 5) budget gate: no subscription quota -> falls back to call count
        ok = await brain.is_within_budget(max_daily_calls=10, max_percent=80)
        assert ok is True
        print("PASS  budget gate (falls back to daily call count, no quota)")

        # 6) preflight for a model that is NOT pulled -> clean setup error
        brain2 = Brain(state, BrainConfig(provider="ollama", model="mistral:7b",
                                          base_url=f"http://127.0.0.1:{PORT}"))
        reason = await brain2.preflight()
        assert "mistral:7b" in reason and "ollama pull" in reason, reason
        print("PASS  preflight missing model (actionable 'ollama pull' message)")

        # 7) a call while paused returns "" without crashing
        brain2.unavailable = "paused"
        out = await brain2.think("hi", max_retries=0)
        assert out == ""
        print("PASS  paused provider (call skipped, returns empty)")

        server.shutdown()
        print(f"\nALL SMOKE TESTS PASSED (mock /api/chat hit {CALLS['chat']} times)")

    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())