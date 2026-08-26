"""LLM client abstraction for the ARC-AGI-3 agent.

Three backends, selected by env var ARC_LLM_BACKEND:
  - "mock":     scripted responses for pipeline testing (no deps)
  - "openai":   OpenAI-compatible HTTP endpoint (vLLM server on Kaggle GPU)
  - "hf":       local transformers pipeline (CPU dev / small models)

All backends expose the same .chat(system, user, max_tokens) -> str.
The agent NEVER hard-depends on any backend importing; a failed import
or call raises LLMUnavailable so the caller can fall back to the
programmatic agent.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from typing import Optional


class LLMUnavailable(Exception):
    pass


class MockLLM:
    """Deterministic stub: returns a canned action so the plumbing can be
    exercised end-to-end without any model."""

    def __init__(self, script: Optional[list[str]] = None) -> None:
        self._script = script or []
        self._i = 0

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        if self._i < len(self._script):
            out = self._script[self._i]
            self._i += 1
            return out
        # default: pick the first listed valid action
        import re
        m = re.search(r"VALID ACTIONS:\s*([^\n]+)", user)
        acts = m.group(1).split(",") if m else ["RESET"]
        first = acts[0].strip().split("(")[0]
        return f'Reasoning: probe.\nACTION: {first}'


class OpenAICompatLLM:
    """Talks to a local vLLM OpenAI-compatible server (Kaggle GPU)."""

    def __init__(self) -> None:
        self.base = os.environ.get("ARC_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = os.environ.get("ARC_LLM_MODEL", "model")
        self.key = os.environ.get("ARC_LLM_KEY", "EMPTY")
        self.timeout = float(os.environ.get("ARC_LLM_TIMEOUT", "25"))

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }).encode()
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"openai backend: {e}")


class HFLocalLLM:
    """Local transformers pipeline (small model, CPU dev)."""

    def __init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"hf import: {e}")
        self.model_id = os.environ.get("ARC_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto")
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"hf load: {e}")

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        try:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(text, return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self.tok.decode(gen, skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"hf generate: {e}")


def make_client():
    backend = os.environ.get("ARC_LLM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAICompatLLM()
    if backend == "hf":
        return HFLocalLLM()
    raise LLMUnavailable(f"unknown backend {backend}")
