"""Model backends.

  anthropic      Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY)
  openai_compat  any OpenAI compatible server: vLLM, Ollama, LM Studio, OpenRouter,
                 or the OpenAI API. Use this to rerun the Qwen, Llama and Mistral
                 models from Manoranjan and Gaikwad.
  ollama         local models through Ollama, the same server the team's RAG app
                 uses (e.g. qwen2.5:32b, llama3.1:8b, mistral). JSON mode on.
  mock           deterministic rule based stand in. It exists only to test the
                 plumbing offline. Its outputs are NOT findings about LLMs.
"""
from __future__ import annotations

import json
import os
import re
import time


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles code fences and prose)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object")


class LLM:
    name = "base"

    def __init__(self, model: str = "", temperature: float = 0.7, max_tokens: int = 1500, **_):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.calls = 0

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, meta: dict | None = None, retries: int = 2) -> dict:
        """Ask for a JSON object; on a parse failure, show the model its error and retry."""
        prompt = user
        last = None
        for attempt in range(retries + 1):
            raw = self._complete(system, prompt)
            self.calls += 1
            try:
                return extract_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                prompt = user + f"\n\nYour previous reply could not be parsed as JSON ({e}). Reply with one valid JSON object only."
        raise RuntimeError(f"{self.name} failed to return JSON: {last}")


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, **kw):
        super().__init__(**kw)
        import anthropic
        self.client = anthropic.Anthropic()

    def _complete(self, system, user):
        for wait in (2, 5, 15, 0):
            try:
                r = self.client.messages.create(model=self.model, system=system, max_tokens=self.max_tokens,
                                                temperature=self.temperature,
                                                messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception:
                if not wait:
                    raise
                time.sleep(wait)


class OpenAICompatLLM(LLM):
    name = "openai_compat"

    def __init__(self, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", **kw):
        super().__init__(**kw)
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "not-needed"))

    def _complete(self, system, user):
        for wait in (2, 5, 15, 0):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            except Exception:
                if not wait:
                    raise
                time.sleep(wait)


class OllamaLLM(LLM):
    """Native Ollama client. num_ctx matters: Ollama's default context window is
    small and silently truncates long prompts, and a delegate prompt with an
    8 passage briefing and a public record runs to several thousand tokens."""
    name = "ollama"

    def __init__(self, host: str | None = None, num_ctx: int = 16384, json_mode: bool = True, **kw):
        super().__init__(**kw)
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.num_ctx, self.json_mode = num_ctx, json_mode

    def _complete(self, system, user):
        for wait in (2, 5, 0):
            try:
                r = self.client.chat(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    format="json" if self.json_mode else "",
                    options={"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens})
                return r["message"]["content"]
            except Exception:
                if not wait:
                    raise
                time.sleep(wait)


class MockLLM(LLM):
    """Calls meta['mock'](), a rule based policy supplied by the calling agent."""
    name = "mock"

    def complete_json(self, system, user, meta=None, retries=0):
        self.calls += 1
        if not meta or "mock" not in meta:
            return {}
        return meta["mock"]()


def make_llm(spec: dict, temperature: float = 0.7) -> LLM:
    kind = spec.get("type", "mock")
    kw = {k: v for k, v in spec.items() if k not in {"type", "name"}}
    kw.setdefault("temperature", temperature)
    if kind == "anthropic":
        return AnthropicLLM(**kw)
    if kind == "openai_compat":
        return OpenAICompatLLM(**kw)
    if kind == "ollama":
        return OllamaLLM(**kw)
    if kind == "mock":
        return MockLLM(**kw)
    raise ValueError(f"unknown backend type {kind}")
