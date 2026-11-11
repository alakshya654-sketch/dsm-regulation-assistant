"""
LLM layer: one interface, two free providers (Design §4.8, decision D-7).

    load_env(PROJECT_ROOT)                 # reads .env into os.environ (key never printed)
    llm = make_client(load_llm_settings(PROJECT_ROOT / "config.yaml"))
    reply = llm.generate(system="...", user="...")
    reply.text, reply.provider, reply.model, reply.latency_s

    # with a tool (Phase 5): the model may call it; the handler runs it and the result goes back
    reply = llm.generate(system, user, tools=[TOOL_SPEC], tool_handler=lambda name, args: {...})
    reply.tool_calls     # [{"name", "args", "result"}, ...]

The rest of the project only uses `generate()`, so switching provider or model is a
config change, not a code change.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VERSION = "5.1"

DEFAULTS = {
    "provider": "gemini",                       # gemini | groq
    "gemini_models": ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"],
    "groq_model": "llama-3.3-70b-versatile",
    "fallback": "groq",                         # used only if its key is present; "" to disable
    "temperature": 0.0,
    "max_output_tokens": 4096,
    "thinking_level": "low",                    # Gemini 3.x reasoning effort: minimal | low | medium | high | "" (model default)
    "max_retries": 4,
}


def load_llm_settings(config_path: Path | None = None) -> dict:
    s = dict(DEFAULTS)
    if config_path and Path(config_path).exists():
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        s.update(cfg.get("llm", {}))
    return s


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------
def parse_env(text: str) -> dict[str, str]:
    """Tolerant .env parser: ignores a BOM, comments and blank lines; trims spaces around
    the name, '=' and value; removes surrounding quotes. ('GEMINI_API_KEY = "AQ.x"' works.)"""
    out = {}
    for line in text.lstrip("﻿").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def load_env(project_root: Path, override: bool = True) -> dict[str, bool]:
    """Read <project>/.env into os.environ. Returns {name: is_set} — never the values."""
    path = Path(project_root) / ".env"
    if not path.exists():
        return {}
    values = parse_env(path.read_text(encoding="utf-8"))
    for k, v in values.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return {k: bool(v) for k, v in values.items()}


def mask(key: str | None) -> str:
    return f"{key[:4]}… ({len(key)} chars)" if key else "not set"


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None              # tokens the model spent reasoning before answering
    attempts: int = 1
    notes: list[str] = field(default_factory=list)   # e.g. "fell back from gemini: quota"
    tool_calls: list[dict] = field(default_factory=list)   # [{"name", "args", "result"}] in call order


MAX_TOOL_ROUNDS = 3        # tool calls per question before the model must answer in text


def _run_handler(handler, name: str, args: dict, log: list) -> dict:
    """Run one tool call; a handler error becomes data for the model, never a crash."""
    try:
        result = handler(name, args) if handler else {"ok": False, "error": "no handler"}
    except Exception as e:                      # noqa: BLE001 — reported back to the model
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not isinstance(result, dict):
        result = {"result": result}
    log.append({"name": name, "args": args, "result": result})
    return result


class LLMError(Exception):
    """Raised when a provider cannot produce an answer. `kind` is one of
    auth | quota | not_found | server | network | empty | other."""

    def __init__(self, message: str, kind: str = "other", provider: str = "", model: str = ""):
        super().__init__(message)
        self.kind, self.provider, self.model = kind, provider, model


def _retry_delay(message: str, attempt: int) -> float:
    """Use the server's suggested delay if it gives one ("retry in 23s", "retryDelay": "23s"),
    otherwise exponential backoff: 2, 4, 8, 16 s."""
    m = re.search(r"retry(?:Delay)?[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", message, re.I)
    return min(float(m.group(1)) + 1, 60) if m else min(2 ** (attempt + 1), 30)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
class GeminiClient:
    """Google Gemini via the official `google-genai` SDK. Tries the configured models in
    order: if one is unavailable to this key (404) or out of free quota, the next is used."""

    provider = "gemini"

    def __init__(self, models: list[str], api_key: str | None = None, temperature: float = 0.0,
                 max_output_tokens: int = 4096, max_retries: int = 4, sleep=time.sleep,
                 thinking_level: str = "low", clock=time.monotonic):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (check the .env file)", "auth", self.provider)
        self.client = genai.Client(api_key=key)      # key passed explicitly; format not checked
        self.models = list(models)
        self.temperature, self.max_output_tokens, self.max_retries = temperature, max_output_tokens, max_retries
        self.sleep = sleep
        # Models that rejected the thinking setting; they are called without it from then on.
        self.thinking_level = (thinking_level or "").upper()
        self.no_thinking_config: set[str] = set()
        # Models out of free quota are skipped until this time (clock seconds), so one exhausted
        # model doesn't cost every later question a wait (owner's run 2: up to 621 s/answer).
        self.clock = clock
        self.cooldown_until: dict[str, float] = {}

    def _order(self) -> list[str]:
        """Models to try, in config order, skipping those cooling down. If all are cooling
        down, try the one that recovers soonest first."""
        now = self.clock()
        ready = [m for m in self.models if self.cooldown_until.get(m, 0) <= now]
        return ready or sorted(self.models, key=lambda m: self.cooldown_until.get(m, 0))

    @staticmethod
    def _kind(err) -> str:
        code = getattr(err, "code", None)
        text = str(err).lower()
        if code in (401, 403) or "api key" in text or "permission" in text or "unauthenticated" in text:
            return "auth"
        if code == 429 or "resource_exhausted" in text or "quota" in text:
            return "quota"
        if code == 404 or "not found" in text or "not supported" in text:
            return "not_found"
        if code and code >= 500:
            return "server"
        return "other"

    def _config(self, model: str, system: str, tools: list[dict] | None = None):
        from google.genai import types
        extra = {}
        if self.thinking_level and model not in self.no_thinking_config:
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        if tools:
            extra["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"]) for t in tools])]
        return types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature, max_output_tokens=self.max_output_tokens,
            # Tools (if any) are declarations only: our code runs them (see generate), so the SDK's
            # automatic function calling stays off. Saying so also silences its warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **extra)

    def _call(self, model: str, system: str, contents, tools: list[dict] | None = None):
        return self.client.models.generate_content(model=model, contents=contents,
                                                   config=self._config(model, system, tools))

    def generate(self, system: str, user: str, tools: list[dict] | None = None,
                 tool_handler=None) -> LLMResponse:
        """One answer. With `tools`, the model may request a tool; `tool_handler(name, args)` runs it,
        the result is sent back, and the model answers (at most MAX_TOOL_ROUNDS tool rounds)."""
        if not tools:
            return self._generate(system, user)
        from google.genai import types
        contents = [types.Content(role="user", parts=[types.Part(text=user)])]
        log, notes, total = [], [], 0.0
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            reply, resp = self._generate(system, contents, tools, raw=True)
            notes += [n for n in reply.notes if n not in notes]
            total += reply.latency_s
            calls = (getattr(resp, "function_calls", None) or []) if round_no < MAX_TOOL_ROUNDS else []
            if not calls:
                reply.tool_calls, reply.notes, reply.latency_s = log, notes, total
                if not reply.text:
                    raise LLMError(f"{reply.model}: empty reply after tool use", "empty", self.provider, reply.model)
                return reply
            contents.append(resp.candidates[0].content)
            parts = [types.Part.from_function_response(name=c.name,
                                                       response=_run_handler(tool_handler, c.name, dict(c.args or {}), log))
                     for c in calls]
            contents.append(types.Content(role="user", parts=parts))
        raise LLMError("tool loop did not finish", "other", self.provider)

    def _generate(self, system: str, contents, tools: list[dict] | None = None, raw: bool = False):
        """Retry policy (v4c.3):
        - quota (429): note the model's cooldown; switch to the next model at once. Only the last
          model left is waited for (server's suggested delay) and retried.
        - server busy (5xx): one quick retry, then the next model; the last model gets full retries.
        - not found (404) / empty reply: next model.  Bad key: stop.  Network error: retry, then stop."""
        from google.genai import errors
        notes, last = [], None
        order = self._order()
        skipped = [m for m in self.models if m not in order]
        if skipped:
            notes.append(f"skipped (out of quota, cooling down): {', '.join(skipped)}")
        for i, model in enumerate(order):
            is_last = i == len(order) - 1
            for attempt in range(self.max_retries + 1):
                start = time.time()
                try:
                    resp = self._call(model, system, contents, tools)
                except errors.APIError as e:
                    if (getattr(e, "code", None) == 400 and "think" in str(e).lower()
                            and model not in self.no_thinking_config):
                        # This model doesn't accept the thinking setting: retry once without it.
                        self.no_thinking_config.add(model)
                        notes.append(f"{model}: thinking setting not supported, using model default")
                        continue
                    kind = self._kind(e)
                    last = LLMError(f"{model}: {e}", kind, self.provider, model)
                    if kind == "auth":
                        raise last                      # a bad key won't fix itself
                    if kind == "quota":
                        delay = _retry_delay(str(e), attempt)
                        self.cooldown_until[model] = self.clock() + delay
                        if not is_last:
                            notes.append(f"{model}: out of quota, switching to next model")
                            break
                        if attempt < self.max_retries:
                            notes.append(f"{model}: out of quota, retry in {delay:.0f}s")
                            self.sleep(delay)
                            continue
                        notes.append(f"{model}: out of quota")
                        break
                    if kind == "server":
                        allowed = self.max_retries if is_last else 1
                        if attempt < allowed:
                            delay = _retry_delay(str(e), attempt)
                            notes.append(f"{model}: server busy, retry in {delay:.0f}s")
                            self.sleep(delay)
                            continue
                        notes.append(f"{model}: server busy, trying next model")
                        break
                    notes.append(f"{model}: {kind}, trying next model")
                    break                               # not_found / other → next model
                except Exception as e:                  # network trouble: timeout, DNS, proxy, reset
                    last = LLMError(f"{model}: {type(e).__name__}: {e}", "network", self.provider, model)
                    if attempt < self.max_retries:
                        delay = _retry_delay("", attempt)
                        notes.append(f"{model}: network error ({type(e).__name__}), retry in {delay:.0f}s")
                        self.sleep(delay)
                        continue
                    raise last                          # same network for every model: stop here
                has_calls = bool(tools) and bool(getattr(resp, "function_calls", None))
                text = "" if has_calls else (getattr(resp, "text", None) or "").strip()
                if not text and not has_calls:
                    reason = ""
                    try:
                        reason = str(resp.candidates[0].finish_reason)
                    except Exception:
                        pass
                    last = LLMError(f"{model}: empty reply {reason}".strip(), "empty", self.provider, model)
                    notes.append(f"{model}: empty reply {reason}".strip())
                    break
                usage = getattr(resp, "usage_metadata", None)
                reply = LLMResponse(text, self.provider, model, time.time() - start,
                                    getattr(usage, "prompt_token_count", None),
                                    getattr(usage, "candidates_token_count", None),
                                    getattr(usage, "thoughts_token_count", None),
                                    attempt + 1, notes)
                return (reply, resp) if raw else reply
        raise last or LLMError("no Gemini model configured", "other", self.provider)

    def list_models(self) -> list[str]:
        """Models this key can use for text generation (for diagnosing a 404)."""
        names = []
        for m in self.client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(m.name.removeprefix("models/"))
        return sorted(names)


# ---------------------------------------------------------------------------
# Groq (fallback)
# ---------------------------------------------------------------------------
class GroqClient:
    provider = "groq"

    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.0,
                 max_output_tokens: int = 4096, max_retries: int = 4, sleep=time.sleep):
        from groq import Groq
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set", "auth", self.provider)
        self.client = Groq(api_key=key)
        self.model, self.temperature, self.max_output_tokens, self.max_retries = model, temperature, max_output_tokens, max_retries
        self.sleep = sleep

    def generate(self, system: str, user: str, tools: list[dict] | None = None,
                 tool_handler=None) -> LLMResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if not tools:
            return self._generate(messages)
        spec = [{"type": "function", "function": t} for t in tools]
        log, notes, total = [], [], 0.0
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            reply, msg = self._generate(messages, spec if round_no < MAX_TOOL_ROUNDS else None, raw=True)
            notes += reply.notes
            total += reply.latency_s
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                reply.tool_calls, reply.notes, reply.latency_s = log, notes, total
                if not reply.text:
                    raise LLMError(f"{self.model}: empty reply after tool use", "empty", self.provider, self.model)
                return reply
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _run_handler(tool_handler, c.function.name, args, log)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, default=str)})
        raise LLMError("tool loop did not finish", "other", self.provider, self.model)

    def _generate(self, messages: list[dict], tools: list[dict] | None = None, raw: bool = False):
        import groq
        notes = []
        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                extra = {"tools": tools} if tools else {}
                r = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=self.max_output_tokens,
                    messages=messages, **extra)
            except groq.AuthenticationError as e:
                raise LLMError(f"{self.model}: {e}", "auth", self.provider, self.model)
            except (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError) as e:
                kind = "quota" if isinstance(e, groq.RateLimitError) else "server"
                if attempt < self.max_retries:
                    delay = _retry_delay(str(e), attempt)
                    notes.append(f"{self.model}: {kind}, retry in {delay:.0f}s")
                    self.sleep(delay)
                    continue
                raise LLMError(f"{self.model}: {e}", kind, self.provider, self.model)
            except groq.APIStatusError as e:
                kind = "not_found" if e.status_code == 404 else "other"
                raise LLMError(f"{self.model}: {e}", kind, self.provider, self.model)
            msg = r.choices[0].message
            text = (msg.content or "").strip()
            if not text and not (tools and getattr(msg, "tool_calls", None)):
                raise LLMError(f"{self.model}: empty reply", "empty", self.provider, self.model)
            u = getattr(r, "usage", None)
            reply = LLMResponse(text, self.provider, self.model, time.time() - start,
                                getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                                None, attempt + 1, notes)
            return (reply, msg) if raw else reply
        raise LLMError("unreachable", "other", self.provider, self.model)


# ---------------------------------------------------------------------------
# Primary + fallback
# ---------------------------------------------------------------------------
class FallbackClient:
    """Try the primary provider; on any failure except a bad request, try the fallback.
    The reply's `notes` record that a fallback happened, so evaluation can report it."""

    def __init__(self, primary, fallback=None):
        self.primary, self.fallback = primary, fallback
        self.provider = primary.provider

    def generate(self, system: str, user: str, tools: list[dict] | None = None,
                 tool_handler=None) -> LLMResponse:
        kw = {"tools": tools, "tool_handler": tool_handler} if tools else {}
        try:
            return self.primary.generate(system, user, **kw)
        except LLMError as e:
            if self.fallback is None:
                raise
            reply = self.fallback.generate(system, user, **kw)
            reply.notes.insert(0, f"fell back from {e.provider} ({e.kind}): {str(e)[:120]}")
            return reply


def make_client(settings: dict | None = None):
    s = settings or dict(DEFAULTS)
    common = dict(temperature=s["temperature"], max_output_tokens=s["max_output_tokens"], max_retries=s["max_retries"])

    def build(provider):
        if provider == "gemini":
            return GeminiClient(s["gemini_models"], thinking_level=s.get("thinking_level", "low"), **common)
        if provider == "groq":
            return GroqClient(s["groq_model"], **common)
        raise ValueError(f"unknown provider {provider!r}")

    primary = build(s["provider"])
    fallback = None
    fb = s.get("fallback") or ""
    if fb and fb != s["provider"] and os.environ.get(f"{fb.upper()}_API_KEY"):
        fallback = build(fb)
    return FallbackClient(primary, fallback)


def check_provider(client, label: str) -> dict:
    """One tiny call to prove the key and model work. Returns a dict for display."""
    try:
        r = client.generate("You are a connectivity test. Reply with exactly: OK", "Reply with OK.")
        return {"provider": label, "ok": True, "model": r.model, "reply": r.text[:40],
                "seconds": round(r.latency_s, 1), "thinking tokens": r.thinking_tokens, "notes": "; ".join(r.notes)}
    except LLMError as e:
        return {"provider": label, "ok": False, "model": e.model, "reply": "",
                "seconds": None, "notes": f"{e.kind}: {str(e)[:300]}"}
