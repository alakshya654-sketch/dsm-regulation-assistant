"""
Tests for src/dsm_assistant/llm.py and answer.py.

Run from a notebook:   from test_answer import run_all; run_all()
Run from a terminal:   python tests/test_answer.py

No network and no API key: the LLM is replaced by a scripted fake, and the provider
clients' network call is replaced by scripted errors. Reuses the small corpus from
test_retrieve.py.
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from dsm_assistant import answer, llm  # noqa: E402
from test_retrieve import CHUNKS, RETRIEVER, TODAY  # noqa: E402


class FakeLLM:
    """Returns a scripted reply. `reply` may be a string or a function(system, user) -> str.
    `tool_args`: tool calls (dicts of arguments) the fake model makes before replying, if tools are offered."""
    provider, model = "fake", "fake-1"

    def __init__(self, reply, tool_args=None):
        self.reply, self.calls, self.tool_args, self.tools_offered = reply, [], list(tool_args or []), []

    def generate(self, system, user, tools=None, tool_handler=None):
        self.calls.append((system, user))
        self.tools_offered.append([t["name"] for t in tools or []])
        log = []
        if tools and tool_handler:
            for args in self.tool_args:
                log.append({"name": tools[0]["name"], "args": args, "result": tool_handler(tools[0]["name"], args)})
        text = self.reply(system, user) if callable(self.reply) else self.reply
        return llm.LLMResponse(text, self.provider, self.model, 0.01, tool_calls=log)


def _ask(question, reply):
    fake = FakeLLM(reply)
    return answer.ask(question, RETRIEVER, fake, today=TODAY), fake


def _chunk(doc_id, reg=None):
    return next(c for c in CHUNKS if c.doc_id == doc_id and (reg is None or c.regulation_no == reg))


# --- .env handling ----------------------------------------------------------------------
def test_parse_env_tolerates_common_mistakes():
    text = ('﻿# comment\nGEMINI_API_KEY = AQ.abc123 \nGROQ_API_KEY="gsk_x"\n'
            "export OTHER='y'\n\nnot a pair\n")
    assert llm.parse_env(text) == {"GEMINI_API_KEY": "AQ.abc123", "GROQ_API_KEY": "gsk_x", "OTHER": "y"}


def test_load_env_sets_environment_and_hides_values():
    d = Path(tempfile.mkdtemp())
    (d / ".env").write_text("TEST_DSM_KEY=secret-value\n", encoding="utf-8")
    shown = llm.load_env(d)
    assert shown == {"TEST_DSM_KEY": True}, shown              # names only, never values
    assert os.environ["TEST_DSM_KEY"] == "secret-value"
    assert llm.mask("AQ.Abcdefgh") == "AQ.A… (11 chars)" and llm.mask(None) == "not set"
    del os.environ["TEST_DSM_KEY"]


def test_retry_delay_uses_server_hint():
    assert llm._retry_delay('{"retryDelay": "23s"}', 0) == 24
    assert llm._retry_delay("Please retry in 7.5s.", 0) == 8.5
    assert llm._retry_delay("quota exceeded", 0) == 2 and llm._retry_delay("quota exceeded", 3) == 16


# --- Prompt -----------------------------------------------------------------------------
def test_prompt_contains_tagged_passages_with_status():
    a, fake = _ask("What value of X applies to wind generators?", "X is 85% in 2028-29 [S1].")
    system, user = fake.calls[0]
    assert "[S1]" in user and "QUESTION\nWhat value of X" in user
    assert "SECONDARY SOURCE" in user, "secondary source not labelled for the LLM (FR-10)"
    assert "Today's date: 13.10.2026" in user
    assert answer.REFUSAL in system


def test_draft_labelled_when_drafts_allowed():
    a, fake = _ask("What did the draft propose for WS sellers?", "It proposed parity [S1].")
    user = fake.calls[0][1]
    assert "DRAFT (proposal, not law)" in user


def test_date_question_states_date():
    _, fake = _ask("WS seller charges on 01.10.2026", "Parity applies [S1].")
    assert "The question concerns 01.10.2026." in fake.calls[0][1]


def test_applies_line_told_to_use_question_date():
    """D4c-8: run 3's A3 said 'as of 14.10.2026' (today) for a question about 01.06.2025."""
    _, fake = _ask("WS seller charges on 01.06.2025", "Parity applies [S1].")
    system, user = fake.calls[0]
    assert 'the Applies line must say "as of 01.06.2025"' in user
    assert "14.10.2026" not in system and TODAY.strftime("%d.%m.%Y") not in system, \
        "the prompt's example must not use a date that looks like today"


def test_no_date_question_uses_today_in_applies_line():
    _, fake = _ask("What value of X applies to wind generators?", "X is 85% [S1].")
    assert f'the Applies line must say "as of {TODAY.strftime("%d.%m.%Y")}"' in fake.calls[0][1]


def test_as_of_setting_changes_version_but_not_today():
    from datetime import date
    fake = FakeLLM("Parity applies [S1].")
    a = answer.ask("What applies to WS sellers?", RETRIEVER, fake, today=TODAY, as_of=date(2025, 6, 1))
    user = fake.calls[0][1]
    assert a.analysis.as_of == date(2025, 6, 1)
    assert "Today's date: 13.10.2026" in user and "the user chose the date 01.06.2025" in user


def test_date_in_question_beats_as_of_setting():
    from datetime import date
    fake = FakeLLM("Parity applies [S1].")
    a = answer.ask("WS seller charges on 01.10.2026", RETRIEVER, fake, today=TODAY, as_of=date(2025, 6, 1))
    assert a.analysis.as_of == date(2026, 10, 1)


# --- Checking the reply -----------------------------------------------------------------
def test_valid_citations_accepted():
    a, _ = _ask("late payment surcharge days", "Pay within 10 days [S1]; surcharge 0.04% per day [S1, S2].")
    assert a.cited == ["S1", "S2"] and not a.invalid_citations and a.ok
    rendered = a.render()
    assert "Sources" in rendered and f"  [S1] {answer.citation(a.sources['S1'])}" in rendered, rendered


def test_invented_citation_flagged():
    a, _ = _ask("late payment surcharge days", "Pay within 10 days [S9].")
    assert a.invalid_citations == ["S9"] and not a.ok
    assert "cites passages it was not given: S9" in a.render()


def test_uncited_answer_flagged():
    a, _ = _ask("late payment surcharge days", "Pay within 10 days.")
    assert a.uncited and not a.ok and "no citations" in a.render()


def test_refusal_recognised():
    a, _ = _ask("What is the capital of France?", answer.REFUSAL)
    assert a.refused and a.ok and not a.cited


def test_no_passages_means_no_llm_call():
    a, fake = _ask("general seller deviation charges on 01.01.2019", "should not be called")
    assert a.refused and not fake.calls and a.text == answer.REFUSAL, (a.text, len(fake.calls))


def test_secondary_use_detected():
    a, _ = _ask("value of X for wind generators", lambda s, u: "According to a summary, X is 85% [S1].")
    assert a.sources["S1"].meta["doc_type"] == "secondary", a.sources["S1"].doc_id
    assert a.uses_secondary
    b, _ = _ask("late payment surcharge days", "Pay within 10 days [S1].")
    assert not b.uses_secondary


def test_render_shows_passages_on_request():
    a, _ = _ask("late payment surcharge days", "Pay within 10 days [S1].")
    assert "Passages given to the model" in a.render(show_passages=True)       # FR-9
    assert "Passages given to the model" not in a.render()


# --- Citations built from metadata ------------------------------------------------------
def test_citation_strings():
    assert answer.citation(_chunk("dsm-2024", "8")).startswith("dsm-2024, Reg. 8 (1), p. 1")
    assert "amending Reg. 8" in answer.citation(_chunk("dsm-2024-amend-3", "8"))
    assert answer.citation(_chunk("secondary-jsa")).endswith("[secondary source]")
    assert "superseded after 15.09.2024" in answer.citation(_chunk("dsm-2022"))
    assert answer.citation(_chunk("draft-dsm-2024-amend-3")).endswith("[draft]")


# --- Provider clients: retries, model fallback, provider fallback ----------------------
class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _gemini(script, models=("m1", "m2"), clock=None):
    """GeminiClient whose network call follows `script`: a list of exceptions/replies.
    Sleeping advances the fake clock, so cooldowns behave as in real time."""
    from google.genai import errors
    sleeps = []
    clock = clock or FakeClock()

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    client = llm.GeminiClient(list(models), api_key="AQ.test-key", sleep=sleep, max_retries=2, clock=clock)
    calls = []

    class Resp:
        def __init__(self, text):
            self.text, self.candidates, self.usage_metadata = text, [], None

    def fake_call(model, system, user, tools=None):
        calls.append(model)
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):
            raise errors.APIError(item, {"error": {"message": f"code {item}", "status": "X"}})
        return Resp(item)

    client._call = fake_call
    return client, calls, sleeps


def test_gemini_quota_switches_to_next_model_without_waiting():
    client, calls, sleeps = _gemini([429, "OK"])
    r = client.generate("s", "u")
    assert r.text == "OK" and r.model == "m2" and calls == ["m1", "m2"] and not sleeps, (calls, sleeps)
    assert "m1: out of quota, switching to next model" in r.notes


def test_gemini_last_model_waits_and_retries_on_quota():
    client, calls, sleeps = _gemini([429, "OK"], models=("m1",))
    r = client.generate("s", "u")
    assert r.text == "OK" and calls == ["m1", "m1"] and len(sleeps) == 1


def test_gemini_skips_model_in_cooldown_on_later_questions():
    clock = FakeClock()
    client, calls, _ = _gemini([429, "OK", "OK", "OK"], clock=clock)
    client.generate("s", "u")                    # m1 out of quota → m2
    r2 = client.generate("s", "u")               # m1 still cooling down → straight to m2
    assert calls == ["m1", "m2", "m2"], calls
    assert r2.notes and r2.notes[0].startswith("skipped (out of quota, cooling down): m1")
    clock.now += 120                             # cooldown over → m1 first again
    r3 = client.generate("s", "u")
    assert calls[-1] == "m1" and r3.model == "m1", calls


def test_gemini_server_busy_gets_one_retry_then_next_model():
    client, calls, sleeps = _gemini([503, 503, "OK"])
    r = client.generate("s", "u")
    assert calls == ["m1", "m1", "m2"] and len(sleeps) == 1 and r.model == "m2", (calls, sleeps)


def test_gemini_moves_to_next_model_on_404():
    client, calls, _ = _gemini([404, "OK"])
    r = client.generate("s", "u")
    assert r.model == "m2" and calls == ["m1", "m2"] and "m1: not_found" in r.notes[0]


def test_gemini_bad_key_fails_fast():
    client, calls, sleeps = _gemini([401, "OK"])
    try:
        client.generate("s", "u")
        assert False, "should have raised"
    except llm.LLMError as e:
        assert e.kind == "auth" and calls == ["m1"] and not sleeps


def test_gemini_quota_exhausted_on_all_models_raises():
    client, calls, _ = _gemini([429, 429, 429, 429])
    try:
        client.generate("s", "u")
        assert False, "should have raised"
    except llm.LLMError as e:
        assert e.kind == "quota" and calls == ["m1", "m2", "m2", "m2"], calls


def test_gemini_network_error_retried_then_reported():
    client, calls, sleeps = _gemini([ConnectionError("reset"), "OK"])
    assert client.generate("s", "u").text == "OK" and len(sleeps) == 1
    client, calls, _ = _gemini([ConnectionError("down")] * 3)
    try:
        client.generate("s", "u")
        assert False, "should have raised"
    except llm.LLMError as e:
        assert e.kind == "network" and calls == ["m1"] * 3


def test_gemini_config_sets_thinking_and_disables_function_calling():
    client = llm.GeminiClient(["m1"], api_key="AQ.test-key", thinking_level="low")
    cfg = client._config("m1", "system text")
    assert cfg.thinking_config.thinking_level.value == "LOW", cfg.thinking_config
    assert cfg.automatic_function_calling.disable is True and cfg.temperature == 0.0
    off = llm.GeminiClient(["m1"], api_key="AQ.test-key", thinking_level="")._config("m1", "s")
    assert off.thinking_config is None


def test_gemini_drops_thinking_setting_if_model_rejects_it():
    from google.genai import errors
    client, calls, _ = _gemini([400, "OK"])
    # make the scripted 400 look like a rejected thinking setting
    original = client._call
    def call(model, system, user, tools=None):
        try:
            return original(model, system, user, tools)
        except errors.APIError as e:
            raise errors.APIError(400, {"error": {"message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}}) from e
    client._call = call
    r = client.generate("s", "u")
    assert r.text == "OK" and "m1" in client.no_thinking_config and "thinking setting not supported" in r.notes[0]


def test_identical_citations_grouped():
    a, _ = _ask("late payment surcharge days", "Pay within 10 days [S1].")
    a.cited = ["S1", "S2", "S1"]
    a.sources = {"S1": a.sources["S1"], "S2": a.sources["S1"]}       # two tags, same chunk
    assert a.source_list() == [(["S1", "S2"], answer.citation(a.sources["S1"]))]
    assert "[S1] [S2] " in a.render()


def test_fallback_provider_used_and_noted():
    class Failing:
        provider = "gemini"
        def generate(self, s, u, **kw):
            raise llm.LLMError("quota gone", "quota", "gemini", "m1")
    r = llm.FallbackClient(Failing(), FakeLLM("from backup")).generate("s", "u")
    assert r.text == "from backup" and r.notes[0].startswith("fell back from gemini (quota)")


def test_no_fallback_without_its_key():
    saved = os.environ.pop("GROQ_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "AQ.test")
    try:
        c = llm.make_client(llm.DEFAULTS)
        assert c.fallback is None and c.primary.provider == "gemini"
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved


# --- DSM calculator tool (Phase 5) ------------------------------------------------------
CALC_ARGS = {"entity": "general", "scheduled_mw": 500, "actual_mw": 520, "frequency_hz": 50.0, "reference_rate": 3.0}


def test_calculator_tool_offered_and_result_recorded():
    fake = FakeLLM("The charge is ₹15,000 receivable, computed by the DSM calculator, under Reg. 8(1) [S1].",
                   tool_args=[CALC_ARGS])
    a = answer.ask("How much does a general seller get for 20 MW over injection at 50 Hz with RR Rs 3?",
                   RETRIEVER, fake, today=TODAY)
    assert fake.tools_offered[0] == ["dsm_calculator"]
    assert len(a.tool_calls) == 1 and a.calculations[0]["total_rs"] == 15000 and a.ok
    assert "DSM calculator" in a.render() and "Total: ₹15,000.00 receivable" in a.render()


def test_calculator_uses_question_date_when_model_omits_it():
    fake = FakeLLM("Computed by the DSM calculator [S1].", tool_args=[CALC_ARGS])
    a = answer.ask("Charge for a general seller on 01.10.2026: scheduled 500, actual 520, 50 Hz, RR 3",
                   RETRIEVER, fake, today=TODAY)
    assert a.calculations[0]["date"] == "2026-10-01"


def test_calculator_missing_input_is_reported_not_crashed():
    args = {"entity": "buyer", "scheduled_mw": 100, "actual_mw": 110, "frequency_hz": 50.0}   # no normal_rate
    fake = FakeLLM("The Normal Rate is needed to compute this [S1].", tool_args=[args])
    a = answer.ask("What does a buyer pay for 10 MW over drawal?", RETRIEVER, fake, today=TODAY)
    assert a.tool_calls[0]["result"]["field"] == "normal_rate" and not a.calculations
    assert "missing_or_invalid_input" in a.render()


def test_tools_can_be_switched_off():
    fake = FakeLLM("Parity applies [S1].")
    answer.ask("What applies to WS sellers?", RETRIEVER, fake, today=TODAY, use_tools=False)
    assert fake.tools_offered == [[]]


def test_prompt_has_calculator_and_terms_rules():
    assert "dsm_calculator" in answer.SYSTEM_PROMPT and "Reference Charge Rate" in answer.SYSTEM_PROMPT
    assert 'Never call RR the "Normal Rate"' in answer.SYSTEM_PROMPT


def test_tool_handler_error_becomes_data():
    log = []
    out = llm._run_handler(lambda n, a: 1 / 0, "dsm_calculator", {}, log)
    assert out["ok"] is False and "ZeroDivisionError" in out["error"] and log[0]["name"] == "dsm_calculator"


def test_gemini_tool_loop_calls_handler_then_answers():
    from google.genai import types
    client = llm.GeminiClient(["m1"], api_key="AQ.test-key", sleep=lambda s: None, max_retries=0)
    first = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(name="dsm_calculator", args=CALC_ARGS))]))])
    second = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(text="₹15,000 receivable, computed by the DSM calculator.")]))])
    seen = []

    def fake_call(model, system, contents, tools=None):
        seen.append((len(contents), [t["name"] for t in tools or []]))
        return [first, second][len(seen) - 1]

    client._call = fake_call
    r = client.generate("sys", "question", tools=[answer.TOOL_SPEC],
                        tool_handler=lambda n, a: {"ok": True, "total": 15000})
    assert r.text.startswith("₹15,000") and r.tool_calls[0]["args"]["reference_rate"] == 3.0
    assert seen == [(1, ["dsm_calculator"]), (3, ["dsm_calculator"])]     # question; then + call + result


def test_gemini_tool_declaration_built_from_schema():
    client = llm.GeminiClient(["m1"], api_key="AQ.test-key")
    cfg = client._config("m1", "sys", [answer.TOOL_SPEC])
    decl = cfg.tools[0].function_declarations[0]
    assert decl.name == "dsm_calculator" and "entity" in decl.parameters_json_schema["required"]
    assert cfg.automatic_function_calling.disable is True


def test_groq_tool_loop_calls_handler_then_answers():
    from types import SimpleNamespace as NS
    import json as _json
    client = llm.GroqClient("llama", api_key="gsk_test", sleep=lambda s: None, max_retries=0)
    call = NS(id="c1", function=NS(name="dsm_calculator", arguments=_json.dumps(CALC_ARGS)))
    replies = [NS(choices=[NS(message=NS(content=None, tool_calls=[call]))], usage=None),
               NS(choices=[NS(message=NS(content="₹15,000 receivable.", tool_calls=None))], usage=None)]
    sent = []

    def create(**kw):
        sent.append(kw)
        return replies[len(sent) - 1]

    client.client = NS(chat=NS(completions=NS(create=create)))
    r = client.generate("sys", "q", tools=[answer.TOOL_SPEC], tool_handler=lambda n, a: {"ok": True})
    assert r.text == "₹15,000 receivable." and r.tool_calls[0]["args"]["actual_mw"] == 520
    assert sent[1]["messages"][-1]["role"] == "tool" and sent[1]["messages"][-1]["tool_call_id"] == "c1"


# --- Phase 6 run-1 fixes (answer 5.2) ---------------------------------------------------
class SeqFake(FakeLLM):
    """Replies in turn from a list; tool_args are used on every call where tools are offered."""
    def __init__(self, replies, tool_args_per_call=None):
        super().__init__(None)
        self.replies, self.tool_args_per_call = list(replies), list(tool_args_per_call or [])

    def generate(self, system, user, tools=None, tool_handler=None):
        self.calls.append((system, user))
        n = len(self.calls) - 1
        log = []
        args_list = self.tool_args_per_call[n] if n < len(self.tool_args_per_call) else []
        if tools and tool_handler:
            for args in args_list:
                log.append({"name": tools[0]["name"], "args": args, "result": tool_handler(tools[0]["name"], args)})
        return llm.LLMResponse(self.replies[min(n, len(self.replies) - 1)], "fake", "fake-1", 0.01, tool_calls=log)


RETRY_CALC_Q = ("A coal-based general seller is scheduled at 500 MW and injects 430 MW in a block on 15.10.2026 at 49.95 Hz. "
          "Its RR is Rs 4.00/kWh. What is the deviation charge?")
RETRY_CALC_ARGS = {"entity": "general", "date": "2026-10-15", "scheduled_mw": 500, "actual_mw": 430,
             "frequency_hz": 49.95, "reference_rate": 4.0}


def test_refusal_on_calculation_question_is_retried_once():
    fake = SeqFake([answer.REFUSAL, "Applies: x. Computed by the DSM calculator: Rs 87,150.00 payable [S1]."],
                   tool_args_per_call=[[], [RETRY_CALC_ARGS]])
    a = answer.ask(RETRY_CALC_Q, RETRIEVER, fake, today=TODAY)
    assert len(fake.calls) == 2 and answer.RETRY_REMINDER in fake.calls[1][1]
    assert not a.refused and a.calculations and "retried after refusal (FX-2)" in a.notes, (a.refused, a.notes)


def test_refusal_kept_when_model_refuses_again():
    fake = SeqFake([answer.REFUSAL, answer.REFUSAL])
    a = answer.ask(RETRY_CALC_Q, RETRIEVER, fake, today=TODAY)
    assert len(fake.calls) == 2 and a.refused


def test_no_retry_for_a_clean_answer_or_weak_refusal():
    fake = SeqFake(["Applies: x. Within 10 days [S1]."])
    answer.ask("Within how many days must deviation charges be paid?", RETRIEVER, fake, today=TODAY)
    assert len(fake.calls) == 1
    # A refusal with no calculation asked and no passage found by both searches is not retried.
    fake = SeqFake([answer.REFUSAL])
    r = RETRIEVER.search("zebra quantum", k=5, today=TODAY)
    assert not answer._should_retry_refusal(llm.LLMResponse(answer.REFUSAL, "f", "f", 0.0),
                                            type(r)(r.analysis, [h for h in r.hits if h.vector_rank is None or h.keyword_rank is None]))


def test_prompt_rules_for_refusal_tool_and_abbreviations():
    p = answer.SYSTEM_PROMPT
    assert "only if NO passage is about the question" in p and "State commission" in p     # FX-1
    assert "MUST call the dsm_calculator tool before answering" in p                       # FX-1
    assert "Never expand or explain an abbreviation unless a passage defines it" in p      # FX-3


def test_calculator_source_note_added_when_model_omits_it():
    """FX-5 (FR-10): X from the law-firm summary must be named; code adds the calculator's note if the model didn't."""
    args = {"entity": "ws", "ws_type": "wind", "date": "2027-06-10", "scheduled_mw": 60, "actual_mw": 45,
            "available_capacity_mw": 100, "contract_rate": 3.0}
    q = "On 10.06.2027 a wind plant ... What is the deviation charge? Rs 3.00/kWh"
    a = answer.ask(q, RETRIEVER, SeqFake(["Applies: x. Rs 11,842.50 payable, computed by the DSM calculator [S1]."],
                                          tool_args_per_call=[[args]]), today=TODAY)
    assert "Note (from the DSM calculator):" in a.text and "law-firm summary" in a.text
    assert "calculator source note added (FX-5)" in a.notes
    b = answer.ask(q, RETRIEVER, SeqFake(["Applies: x. Rs 11,842.50 payable; X from a law-firm summary [S1]."],
                                          tool_args_per_call=[[args]]), today=TODAY)
    assert "Note (from the DSM calculator)" not in b.text                 # already said: nothing added


# ---------------------------------------------------------------------------------------
def run_all(verbose: bool = True) -> bool:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
