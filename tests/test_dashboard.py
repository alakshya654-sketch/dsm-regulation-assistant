"""
Tests for the dashboard (step 4d) — run without the real index, model or API key.

Run from a notebook:   from test_dashboard import run_all; run_all()
Run from a terminal:   python tests/test_dashboard.py

Uses Streamlit's built-in test runner (AppTest): it runs app/dashboard.py as a script,
with the index, embedding model and LLM replaced by the small fixtures from test_retrieve.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from dsm_assistant import index, ingest, llm, retrieve  # noqa: E402
from test_retrieve import CHUNKS, RETRIEVER  # noqa: E402

APP = str(ROOT / "app" / "dashboard.py")


class FakeClient:
    """Stands in for llm.make_client(): answers with a scripted reply, or raises."""
    provider, model = "fake", "fake-1"

    def __init__(self, reply="Parity applies [S1].", error=None, tool_args=None):
        self.primary, self.fallback = self, None
        self.reply, self.error, self.calls, self.tool_args = reply, error, [], tool_args

    def generate(self, system, user, tools=None, tool_handler=None):
        self.calls.append(user)
        if self.error:
            raise self.error
        log = []
        if tools and tool_handler and self.tool_args:
            log.append({"name": "dsm_calculator", "args": self.tool_args,
                        "result": tool_handler("dsm_calculator", self.tool_args)})
        return llm.LLMResponse(self.reply, "fake", "fake-1", 0.01, tool_calls=log)


class FakeCollection:
    def count(self):
        return len(CHUNKS)


def _patch(client):
    ingest.read_jsonl = lambda path: CHUNKS
    index.open_vector_index = lambda path: FakeCollection()
    index.load_embedder = lambda *a, **k: None
    retrieve.Retriever = lambda *a, **k: RETRIEVER
    llm.make_client = lambda settings=None: client


def _app(client):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    _patch(client)
    st.cache_resource.clear()          # each test gets its own fake client
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    return at


def _all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_dashboard_starts_and_shows_status():
    at = _app(FakeClient())
    assert "DSM Regulation Assistant" in _all_markdown(at)
    side = "\n".join(m.value for m in at.sidebar.markdown)
    assert f"<b>{len(CHUNKS)}</b> passages" in side
    assert "Gemini key:" in side
    assert len(at.button) >= 4, "example questions not shown"


def test_question_shows_answer_sources_and_behind_the_scenes():
    client = FakeClient("Parity applies to WS sellers [S1].")
    at = _app(client)
    at.chat_input[0].set_value("What applies to WS sellers?").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    text = _all_markdown(at)
    assert "Parity applies to WS sellers" in text and '<span class="tag">S1</span>' in text
    assert 'class="srcs"' in text and "Sources" in text
    assert "How the question was understood" in text
    assert len(at.dataframe) == 1, "passages table missing"
    df = at.dataframe[0].value
    assert {"tag", "used in answer", "fused score", "meaning rank", "keyword rank", "source"} <= set(df.columns)
    assert df.loc[df["tag"] == "S1", "used in answer"].item() == "✔"
    assert len(client.calls) == 1


def test_example_button_asks_that_question():
    client = FakeClient("Normal Rate applies [S1].")
    at = _app(client)
    at.button(key="example-0").click().run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert len(client.calls) == 1 and "49.97 Hz" in client.calls[0]
    assert "Normal Rate applies" in _all_markdown(at)
    assert not any(b.key and b.key.startswith("example-") for b in at.button), "examples should hide after a question"


def test_invented_citation_warned():
    at = _app(FakeClient("Parity applies [S9]."))
    at.chat_input[0].set_value("What applies to WS sellers?").run()
    assert any("Check failed" in w.value for w in at.warning)


def test_model_error_shown_not_crash():
    at = _app(FakeClient(error=llm.LLMError("quota: all models out of free quota")))
    at.chat_input[0].set_value("What applies to WS sellers?").run()
    assert not at.exception
    assert any("could not answer" in e.value for e in at.error)


def test_as_of_setting_reaches_the_prompt():
    from datetime import date
    client = FakeClient()
    at = _app(client)
    at.sidebar.checkbox[0].check().run()
    at.sidebar.date_input[0].set_value(date(2025, 6, 1)).run()
    at.chat_input[0].set_value("What applies to WS sellers?").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert "01.06.2025" in client.calls[-1]


def test_chat_answer_with_calculator_shows_result_card_and_tool_step():
    args = {"entity": "general", "scheduled_mw": 500, "actual_mw": 520, "frequency_hz": 50.0, "reference_rate": 3.0}
    at = _app(FakeClient("₹15,000 receivable, computed by the DSM calculator [S1].", tool_args=args))
    at.chat_input[0].set_value("How much for 20 MW over injection at 50 Hz, RR 3?").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    text = _all_markdown(at)
    assert "Computed by the DSM calculator" in text and "₹15,000.00" in text
    assert "Tool called by the model: dsm_calculator" in text
    assert len(at.json) == 1 and at.json[0].value is not None


def test_calculator_page_computes_live():
    at = _app(FakeClient())
    at.radio(key="page").set_value("DSM calculator").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    # defaults: general seller, scheduled 100, actual 90, 50.00 Hz, RR 3 → −10 MW within L=10 → 2.5 MWh × 3000
    assert "₹7,500.00" in _all_markdown(at) and "payable" in _all_markdown(at)
    at.selectbox(key="calc-entity").set_value("ws").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert "Charge for deviation" in _all_markdown(at)
    at.selectbox(key="calc-entity").set_value("buyer").run()
    at.checkbox(key="calc-nrparts").check().run()
    assert not at.exception and any("NR = ₹" in c.value for c in at.caption)


def test_calculator_page_every_entity_renders():
    at = _app(FakeClient())
    at.radio(key="page").set_value("DSM calculator").run()
    for ent in ["general", "ror", "msw", "ws", "ess", "psp", "buyer", "infirm"]:
        at.selectbox(key="calc-entity").set_value(ent).run()
        assert not at.exception, f"{ent}: {at.exception}"
        assert "Charge for deviation" in _all_markdown(at) or at.warning or at.info, ent


FULL_BRIEF = ("## In one paragraph\nCharges for deviation are settled per block [S1].\n"
              "## Rules in force today\n- Paid within 10 days [S2].\n## How it changed\nNothing in the documents.\n"
              "## Related rules in other regulations\nNothing in the documents.\n## Why\nNothing in the documents.\n"
              "## Proposals (drafts)\nNothing in the documents.\n## Not covered\nNothing in the documents.")


def test_topic_page_warns_when_briefing_is_cut_off():
    at = _app(FakeClient(reply="## In one paragraph\nCharges for deviation are settled per block [S1]."))
    at.radio(key="page").set_value("Topic brief").run()
    at.button(key="topic-example-2").click().run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert any("Incomplete briefing" in w.value for w in at.warning) and "incomplete" in _all_markdown(at)


def test_topic_page_shows_briefing_and_grouped_passages():
    client = FakeClient(reply=FULL_BRIEF)
    at = _app(client)
    at.radio(key="page").set_value("Topic brief").run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    assert "Give a topic" in _all_markdown(at)
    at.button(key="topic-example-2").click().run()               # "payment of deviation charges"
    assert not at.exception, f"dashboard raised: {at.exception}"
    md = _all_markdown(at)
    assert "In one paragraph" in md and "payment of deviation charges" in md and "citations valid" in md
    assert "DSM Regulations" in md and len(at.dataframe) >= 1      # grouped passages panel
    assert client.calls and "TOPIC\npayment of deviation charges" in client.calls[-1]


def test_demo_mode_shows_banner_and_caps_questions():
    """CR-6: in the public demo a visitor gets per_visitor questions; then the model is not called."""
    import os
    os.environ["DEMO_MODE"] = "1"
    try:
        client = FakeClient()
        at = _app(client)
        assert any("Public demo" in i.value for i in at.info), [i.value for i in at.info]
        at.session_state["demo_used"] = 10                         # the default cap is 10
        at.chat_input[0].set_value("What does Regulation 8 say?").run()
        assert not at.exception, f"dashboard raised: {at.exception}"
        assert any("allows 10 questions per visitor" in w.value for w in at.warning)
        assert not client.calls                                      # no model call once the cap is reached
        at.session_state["demo_used"] = 0
        at.chat_input[0].set_value("What does Regulation 8 say?").run()
        assert client.calls and at.session_state["demo_used"] == 1
    finally:
        os.environ.pop("DEMO_MODE", None)


def test_no_banner_or_cap_outside_demo_mode():
    at = _app(FakeClient())
    assert not any("Public demo" in i.value for i in at.info)


def run_all(verbose: bool = True) -> bool:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__} {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
