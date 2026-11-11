"""
Tests for src/dsm_assistant/topic.py (CR-5 topic briefing).

Run from a notebook:   from test_topic import run_all; run_all()
Run from a terminal:   python tests/test_topic.py

Uses the small hand-made corpus of test_retrieve (hashing embedder, no model download) and a fake LLM.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

from dsm_assistant import answer, llm, topic  # noqa: E402
from test_retrieve import RETRIEVER, TODAY  # noqa: E402


class FakeLLM:
    provider, model = "fake", "fake-1"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, system, user, tools=None, tool_handler=None):
        self.calls.append((system, user, tools))
        text = self.reply(user) if callable(self.reply) else self.reply
        return llm.LLMResponse(text, self.provider, self.model, 0.01)


def _doc_ids(res):
    return [h.chunk.doc_id for h in res.hits]


def test_gather_covers_all_versions_and_types():
    """No date filter: superseded (DSM 2022), draft, SoR and secondary passages are all eligible."""
    res = topic.gather("deviation charges for general sellers and WS sellers", RETRIEVER)
    ids = set(_doc_ids(res))
    assert {"dsm-2024", "dsm-2022"} <= ids, ids
    assert ids & {"draft-dsm-2024-amend-3", "sor-dsm-2024"}, ids


def test_gather_groups_by_family_oldest_first():
    res = topic.gather("deviation charges", RETRIEVER)
    fams = [f for f, _ in res.groups]
    assert fams[0] == "DSM Regulations", fams
    dsm = dict(res.groups)["DSM Regulations"]
    starts = [h.chunk.meta["effective_from_int"] for h in dsm]
    assert starts == sorted(starts), starts
    assert res.hits == [h for _, g in res.groups for h in g]          # presentation order = groups


def test_per_document_cap():
    old = topic.MAX_PER_DOC
    try:
        topic.MAX_PER_DOC = 1
        res = topic.gather("deviation charges", RETRIEVER)
        ranked = [h.chunk.doc_id for h in res.hits if not h.expanded]
        assert len(ranked) == len(set(ranked)), ranked
    finally:
        topic.MAX_PER_DOC = old


def test_exact_phrase_document_gets_a_place():
    """A document containing the topic's exact words is included even if it ranks low."""
    res = topic.gather("weighted average prices", RETRIEVER, n=2)
    ranked = [h for h in res.hits if not h.expanded]
    assert len(ranked) <= 2, len(ranked)                                 # never more than n ranked passages
    assert "dsm-2024" in _doc_ids(res)                                  # Reg 7 contains the phrase


def test_expansion_only_amends_the_same_principal():
    res = topic.gather("Regulation 8 general seller tolerance", RETRIEVER)
    for h in res.hits:
        if h.expanded:
            assert h.chunk.meta["amends"] in {x.chunk.doc_id for x in res.hits if not x.expanded}, h.chunk.doc_id
            assert h.chunk.doc_id != "gna-2022-amend-3"


def test_brief_prompt_sections_groups_and_citations():
    fake = FakeLLM("## In one paragraph\nDeviation is settled per block [S1].\n## Rules in force today\n- 10 days [S2] [S99]")
    b = topic.brief("payment of deviation charges", RETRIEVER, fake, today=TODAY)
    system, user, tools = fake.calls[0]
    assert system == topic.BRIEF_SYSTEM and tools is None                # one call, no tools
    for h in ["## In one paragraph", "## Rules in force today", "## How it changed", "## Related rules in other regulations",
              "## Why", "## Proposals (drafts)", "## Not covered"]:
        assert h in system, h
    assert "=== DSM Regulations ===" in user and "TOPIC\npayment of deviation charges" in user
    assert b.cited == ["S1", "S2"] and b.invalid_citations == ["S99"] and not b.ok
    assert len(b.sources) == len(b.hits)


def test_brief_refusal_and_empty_topic():
    b = topic.brief("payment", RETRIEVER, FakeLLM(answer.REFUSAL), today=TODAY)
    assert b.refused and b.ok
    fake = FakeLLM("should not be called")
    e = topic.brief("   ", RETRIEVER, fake, today=TODAY)
    assert e.refused and not fake.calls


def test_brief_labels_draft_and_secondary_use():
    def reply(user):
        tags = {}
        for line in user.splitlines():
            if line.startswith("[S"):
                tag = line[1:line.index("]")]
                tags[tag] = line
        draft = next(t for t, l in tags.items() if "Draft" in l or "draft" in l)
        return f"## Proposals (drafts)\n- proposed [{draft}]"
    b = topic.brief("WS seller charges at par with general sellers", RETRIEVER, FakeLLM(reply), today=TODAY)
    assert b.uses_draft


def test_coverage_counts_found_and_cited():
    fake = FakeLLM(lambda user: "x [S1]")
    b = topic.brief("deviation charges", RETRIEVER, fake, today=TODAY)
    first = b.sources["S1"].doc_id
    cl = [{"item": "first", "labels": [{"doc_id": first}]},
          {"item": "absent", "labels": [{"doc_id": "no-such-doc"}]}]
    cov = topic.coverage(b, cl)
    assert cov["cited"] == 0.5 and cov["found"] == 0.5 and cov["rows"][0]["cited"]


# --- cr5.2 (owner's run 1 defects) -----------------------------------------------------
SEVEN = "\n".join(f"## {h}\nx [S1]" for h in topic.SECTIONS)


def test_incomplete_briefing_is_flagged():
    """DT-1: a reply cut off before all 7 headings is marked incomplete (and not ok)."""
    b = topic.brief("deviation charges", RETRIEVER, FakeLLM("## In one paragraph\nx [S1]\n## Rules in force today\n- y [S1"),
                    today=TODAY)
    assert b.missing_sections[:2] == ["How it changed", "Related rules in other regulations"] and not b.complete
    assert not b.ok and any(n.startswith("INCOMPLETE") for n in b.notes)
    full = topic.brief("deviation charges", RETRIEVER, FakeLLM(SEVEN), today=TODAY)
    assert full.complete and full.ok and not full.missing_sections


def test_output_limit_raised_for_the_call_then_restored():
    class Prov:
        max_output_tokens = 4096
    class Client:
        def __init__(self):
            self.primary, self.fallback, self.seen = Prov(), Prov(), []
        def generate(self, system, user, tools=None, tool_handler=None):
            self.seen.append((self.primary.max_output_tokens, self.fallback.max_output_tokens))
            return llm.LLMResponse(SEVEN, "fake", "fake-1", 0.0)
    c = Client()
    topic.brief("deviation charges", RETRIEVER, c, today=TODAY)
    assert c.seen == [(topic.BRIEF_MAX_OUTPUT_TOKENS, topic.BRIEF_MAX_OUTPUT_TOKENS)]
    assert (c.primary.max_output_tokens, c.fallback.max_output_tokens) == (4096, 4096)


def test_reason_or_draft_cited_as_rule_in_force_is_flagged():
    """DT-2: a Statement of Reasons or draft cited under 'Rules in force today' is reported."""
    def reply(user):
        lines = user.splitlines()
        sor = next(lines[i][1:lines[i].index("]")] for i in range(len(lines) - 1)
                   if lines[i].startswith("[S") and lines[i + 1].startswith("Status: STATEMENT OF REASONS"))
        return SEVEN.replace("## Rules in force today\nx [S1]", f"## Rules in force today\n- rule [{sor}]")
    b = topic.brief("tolerance bands for wind and solar", RETRIEVER, FakeLLM(reply), today=TODAY)
    assert b.rule_section_misuse and b.sources[b.rule_section_misuse[0]].meta["doc_type"] == "sor"


def test_prompt_rules_for_memorandum_and_off_topic():
    p = topic.BRIEF_SYSTEM
    assert "never a secondary source" in p                                  # DT-3
    assert "skip any passage that is not about the topic" in p              # DT-4
    assert "Never cite a Statement of Reasons, an Explanatory Memorandum or a draft in this section" in p   # DT-2


# ---------------------------------------------------------------------------------------
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
