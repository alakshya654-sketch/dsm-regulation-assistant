"""
Tests for the DSM calculator (Phase 5) — hand-worked examples from docs/05-calculator-rules.md.

Run from a notebook:   from test_calculator import run_all; run_all()
Run from a terminal:   python tests/test_calculator.py

Each expected figure is worked by hand in the comment above it:
  energy (MWh) = MW × 0.25;  amount (₹) = MWh × 1000 × rate (₹/kWh);  + receivable, − payable.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsm_assistant import calculator as calc  # noqa: E402
from dsm_assistant.calculator import calculate  # noqa: E402


def close(a, b, tol=0.011):
    return abs(a - b) <= tol


def _pcts(r):
    return [s.rate_pct for s in r.slabs]


# --- General seller, Reg. 8(1) ------------------------------------------------------------
def test_general_over_injection_within_limit_at_50hz():
    # sched 500, actual 520: +20 MW; L = min(50, 100) = 50 → all within; f 50.00 → 100% RR (I)(i)
    # 20 × 0.25 = 5 MWh × 1000 × 3.00 = +15,000
    r = calculate("general", "15.10.2026", 500, 520, frequency_hz=50.00, reference_rate=3.00)
    assert r.direction == "over_injection" and close(r.total_rs, 15000), r.summary()
    assert r.slabs[0].clause == "Reg. 8(1)(I)(i)"


def test_general_under_injection_split_across_limit():
    # sched 500, actual 430: −70 MW; L = 50. f 49.95: 2 steps below 49.97 → 100 + 2×7.15 = 114.3% (I)(vi)
    # within: 50 MW → 12.5 MWh × 1000 × 4 × 1.143 = 57,150 payable
    # beyond: 20 MW, 49.90 ≤ f < 50.00 → 150% (III)(iii) → 5 MWh × 1000 × 6.00 = 30,000 payable → −87,150
    r = calculate("general", "15.10.2026", 500, 430, frequency_hz=49.95, reference_rate=4.00)
    assert _pcts(r) == [114.3, 150.0], _pcts(r)
    assert close(r.total_rs, -87150), r.summary()
    assert r.outcome == "payable"


def test_interpretation_I1_caps_at_stated_end_values():
    # at 49.90: 7 steps → 2.15×7 = 15.05 → capped 115%; 7.15×7 = 50.05 → capped 150%
    over = calculate("general", "15.10.2026", 500, 510, frequency_hz=49.90, reference_rate=1)
    under = calculate("general", "15.10.2026", 500, 490, frequency_hz=49.90, reference_rate=1)
    assert _pcts(over) == [115.0] and _pcts(under) == [150.0], (_pcts(over), _pcts(under))
    # one step: 49.96 → 102.15% over, 107.15% under
    over = calculate("general", "15.10.2026", 500, 510, frequency_hz=49.96, reference_rate=1)
    under = calculate("general", "15.10.2026", 500, 490, frequency_hz=49.96, reference_rate=1)
    assert _pcts(over) == [102.15] and _pcts(under) == [107.15]


def test_frequency_above_50_03_reduces_rates():
    # 50.04: over 100 − 25 = 75% (I)(ii); under 100 − 7.5 = 92.5% (I)(v); 50.05: 50% and 85%
    for f, o, u in [(50.04, 75.0, 92.5), (50.05, 50.0, 85.0), (50.03, 100.0, 100.0), (49.97, 100.0, 100.0)]:
        over = calculate("general", "15.10.2026", 500, 510, frequency_hz=f, reference_rate=1)
        under = calculate("general", "15.10.2026", 500, 490, frequency_hz=f, reference_rate=1)
        assert _pcts(over) == [o] and _pcts(under) == [u], (f, _pcts(over), _pcts(under))


def test_outside_band_rates():
    # 50.06 (outside band, < 50.10): over → zero (II)(i); under → 85% (II)(iii)
    over = calculate("general", "15.10.2026", 500, 510, frequency_hz=50.06, reference_rate=2)
    under = calculate("general", "15.10.2026", 500, 490, frequency_hz=50.06, reference_rate=2)
    assert _pcts(over) == [0.0] and over.total_rs == 0 and _pcts(under) == [85.0]
    # 49.85: over 115% (II)(ii); under 150% (II)(iv)
    over = calculate("general", "15.10.2026", 500, 510, frequency_hz=49.85, reference_rate=2)
    under = calculate("general", "15.10.2026", 500, 490, frequency_hz=49.85, reference_rate=2)
    assert _pcts(over) == [115.0] and _pcts(under) == [150.0]


def test_over_injection_at_50_10_seller_pays_10_percent():
    # +60 MW at 50.10, L = 50: within 50 → pays 10% (II)(i) proviso; beyond 10 → pays 10% (III)(i) proviso
    # 50 MW = 12.5 MWh × 1000 × 0.30 = 3,750 + 10 MW = 2.5 MWh × 300 = 750 → −4,500 (RR 3.00)
    r = calculate("general", "15.10.2026", 500, 560, frequency_hz=50.10, reference_rate=3.00)
    assert r.direction == "over_injection" and close(r.total_rs, -4500), r.summary()


def test_beyond_limit_rates_for_under_injection():
    # −80 MW, L = 50, beyond 30 MW: f 50.02 → 100% (III)(ii); f 49.85 → 200% (III)(iv)
    r = calculate("general", "15.10.2026", 500, 420, frequency_hz=50.02, reference_rate=1)
    assert _pcts(r) == [100.0, 100.0]
    r = calculate("general", "15.10.2026", 500, 420, frequency_hz=49.85, reference_rate=1)
    assert _pcts(r) == [150.0, 200.0] and r.slabs[1].clause == "Reg. 8(1)(III)(iv)"


def test_100_mw_cap_on_volume_limit():
    # sched 2000: 10% = 200 MW, capped at 100 MW. +150 MW at 50.00: within 100 → 25 MWh × 2000 = 50,000;
    # beyond 50 → zero (III)(i). Total +50,000 (RR 2.00)
    r = calculate("general", "15.10.2026", 2000, 2150, frequency_hz=50.00, reference_rate=2.00)
    assert close(r.slabs[0].mwh, 25) and close(r.slabs[1].mwh, 12.5) and close(r.total_rs, 50000), r.summary()


def test_interpretation_I2_rounds_frequency_half_up():
    # 49.955 → 49.96 (1 step: 107.15%); 49.965 → 49.97 (100%)
    a = calculate("general", "15.10.2026", 500, 490, frequency_hz=49.955, reference_rate=1)
    b = calculate("general", "15.10.2026", 500, 490, frequency_hz=49.965, reference_rate=1)
    assert _pcts(a) == [107.15] and _pcts(b) == [100.0], (_pcts(a), _pcts(b))


def test_interpretation_I5_zero_schedule():
    # scheduled 0, +5 MW at 50.00 → L = 0 → beyond → zero; note added
    r = calculate("general", "15.10.2026", 0, 5, frequency_hz=50.00, reference_rate=3)
    assert r.total_rs == 0 and r.slabs[0].label.startswith("beyond") and any("I-5" in n for n in r.notes)


def test_no_deviation():
    r = calculate("general", "15.10.2026", 500, 500, frequency_hz=50.00, reference_rate=3)
    assert r.direction == "none" and r.slabs == [] and r.total_rs == 0 and r.outcome == "nil"


# --- WS sellers, Reg. 6(2) and 8(4) ---------------------------------------------------------
def test_ws_solar_before_april_2026_uses_available_capacity():
    # avc 100, sched 80, actual 65 → −15 MW; DWS% = −15% (denominator = AvC); limits 10% / 15% → 10 / 15 MW
    # VLwS(1) 10 MW @ CR: 2.5 MWh × 2500 = 6,250; VLwS(2) 5 MW @ 110%: 1.25 × 2750 = 3,437.50 → −9,687.50
    r = calculate("ws", "15.10.2025", 80, 65, ws_type="solar", available_capacity_mw=100, contract_rate=2.50)
    assert close(r.deviation_pct, -15.0) and close(r.total_rs, -9687.50), r.summary()
    assert r.slabs[1].clause == "Reg. 8(4)(v)" and "Reg. 6(2)(a)" in r.clauses


def test_ws_solar_fy_2026_27_new_limits():
    # same case on 15.10.2026: X = 100% → denominator 100; limits 5% / 10% → 5 / 10 MW
    # 5 MW @100%: 3,125; 5 MW @110%: 3,437.50; 5 MW beyond @200%: 1.25 × 5000 = 6,250 → −12,812.50
    r = calculate("ws", "15.10.2026", 80, 65, ws_type="solar", available_capacity_mw=100, contract_rate=2.50)
    assert close(r.total_rs, -12812.50), r.summary()
    assert any("law-firm summary" in n for n in r.notes) and any("writ" in n for n in r.notes)


def test_ws_solar_fy_2028_29_x_weighted_denominator():
    # 01.06.2028, solar X = 75%: den = 0.75 × 100 + 0.25 × 60 = 90; limits 4.5 / 9 MW; actual 45 → −15 MW
    # 4.5 MW @100%: 1.125 × 2500 = 2,812.50; 4.5 MW @110%: 1.125 × 2750 = 3,093.75; 6 MW @200%: 1.5 × 5000 = 7,500
    # → −13,406.25
    r = calculate("ws", "01.06.2028", 60, 45, ws_type="solar", available_capacity_mw=100, contract_rate=2.50)
    assert close(r.denominator_mw, 90) and close(r.total_rs, -13406.25), r.summary()


def test_ws_wind_over_injection_receives_90_percent_in_second_slab():
    # wind, 15.10.2026, avc 200, sched 100, actual 130: den 200 (X=100); limits 10% / 15% → 20 / 30 MW
    # 20 MW @100%: 5 MWh × 3000 = 15,000; 10 MW @90%: 2.5 × 2700 = 6,750 → +21,750
    r = calculate("ws", "15.10.2026", 100, 130, ws_type="wind", available_capacity_mw=200, contract_rate=3.00)
    assert close(r.total_rs, 21750) and _pcts(r) == [100.0, 90.0], r.summary()


def test_ws_over_injection_beyond_second_limit_is_zero():
    # wind 15.10.2026, avc 200, sched 100, actual 150: +50; limits 20/30 → 20, 10, 20 beyond @ zero
    r = calculate("ws", "15.10.2026", 100, 150, ws_type="wind", available_capacity_mw=200, contract_rate=3.00)
    assert _pcts(r) == [100.0, 90.0, 0.0] and r.slabs[2].clause == "Reg. 8(4)(iii)"


def test_ws_x_after_fy_2029_30_must_be_given():
    try:
        calculate("ws", "01.06.2030", 60, 45, ws_type="solar", available_capacity_mw=100, contract_rate=2.5)
        raise AssertionError("expected a request for x_percent")
    except calc.CalculatorError as e:
        assert e.field == "x_percent"
    r = calculate("ws", "01.06.2030", 60, 45, ws_type="solar", available_capacity_mw=100, contract_rate=2.5,
                  x_percent=40)
    # den = 0.4 × 100 + 0.6 × 60 = 76
    assert close(r.denominator_mw, 76) and any("given by the user" in n for n in r.notes)


def test_ws_connectivity_limits_available_capacity_from_23_12_2024():
    # avc 120 limited to connectivity 100 → same as the 15.10.2025 case → −9,687.50
    r = calculate("ws", "15.10.2025", 80, 65, ws_type="solar", available_capacity_mw=120, connectivity_mw=100,
                  contract_rate=2.50)
    assert close(r.total_rs, -9687.50) and any("connectivity" in n for n in r.notes)


def test_ws_at_par_with_general_seller_reg_8_4A():
    # 01.03.2029, non-bid project with COD 15.01.2029 → general seller table, contract rate as RR (I-6)
    # sched 100, actual 90 at 50.00: L = 10 → within 10 MW @100% → 2.5 MWh × 3000 = 7,500 payable
    r = calculate("ws", "01.03.2029", 100, 90, ws_type="solar", contract_rate=3.00, frequency_hz=50.00,
                  ws_route="other", cod_date="15.01.2029")
    assert close(r.total_rs, -7500) and "Reg. 8(4A), 8(1)" in r.clauses and any("I-6" in n for n in r.notes)
    # before 31.08.2026 the rule does not exist: normal WS treatment
    r = calculate("ws", "15.07.2026", 100, 90, ws_type="solar", contract_rate=3.00, available_capacity_mw=100,
                  ws_route="bid", bid_date="15.01.2027")
    assert "Reg. 8(4), Note-1" in r.clauses


# --- Run-of-river and MSW ---------------------------------------------------------------------
def test_ror_under_injection_three_slabs():
    # sched 1000, actual 800 → −200; limits 150 / 200 → 150 @100%: 37.5 × 2000 = 75,000; 50 @105%: 12.5 × 2100
    # = 26,250 → −101,250 (RR 2.00); no frequency needed
    r = calculate("ror", "15.10.2026", 1000, 800, reference_rate=2.00)
    assert close(r.total_rs, -101250) and _pcts(r) == [100.0, 105.0], r.summary()


def test_msw_over_injection_beyond_20_percent_is_zero():
    # sched 10, actual 13 → +3; limit 2 MW: 2 MW @CR 7: 0.5 × 7000 = 3,500; 1 MW @ zero → +3,500
    r = calculate("msw", "15.10.2026", 10, 13, contract_rate=7.00)
    assert close(r.total_rs, 3500) and _pcts(r) == [100.0, 0.0]


# --- Storage ---------------------------------------------------------------------------------
def test_ess_charging_over_drawal_is_under_injection():
    # drawal schedule 50, actual 60 → over-drawal 10 → treated as under injection; L = 5
    # 5 MW within @100%: 1.25 × 4000 = 5,000; 5 MW beyond (f ≥ 50.00) @100%: 5,000 → −10,000 (RR 4)
    r = calculate("ess", "15.10.2026", 50, 60, frequency_hz=50.00, reference_rate=4.00, ess_mode="charging")
    assert r.direction == "over_drawal" and close(r.total_rs, -10000), r.summary()
    assert r.slabs[0].clause == "Reg. 8(1)(I)(iv)"


def test_psp_charging_before_april_2026_as_solar_ws():
    # 01.12.2025, avc 250, drawal sched 200, actual 220 → over-drawal 20 = under injection; limits 10% of 250 = 25
    # 20 MW @100% CR 3: 5 MWh × 3000 = 15,000 payable
    r = calculate("psp", "01.12.2025", 200, 220, ess_mode="charging", available_capacity_mw=250, contract_rate=3.00)
    assert close(r.total_rs, -15000) and r.direction == "over_drawal", r.summary()


def test_psp_section_62_from_31_08_2026_uses_energy_charge_rate():
    # charging, under-drawal 20 (sched 200, actual 180) = over injection; L = 20; f 50.00 → receive 100% of ECR 2.50
    # 5 MWh × 2500 = +12,500
    r = calculate("psp", "15.09.2026", 200, 180, ess_mode="charging", psp_section_62=True, energy_charge_rate=2.50,
                  frequency_hz=50.00)
    assert close(r.total_rs, 12500) and r.slabs[0].base == "energy charge rate", r.summary()


# --- Buyers, Reg. 8(7) ------------------------------------------------------------------------
def test_buyer_over_drawal_two_slabs():
    # sched 1000, actual 1150 → +150; limits 100 / 150. f 49.95: VLB(1) 100 + 5×5 = 125% (I)(vi) →
    # 25 MWh × 1000 × 6.25 = 156,250; VLB(2) 50 MW @150% (III)(iv) → 12.5 × 7500 = 93,750 → −250,000 (NR 5)
    r = calculate("buyer", "15.10.2026", 1000, 1150, frequency_hz=49.95, normal_rate=5.00)
    assert _pcts(r) == [125.0, 150.0] and close(r.total_rs, -250000), r.summary()


def test_buyer_under_drawal_receives_reduced_rate_above_50hz():
    # −50 MW at 50.02: VLB(1) 90 − 2×8 = 74% (I)(ii) → 12.5 × 1000 × 3.70 = +46,250 (NR 5)
    r = calculate("buyer", "15.10.2026", 1000, 950, frequency_hz=50.02, normal_rate=5.00)
    assert _pcts(r) == [74.0] and close(r.total_rs, 46250)


def test_buyer_re_rich_three_slabs_at_50_07():
    # +350 MW at 50.07: VLB(1) 200 @50% (II)(iii): 50 MWh × 2000 = 100,000; VLB(2) 100 @75% (III)(vi): 25 × 3000
    # = 75,000; VLB(3) 50 @100% (IV)(iii): 12.5 × 4000 = 50,000 → −225,000 (NR 4)
    r = calculate("buyer", "15.10.2026", 3000, 3350, frequency_hz=50.07, normal_rate=4.00, buyer_category="re_rich")
    assert _pcts(r) == [50.0, 75.0, 100.0] and close(r.total_rs, -225000), r.summary()


def test_buyer_upto_400_under_drawal_at_50_10_pays():
    # sched 300, actual 260 → −40; limit min(60, 40) = 40 → VLB(1); f 50.10 → pays 10% NR: 10 × 1000 × 0.40 = −4,000
    r = calculate("buyer", "15.10.2026", 300, 260, frequency_hz=50.10, normal_rate=4.00, buyer_category="upto_400")
    assert close(r.total_rs, -4000) and r.slabs[0].clause == "Reg. 8(7)(II)(i), proviso"


def test_buyer_band_edges():
    # VLB(1) over drawal: 50.00 → 100%; 50.05 → 75%; 49.90 → 150%; 50.06 → 50%; 49.89 → 150%
    for f, p in [(50.00, 100.0), (50.05, 75.0), (49.90, 150.0), (50.06, 50.0), (49.89, 150.0), (50.10, 0.0)]:
        r = calculate("buyer", "15.10.2026", 1000, 1010, frequency_hz=f, normal_rate=1)
        assert _pcts(r) == [p], (f, _pcts(r))
    # VLB(1) under drawal: 50.00 → 90%; 49.90 → 100%; 50.05 → 50%; 49.85 → 100%
    for f, p in [(50.00, 90.0), (49.90, 100.0), (50.05, 50.0), (49.85, 100.0)]:
        r = calculate("buyer", "15.10.2026", 1000, 990, frequency_hz=f, normal_rate=1)
        assert _pcts(r) == [p], (f, _pcts(r))


# --- Infirm power, Reg. 8(8) ------------------------------------------------------------------
def test_infirm_thermal_paid_at_nr_with_ceiling_from_july_2025():
    # 01.08.2025, 100 MW, NR 3.50 > 2.86 → 25 MWh × 2860 = +71,500
    r = calculate("infirm", "01.08.2025", 0, 100, frequency_hz=50.00, normal_rate=3.50, infirm_unit="thermal")
    assert close(r.total_rs, 71500) and r.slabs[0].clause == "Reg. 8(8)(2)", r.summary()
    # 30.06.2025 (before the Second Amendment): zero
    r = calculate("infirm", "30.06.2025", 0, 100, frequency_hz=50.00, normal_rate=3.50, infirm_unit="thermal")
    assert r.total_rs == 0
    # f > 50.05: zero
    r = calculate("infirm", "01.08.2025", 0, 100, frequency_hz=50.06, normal_rate=3.50, infirm_unit="thermal")
    assert r.total_rs == 0 and r.slabs[0].clause == "Reg. 8(8)(4)"


def test_infirm_ess_from_31_08_2026():
    # 01.09.2026, 100 MW, NR 1.80 < 2.00 → 25 × 1800 = +45,000; on 30.08.2026 → zero
    r = calculate("infirm", "01.09.2026", 0, 100, frequency_hz=50.00, normal_rate=1.80, infirm_unit="ess")
    assert close(r.total_rs, 45000) and r.slabs[0].clause == "Reg. 8(8)(2A)"
    r = calculate("infirm", "30.08.2026", 0, 100, frequency_hz=50.00, normal_rate=1.80, infirm_unit="ess")
    assert r.total_rs == 0


def test_scheduled_infirm_over_injection_above_50_05_is_zero():
    # after trial, general seller, sched 100, actual 105 at 50.04 → normal: within L=10 → 75% receive
    r = calculate("infirm", "15.10.2026", 100, 105, frequency_hz=50.04, reference_rate=2, infirm_stage="after_trial")
    assert _pcts(r) == [75.0]
    r = calculate("infirm", "15.10.2026", 100, 105, frequency_hz=50.06, reference_rate=2, infirm_stage="after_trial")
    assert r.total_rs == 0 and "8(8)(4)" in r.slabs[0].clause


# --- Scope, inputs, NR, tool ------------------------------------------------------------------
def test_out_of_scope_dates():
    for args in [("general", "15.09.2024", 500, 520), ("infirm", "01.12.2024", 0, 100)]:
        try:
            calculate(*args, frequency_hz=50, reference_rate=3, normal_rate=3)
            raise AssertionError(f"expected OutOfScope for {args}")
        except calc.OutOfScope:
            pass
    r = calculate("general", "16.09.2024", 500, 520, frequency_hz=50, reference_rate=3)
    assert r.total_rs > 0


def test_missing_input_names_the_field():
    try:
        calculate("general", "15.10.2026", 500, 520, reference_rate=3)
        raise AssertionError("expected missing frequency")
    except calc.CalculatorError as e:
        assert e.field == "frequency_hz"


def test_normal_rate_from_components():
    # DAM 450, RTM 520, AS 700 paise/kWh: C = 556.666… → NR = max(450, 520, 556.67) = 556.67 paise = ₹5.5667
    assert close(calc.normal_rate_from_components(450, 520, 700), 5.5667, 1e-6)
    # RTM highest: 600 > (450 + 600 + 300)/3 = 450
    assert close(calc.normal_rate_from_components(450, 600, 300), 6.00, 1e-9)


def test_financial_year_boundaries():
    assert calc.financial_year(date(2026, 3, 31)) == 2025 and calc.financial_year(date(2026, 4, 1)) == 2026


def test_run_tool_returns_data_not_exceptions():
    ok = calc.run_tool({"entity": "general", "scheduled_mw": 500, "actual_mw": 520, "frequency_hz": 50.0,
                        "reference_rate": 3.0}, today=date(2026, 10, 15))
    assert ok["ok"] and close(ok["result"]["total_rs"], 15000) and ok["result"]["date"] == "2026-10-15"
    missing = calc.run_tool({"entity": "buyer", "scheduled_mw": 100, "actual_mw": 110, "frequency_hz": 50.0})
    assert not missing["ok"] and missing["field"] == "normal_rate"
    scope = calc.run_tool({"entity": "general", "date": "01.01.2024", "scheduled_mw": 1, "actual_mw": 2})
    assert scope["error"] == "out_of_scope"
    junk = calc.run_tool({"entity": "general", "scheduled_mw": "abc", "actual_mw": 2})
    assert not junk["ok"]


def test_tool_schema_matches_calculate_signature():
    import inspect
    params = set(inspect.signature(calculate).parameters)
    assert set(calc.TOOL_PARAMETERS["properties"]) <= params, set(calc.TOOL_PARAMETERS["properties"]) - params


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
