"""
DSM calculator: charges for deviation for one 15-minute time block (Phase 5; FR-7; Design §4.9).

    from dsm_assistant.calculator import calculate
    r = calculate(entity="ws", ws_type="solar", date="2026-10-15",
                  scheduled_mw=80, actual_mw=65, available_capacity_mw=100, contract_rate=2.50)
    print(r.summary())

Every rule comes from docs/05-calculator-rules.md (approved 2026-10-15), which was written from
the text of the CERC DSM Regulations, 2024, the errata and the three amendments. Clause numbers
follow the errata. Interpretations I-1 to I-7 use the owner-approved recommended readings.

Sign convention: amount > 0 means the entity RECEIVES money from the pool; amount < 0 means it PAYS.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from dataclasses import dataclass, field
from datetime import date as Date

VERSION = "5.1"

BLOCK_HOURS = 0.25                      # 15-minute time block

DSM_2024_START = Date(2024, 9, 16)      # all provisions except 8(8)
INFIRM_START = Date(2024, 12, 23)       # 8(8) in force (with First Amendment)
AMEND_1 = Date(2024, 12, 23)            # available capacity limited to connectivity
AMEND_2 = Date(2025, 7, 1)              # thermal infirm power at NR, ceiling ₹2.86/kWh
WS_NEW_FORMULA = Date(2026, 4, 1)       # 6(2)(b), new volume limits
AMEND_3 = Date(2026, 8, 31)             # 8(4A), ESS infirm power ₹2.00/kWh, PSP note

# X by financial year (April–March), from the law-firm summary of CERC's order dated 31.03.2026.
X_TABLE = {2026: {"solar": 100, "hybrid": 100, "wind": 100},
           2027: {"solar": 90, "hybrid": 90, "wind": 95},
           2028: {"solar": 75, "hybrid": 75, "wind": 85},
           2029: {"solar": 55, "hybrid": 55, "wind": 65}}

ENTITIES = {
    "general": "General seller (other than RoR and MSW)",
    "ror": "General seller: run-of-river hydro",
    "msw": "General seller: municipal solid waste",
    "ws": "WS seller (wind / solar / wind-solar hybrid)",
    "ess": "Standalone energy storage system",
    "psp": "Standalone ESS: pumped hydro storage plant",
    "buyer": "Buyer",
    "infirm": "Infirm power (before or after trial run)",
}
BUYER_CATEGORIES = {
    "other": "Buyer (other than the categories below)",
    "upto_400": "Buyer with a schedule up to 400 MW",
    "re_rich": "RE-rich State (1,000–5,000 MW wind + solar)",
    "re_super_rich": "RE Super-rich State (5,000 MW or more)",
}

NOTE_X = ("X is taken from a law-firm summary (JSA Prism, April 2026) of CERC's order dated 31.03.2026, "
          "not from the official order.")
NOTE_WRIT = "CERC's order on X is reported to be subject to writ petitions pending before the Delhi High Court."


class CalculatorError(ValueError):
    """Bad or missing input. `field` names the input that needs fixing."""

    def __init__(self, message: str, field: str = ""):
        super().__init__(message)
        self.field = field


class OutOfScope(CalculatorError):
    """The case is real but version 1 of the calculator does not cover it."""


@dataclass
class Slab:
    label: str               # e.g. "within [10% DGS or 100 MW]"
    mwh: float               # energy in this slab (always ≥ 0)
    rate_pct: float          # % of the base rate
    base: str                # "RR", "contract rate", "NR", "energy charge rate"
    rate_rs_kwh: float       # ₹/kWh actually applied
    amount_rs: float         # > 0 receivable by the entity, < 0 payable
    clause: str              # e.g. "Reg. 8(1)(I)(iii)"


@dataclass
class Result:
    entity: str
    date: Date
    direction: str           # over_injection | under_injection | over_drawal | under_drawal | none | infirm_injection
    deviation_mw: float      # block average, actual − scheduled (sign as entered)
    deviation_mwh: float
    deviation_pct: float | None
    denominator_mw: float | None
    frequency_hz: float | None
    slabs: list[Slab] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)

    @property
    def total_rs(self) -> float:
        return round(sum(s.amount_rs for s in self.slabs), 2)

    @property
    def outcome(self) -> str:
        t = self.total_rs
        return "receivable" if t > 0 else "payable" if t < 0 else "nil"

    def summary(self) -> str:
        head = f"{ENTITIES.get(self.entity, self.entity)}, {self.date.strftime('%d.%m.%Y')}: "
        if self.direction == "infirm_injection":
            head += f"infirm power injected {self.deviation_mwh:.3f} MWh"
        else:
            head += (f"deviation {self.deviation_mwh:+.3f} MWh"
                     + (f" ({self.deviation_pct:+.2f}%)" if self.deviation_pct is not None else "")
                     + f" — {self.direction.replace('_', ' ')}")
        lines = [head]
        for s in self.slabs:
            lines.append(f"  {s.label}: {s.mwh:.3f} MWh @ {s.rate_pct:g}% of {s.base} = ₹{s.rate_rs_kwh:.4f}/kWh "
                         f"→ ₹{s.amount_rs:+,.2f}  [{s.clause}]")
        lines.append(f"Total: ₹{abs(self.total_rs):,.2f} {self.outcome}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Compact form returned to the LLM by the tool."""
        return {
            "entity": ENTITIES.get(self.entity, self.entity), "date": self.date.isoformat(),
            "direction": self.direction, "deviation_mwh": round(self.deviation_mwh, 4),
            "deviation_pct": None if self.deviation_pct is None else round(self.deviation_pct, 3),
            "frequency_hz": self.frequency_hz,
            "slabs": [{"part": s.label, "mwh": round(s.mwh, 4), "rate": f"{s.rate_pct:g}% of {s.base}",
                       "rate_rs_per_kwh": round(s.rate_rs_kwh, 4), "amount_rs": round(s.amount_rs, 2),
                       "clause": s.clause} for s in self.slabs],
            "total_rs": self.total_rs, "outcome": self.outcome, "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _date(d) -> Date:
    if isinstance(d, Date):
        return d
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise CalculatorError(f"Date '{d}' not understood; use DD.MM.YYYY.", "date")


def _need(value, name: str, why: str):
    if value is None:
        raise CalculatorError(f"{name} is needed {why}.", name)
    if isinstance(value, (int, float)) and (math.isnan(value) or value < 0):
        raise CalculatorError(f"{name} must be zero or more.", name)
    return float(value)


def _hundredths(f: float) -> int:
    """Interpretation I-2: frequency rounded to 2 decimals (half up), as an integer number of 0.01 Hz.
    49.955 → 4996, 49.965 → 4997."""
    return int(Decimal(str(f)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)


def financial_year(d: Date) -> int:
    """Start year of the April–March financial year: 15.10.2026 → 2026 (FY 2026-27)."""
    return d.year if d.month >= 4 else d.year - 1


def normal_rate_from_components(dam_paise: float, rtm_paise: float, as_charge_paise: float) -> float:
    """Reg. 7(1)–(2): NR = highest of (A) DAM ACP, (B) RTM ACP, (C) ⅓A + ⅓B + ⅓AS; rounded to 2 decimals.
    Inputs in paise/kWh (as in the regulation); returns ₹/kWh."""
    c = (dam_paise + rtm_paise + as_charge_paise) / 3
    nr_paise = round(max(dam_paise, rtm_paise, c), 2)
    return nr_paise / 100


def split_slabs(dev_mw: float, limits: list[float]) -> list[float]:
    """Interpretation I-3: split |deviation| into the part within each limit and the part beyond the last.
    limits are cumulative MW thresholds, ascending. Returns len(limits)+1 parts (MW)."""
    d, parts, prev = abs(dev_mw), [], 0.0
    for lim in limits:
        lim = max(lim, prev)
        parts.append(max(0.0, min(d, lim) - prev))
        prev = lim
    parts.append(max(0.0, d - prev))
    return parts


# ---------------------------------------------------------------------------
# Rate tables (percent of base, receive/pay, clause). Frequency as hundredths of Hz.
# ---------------------------------------------------------------------------
RECEIVE, PAY = 1, -1


def general_rate(direction: str, within: bool, fh: int) -> tuple[float, int, str]:
    """Reg. 8(1). direction: 'over' (over injection) or 'under' (under injection)."""
    if direction == "over":
        if within:
            if 4997 <= fh <= 5003:
                return 100.0, RECEIVE, "Reg. 8(1)(I)(i)"
            if 5003 < fh <= 5005:
                return 100.0 - 25.0 * (fh - 5003), RECEIVE, "Reg. 8(1)(I)(ii)"
            if 4990 <= fh < 4997:
                return min(115.0, 100.0 + 2.15 * (4997 - fh)), RECEIVE, "Reg. 8(1)(I)(iii)"   # I-1 cap
            if 5005 < fh < 5010:
                return 0.0, RECEIVE, "Reg. 8(1)(II)(i)"
            if fh >= 5010:
                return 10.0, PAY, "Reg. 8(1)(II)(i), proviso"
            return 115.0, RECEIVE, "Reg. 8(1)(II)(ii)"                                      # f < 49.90
        if fh >= 5010:
            return 10.0, PAY, "Reg. 8(1)(III)(i), proviso"
        return 0.0, RECEIVE, "Reg. 8(1)(III)(i)"
    # under injection: the seller pays
    if within:
        if 4997 <= fh <= 5003:
            return 100.0, PAY, "Reg. 8(1)(I)(iv)"
        if 5003 < fh <= 5005:
            return 100.0 - 7.5 * (fh - 5003), PAY, "Reg. 8(1)(I)(v)"
        if 4990 <= fh < 4997:
            return min(150.0, 100.0 + 7.15 * (4997 - fh)), PAY, "Reg. 8(1)(I)(vi)"           # I-1 cap
        if fh > 5005:
            return 85.0, PAY, "Reg. 8(1)(II)(iii)"
        return 150.0, PAY, "Reg. 8(1)(II)(iv)"                                              # f < 49.90
    if fh >= 5000:
        return 100.0, PAY, "Reg. 8(1)(III)(ii)"
    if fh >= 4990:
        return 150.0, PAY, "Reg. 8(1)(III)(iii)"
    return 200.0, PAY, "Reg. 8(1)(III)(iv)"


def buyer_rate(direction: str, slab: int, fh: int) -> tuple[float, int, str]:
    """Reg. 8(7). direction: 'under' (under drawal, buyer receives) or 'over' (over drawal, buyer pays).
    slab: 1, 2 or 3 (VLB(1), VLB(2), VLB(3)). Frequency band (I-4): 49.90 ≤ f ≤ 50.05 Hz."""
    if slab == 1:
        in_band = 4990 <= fh <= 5005
        if direction == "under":
            if in_band:
                if fh == 5000:
                    return 90.0, RECEIVE, "Reg. 8(7)(I)(i)"
                if fh > 5000:
                    return 90.0 - 8.0 * (fh - 5000), RECEIVE, "Reg. 8(7)(I)(ii)"
                return 90.0 + 1.0 * (5000 - fh), RECEIVE, "Reg. 8(7)(I)(iii)"
            if 5005 < fh < 5010:
                return 0.0, RECEIVE, "Reg. 8(7)(II)(i)"
            if fh >= 5010:
                return 10.0, PAY, "Reg. 8(7)(II)(i), proviso"
            return 100.0, RECEIVE, "Reg. 8(7)(II)(ii)"
        if in_band:
            if fh == 5000:
                return 100.0, PAY, "Reg. 8(7)(I)(iv)"
            if fh > 5000:
                return 100.0 - 5.0 * (fh - 5000), PAY, "Reg. 8(7)(I)(v)"
            return 100.0 + 5.0 * (5000 - fh), PAY, "Reg. 8(7)(I)(vi)"
        if 5005 < fh < 5010:
            return 50.0, PAY, "Reg. 8(7)(II)(iii)"
        if fh >= 5010:
            return 0.0, PAY, "Reg. 8(7)(II)(iv)"
        return 150.0, PAY, "Reg. 8(7)(II)(v)"
    if slab == 2:
        if direction == "under":
            if fh <= 5000:
                return 80.0, RECEIVE, "Reg. 8(7)(III)(i)"
            if fh <= 5005:
                return 50.0, RECEIVE, "Reg. 8(7)(III)(ii)"
            if fh < 5010:
                return 0.0, RECEIVE, "Reg. 8(7)(III)(iii)"
            return 10.0, PAY, "Reg. 8(7)(III)(iii), proviso"
        if fh < 5000:
            return 150.0, PAY, "Reg. 8(7)(III)(iv)"
        if fh <= 5005:
            return 100.0, PAY, "Reg. 8(7)(III)(v)"
        if fh < 5010:
            return 75.0, PAY, "Reg. 8(7)(III)(vi)"
        return 0.0, PAY, "Reg. 8(7)(III)(vii)"
    if direction == "under":
        if fh < 5010:
            return 0.0, RECEIVE, "Reg. 8(7)(IV)(i)"
        return 10.0, PAY, "Reg. 8(7)(IV)(i), proviso"
    if fh < 5000:
        return 200.0, PAY, "Reg. 8(7)(IV)(ii)"
    if fh < 5010:
        return 100.0, PAY, "Reg. 8(7)(IV)(iii)"
    return 50.0, PAY, "Reg. 8(7)(IV)(iv)"


# ---------------------------------------------------------------------------
# Building results
# ---------------------------------------------------------------------------
def _slab(label, mw, pct, sign, base, base_rate, clause) -> Slab:
    mwh = mw * BLOCK_HOURS
    rate = base_rate * pct / 100
    amount = round(sign * mwh * 1000 * rate, 2)
    return Slab(label, mwh, round(pct, 4), base, rate, amount + 0.0, clause)


def _base(r: Result, d: Date, dev: float, pct, den, f) -> Result:
    r.deviation_mw, r.deviation_mwh = dev, dev * BLOCK_HOURS
    r.deviation_pct, r.denominator_mw, r.frequency_hz = pct, den, f
    return r


def _general(r: Result, scheduled: float, actual: float, f: float | None, rate: float, base: str,
             clause_head: str, swap: bool = False, infirm_zero_over_above_5005: bool = False) -> Result:
    """Reg. 8(1) for general sellers, standalone ESS and WS sellers at par.
    swap=True: ESS charging mode (over-drawal is treated as under injection)."""
    f = _need(f, "frequency_hz", "for frequency-linked charges (Reg. 8(1))")
    fh = _hundredths(f)
    dev = actual - scheduled
    inj_dev = -dev if swap else dev
    pct = None if scheduled == 0 else 100 * dev / scheduled
    _base(r, r.date, dev, pct, scheduled, f)
    if dev == 0:
        r.direction = "none"
        return r
    if swap:
        r.direction = "over_drawal" if dev > 0 else "under_drawal"
        r.notes.append("Charging mode: over-drawal is treated as under injection and under-drawal as over injection "
                       "(Reg. 8(5), first proviso).")
    else:
        r.direction = "over_injection" if dev > 0 else "under_injection"
    limit = min(0.10 * scheduled, 100.0)
    if scheduled == 0:
        r.notes.append("Scheduled is zero, so the volume limit [10% DGS or 100 MW] is 0 MW and all deviation is "
                       "charged as 'beyond' (interpretation I-5).")
    within_mw, beyond_mw = split_slabs(inj_dev, [limit])
    kind = "over" if inj_dev > 0 else "under"
    for mw, within, label in ((within_mw, True, "within [10% DGS or 100 MW]"),
                              (beyond_mw, False, "beyond [10% DGS or 100 MW]")):
        if mw <= 0:
            continue
        pct_, sign, clause = general_rate(kind, within, fh)
        if infirm_zero_over_above_5005 and kind == "over" and fh > 5005:
            pct_, sign, clause = 0.0, RECEIVE, "Reg. 8(8)(4) (over injection of scheduled infirm power at f > 50.05 Hz)"
        r.slabs.append(_slab(label, mw, pct_, sign, base, rate, clause))
    r.clauses.append(clause_head)
    return r


def _volume_table(r: Result, dev: float, limits: list[float], labels: list[str], over: list[tuple],
                  under: list[tuple], base: str, rate: float) -> None:
    """Non-frequency tables (RoR, MSW, WS). over/under: list of (pct, clause) per slab."""
    parts = split_slabs(dev, limits)
    table, sign = (over, RECEIVE) if dev > 0 else (under, PAY)
    for mw, label, (pct_, clause) in zip(parts, labels, table):
        if mw > 0:
            r.slabs.append(_slab(label, mw, pct_, sign, base, rate, clause))


def _ws_denominator(r: Result, d: Date, ws_type: str, scheduled: float, avc: float,
                    x_percent: float | None) -> tuple[float, tuple[float, float]]:
    """Reg. 6(2) denominator and the VLwS limits (in % of DWS) for the date."""
    if d < WS_NEW_FORMULA:
        limits = (10.0, 15.0) if ws_type in ("solar", "hybrid") else (15.0, 20.0)
        r.clauses.append("Reg. 6(2)(a)")
        return avc, limits
    limits = (5.0, 10.0) if ws_type in ("solar", "hybrid") else (10.0, 15.0)
    fy = financial_year(d)
    if x_percent is None:
        if fy not in X_TABLE:
            raise CalculatorError(f"X for FY {fy}-{str(fy + 1)[2:]} is not in the summary of CERC's order; "
                                  "please give x_percent.", "x_percent")
        x_percent = X_TABLE[fy][ws_type]
        r.notes += [f"X = {x_percent:g}% for {ws_type} in FY {fy}-{str(fy + 1)[2:]}. {NOTE_X}", NOTE_WRIT]
    else:
        r.notes.append(f"X = {float(x_percent):g}% as given by the user.")
    x = float(x_percent)
    r.clauses.append("Reg. 6(2)(b)")
    return x / 100 * avc + (100 - x) / 100 * scheduled, limits


def calculate(entity: str, date, scheduled_mw: float | None = None, actual_mw: float | None = None,
              frequency_hz: float | None = None, reference_rate: float | None = None,
              contract_rate: float | None = None, normal_rate: float | None = None,
              available_capacity_mw: float | None = None, connectivity_mw: float | None = None,
              ws_type: str | None = None, ws_route: str | None = None, bid_date=None, cod_date=None,
              x_percent: float | None = None, buyer_category: str | None = None,
              ess_mode: str = "discharging", psp_section_62: bool = False,
              energy_charge_rate: float | None = None, infirm_unit: str | None = None,
              infirm_stage: str | None = None, infirm_seller: str = "general") -> Result:
    """Charges for deviation for one 15-minute block. Rates in ₹/kWh, power in MW (block average)."""
    entity = (entity or "").strip().lower()
    if entity not in ENTITIES:
        raise CalculatorError(f"entity must be one of: {', '.join(ENTITIES)}.", "entity")
    d = _date(date)
    if d < DSM_2024_START:
        raise OutOfScope("Dates before 16.09.2024 fall under the DSM Regulations, 2022, which the calculator "
                         "does not cover (version 1).", "date")
    scheduled = _need(scheduled_mw, "scheduled_mw", "(MW, block average)")
    actual = _need(actual_mw, "actual_mw", "(MW, block average)")
    r = Result(entity, d, "none", 0.0, 0.0, None, None, frequency_hz)

    if entity in ("general", "ess"):
        rr = _need(reference_rate, "reference_rate", "(₹/kWh) for a general seller or ESS")
        swap = entity == "ess" and ess_mode == "charging"
        return _general(r, scheduled, actual, frequency_hz, rr, "RR",
                        "Reg. 8(5)" if entity == "ess" else "Reg. 8(1)", swap=swap)

    if entity == "ror":
        rr = _need(reference_rate, "reference_rate", "(₹/kWh) for a run-of-river station")
        dev = actual - scheduled
        _base(r, d, dev, None if scheduled == 0 else 100 * dev / scheduled, scheduled, frequency_hz)
        r.direction = "none" if dev == 0 else "over_injection" if dev > 0 else "under_injection"
        _volume_table(r, dev, [min(0.15 * scheduled, 150.0), min(0.20 * scheduled, 200.0)],
                      ["up to [15% DGS or 150 MW]", "beyond [15% DGS or 150 MW] up to [20% DGS or 200 MW]",
                       "beyond [20% DGS or 200 MW]"],
                      over=[(100.0, "Reg. 8(2)(i)"), (0.0, "Reg. 8(2)(ii)"), (0.0, "Reg. 8(2)(ii)")],
                      under=[(100.0, "Reg. 8(2)(iii)"), (105.0, "Reg. 8(2)(iv)"), (110.0, "Reg. 8(2)(v)")],
                      base="RR", rate=rr)
        r.clauses.append("Reg. 8(2)")
        return r

    if entity == "msw":
        cr = _need(contract_rate, "contract_rate", "(₹/kWh) for an MSW station")
        dev = actual - scheduled
        _base(r, d, dev, None if scheduled == 0 else 100 * dev / scheduled, scheduled, frequency_hz)
        r.direction = "none" if dev == 0 else "over_injection" if dev > 0 else "under_injection"
        _volume_table(r, dev, [0.20 * scheduled], ["up to [20% DGS]", "beyond [20% DGS]"],
                      over=[(100.0, "Reg. 8(3)(i)"), (0.0, "Reg. 8(3)(ii)")],
                      under=[(100.0, "Reg. 8(3)(iii)"), (110.0, "Reg. 8(3)(iv)")],
                      base="contract rate", rate=cr)
        r.clauses.append("Reg. 8(3)")
        return r

    if entity == "ws":
        ws_type = (ws_type or "").lower()
        if ws_type not in ("solar", "wind", "hybrid"):
            raise CalculatorError("ws_type must be solar, wind or hybrid.", "ws_type")
        cr = _need(contract_rate, "contract_rate", "(₹/kWh) for a WS seller")
        at_par = d >= AMEND_3 and (
            (ws_route == "bid" and bid_date is not None and _date(bid_date) >= Date(2027, 1, 1)) or
            (ws_route == "other" and cod_date is not None and _date(cod_date) >= Date(2029, 1, 1)))
        if at_par:
            r.notes.append("Treated at par with a general seller under Reg. 8(4A); the contract rate is used as RR "
                           "(interpretation I-6).")
            return _general(r, scheduled, actual, frequency_hz, cr, "contract rate (as RR)", "Reg. 8(4A), 8(1)")
        avc = _need(available_capacity_mw, "available_capacity_mw", "(MW) for a WS seller")
        if connectivity_mw is not None and d >= AMEND_1 and avc > connectivity_mw:
            r.notes.append(f"Available capacity limited to the connectivity granted, {connectivity_mw:g} MW "
                           "(Reg. 3(1)(g) as amended from 23.12.2024).")
            avc = float(connectivity_mw)
        return _ws(r, d, ws_type, scheduled, actual, avc, x_percent, cr, "contract rate")

    if entity == "psp":
        if ess_mode == "charging" and d < WS_NEW_FORMULA:
            rate = contract_rate if contract_rate is not None else reference_rate
            rate = _need(rate, "contract_rate", "(₹/kWh): until 31.03.2026 PSP charging is treated as a solar WS seller")
            avc = _need(available_capacity_mw, "available_capacity_mw", "(MW): PSP charging uses the solar WS formula")
            r.notes.append("Pumped hydro charging until 31.03.2026: charges and deviation formula as for a solar WS "
                           "seller (Reg. 8(5), second proviso). Drawal is entered as MW; over-drawal is treated as "
                           "under injection.")
            # charging: over-drawal = under injection, so flip the sign of the deviation
            res = _ws(r, d, "solar", scheduled, actual, avc, None, rate, "contract rate", flip=True)
            res.clauses.append("Reg. 8(5), second proviso")
            return res
        if psp_section_62 and d >= AMEND_3:
            ecr = _need(energy_charge_rate, "energy_charge_rate",
                        "(₹/kWh): from 31.08.2026 a Section 62 PSP is charged at its energy charge rate")
            r.notes.append("Section 62 pumped hydro from 31.08.2026: computed at the energy charge rate (Note after "
                           "Reg. 8(5)), applied to the Reg. 8(1) table (interpretation I-8).")
            return _general(r, scheduled, actual, frequency_hz, ecr, "energy charge rate", "Reg. 8(5), Note",
                            swap=ess_mode == "charging")
        rr = _need(reference_rate, "reference_rate", "(₹/kWh) for a pumped hydro ESS")
        return _general(r, scheduled, actual, frequency_hz, rr, "RR", "Reg. 8(5)", swap=ess_mode == "charging")

    if entity == "buyer":
        return _buyer(r, scheduled, actual, frequency_hz, normal_rate, buyer_category)

    return _infirm(r, d, scheduled, actual, frequency_hz, normal_rate, reference_rate, contract_rate,
                   available_capacity_mw, ws_type, x_percent, infirm_unit, infirm_stage, infirm_seller)


def _ws(r: Result, d: Date, ws_type: str, scheduled: float, actual: float, avc: float, x_percent,
        rate: float, base: str, flip: bool = False) -> Result:
    den, (p1, p2) = _ws_denominator(r, d, ws_type, scheduled, avc, x_percent)
    if den <= 0:
        raise CalculatorError("The deviation denominator is zero; check available capacity and schedule.",
                              "available_capacity_mw")
    dev = actual - scheduled
    inj_dev = -dev if flip else dev
    _base(r, d, dev, 100 * dev / den, den, r.frequency_hz)
    if dev == 0:
        r.direction = "none"
        return r
    if flip:
        r.direction = "over_drawal" if dev > 0 else "under_drawal"
    else:
        r.direction = "over_injection" if dev > 0 else "under_injection"
    _volume_table(r, inj_dev, [p1 / 100 * den, p2 / 100 * den],
                  [f"VLwS(1): up to {p1:g}% DWS", f"VLwS(2): {p1:g}–{p2:g}% DWS", f"beyond {p2:g}% DWS"],
                  over=[(100.0, "Reg. 8(4)(i)"), (90.0, "Reg. 8(4)(ii)"), (0.0, "Reg. 8(4)(iii)")],
                  under=[(100.0, "Reg. 8(4)(iv)"), (110.0, "Reg. 8(4)(v)"), (200.0, "Reg. 8(4)(vi)")],
                  base=base, rate=rate)
    r.clauses.append("Reg. 8(4), Note-1")
    return r


def _buyer(r: Result, scheduled: float, actual: float, f, nr, category) -> Result:
    category = category or "other"
    if category not in BUYER_CATEGORIES:
        raise CalculatorError(f"buyer_category must be one of: {', '.join(BUYER_CATEGORIES)}.", "buyer_category")
    nr = _need(nr, "normal_rate", "(₹/kWh) for a buyer (or give its three components)")
    f = _need(f, "frequency_hz", "for a buyer (Reg. 8(7))")
    fh = _hundredths(f)
    dev = actual - scheduled
    _base(r, r.date, dev, None if scheduled == 0 else 100 * dev / scheduled, scheduled, f)
    if dev == 0:
        r.direction = "none"
        return r
    r.direction = "over_drawal" if dev > 0 else "under_drawal"
    if category == "other":
        limits = [min(0.10 * scheduled, 100.0), min(0.15 * scheduled, 200.0)]
        labels = ["VLB(1): up to [10% DBUY or 100 MW]", "VLB(2): to [15% DBUY or 200 MW]", "VLB(3): beyond"]
    elif category == "upto_400":
        if scheduled > 400:
            raise CalculatorError("A buyer with a schedule above 400 MW is not in the 'up to 400 MW' category.",
                                  "buyer_category")
        limits = [min(0.20 * scheduled, 40.0)]
        labels = ["VLB(1): up to [20% DBUY or 40 MW]", "VLB(2): beyond"]
    elif category == "re_rich":
        limits, labels = [200.0, 300.0], ["VLB(1): up to 200 MW", "VLB(2): 200–300 MW", "VLB(3): beyond 300 MW"]
    else:
        limits, labels = [250.0, 350.0], ["VLB(1): up to 250 MW", "VLB(2): 250–350 MW", "VLB(3): beyond 350 MW"]
    if scheduled == 0 and category in ("other", "upto_400"):
        r.notes.append("Scheduled drawal is zero, so the percentage limits are 0 MW (interpretation I-5).")
    kind = "over" if dev > 0 else "under"
    for i, (mw, label) in enumerate(zip(split_slabs(dev, limits), labels), 1):
        if mw > 0:
            pct_, sign, clause = buyer_rate(kind, i, fh)
            r.slabs.append(_slab(label, mw, pct_, sign, "NR", nr, clause))
    r.clauses += ["Reg. 8(7)", f"Volume limits: {BUYER_CATEGORIES[category]}"]
    return r


def _infirm(r: Result, d: Date, scheduled, actual, f, nr, rr, cr, avc, ws_type, x_percent,
            unit, stage, seller) -> Result:
    if d < INFIRM_START:
        raise OutOfScope("Infirm power before 23.12.2024 was governed by CERC's order of 06.02.2023 "
                         "(Petition 01/SM/2023), which is not in the corpus.", "date")
    stage = stage or "before_trial"
    if stage == "after_trial":
        r.notes.append("Scheduled infirm power after a successful trial run: charged as a general or WS seller; "
                       "over injection at f > 50.05 Hz is zero (Reg. 8(8)).")
        if seller == "ws":
            if f is not None and _hundredths(f) > 5005 and actual > scheduled:
                _base(r, d, actual - scheduled, None, None, f)
                r.direction = "over_injection"
                r.slabs.append(_slab("over injection of scheduled infirm power", actual - scheduled, 0.0,
                                     RECEIVE, "contract rate", 0.0, "Reg. 8(8)(4)"))
                return r
            cr = _need(cr, "contract_rate", "(₹/kWh) for a WS seller")
            avc = _need(avc, "available_capacity_mw", "(MW) for a WS seller")
            return _ws(r, d, (ws_type or "solar").lower(), scheduled, actual, avc, x_percent, cr, "contract rate")
        rr = _need(rr, "reference_rate", "(₹/kWh) for a general seller")
        return _general(r, scheduled, actual, f, rr, "RR", "Reg. 8(8)(3)", infirm_zero_over_above_5005=True)

    # before the trial run is complete: injection of infirm power
    unit = (unit or "other").lower()
    r.direction = "infirm_injection"
    _base(r, d, actual, None, None, f)
    if f is not None and _hundredths(f) > 5005:
        r.slabs.append(_slab("infirm power injected (f > 50.05 Hz)", actual, 0.0, RECEIVE, "NR", 0.0,
                             "Reg. 8(8)(4)" if d >= AMEND_2 else "Reg. 8(8)"))
        return r
    if unit == "thermal" and d >= AMEND_2:
        ceiling, clause = 2.86, "Reg. 8(8)(2)"
    elif unit == "ess" and d >= AMEND_3:
        ceiling, clause = 2.00, "Reg. 8(8)(2A)"
    else:
        r.slabs.append(_slab("infirm power injected", actual, 0.0, RECEIVE, "NR", 0.0,
                             "Reg. 8(8)(1)" if d >= AMEND_2 else "Reg. 8(8)"))
        r.notes.append("Injection of infirm power before the trial run is complete is charged at zero for this unit "
                       "type and date.")
        return r
    nr = _need(nr, "normal_rate", "(₹/kWh) for infirm power paid at the Normal Rate")
    paid = min(nr, ceiling)
    if nr > ceiling:
        r.notes.append(f"Normal Rate ₹{nr:.2f}/kWh is above the ceiling, so ₹{ceiling:.2f}/kWh is paid.")
    r.slabs.append(Slab("infirm power injected", actual * BLOCK_HOURS, round(100 * paid / nr, 4) if nr else 0.0,
                        f"NR (ceiling ₹{ceiling:.2f}/kWh)", paid, round(actual * BLOCK_HOURS * 1000 * paid, 2),
                        clause))
    return r


# ---------------------------------------------------------------------------
# Tool interface for the LLM (function calling)
# ---------------------------------------------------------------------------
TOOL_NAME = "dsm_calculator"
TOOL_DESCRIPTION = (
    "Computes CERC DSM 2024 charges for deviation for ONE 15-minute time block, from 16.09.2024 onwards. "
    "Use it whenever the user asks for a rupee amount. Pass only values stated in the question or conversation; "
    "never invent rates, frequency or capacity. If a needed value is missing, the tool says which one.")

TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "entity": {"type": "string", "enum": list(ENTITIES),
                   "description": "general | ror (run-of-river) | msw (municipal solid waste) | ws (wind/solar/hybrid)"
                                  " | ess (standalone storage) | psp (pumped hydro storage) | buyer | infirm"},
        "date": {"type": "string", "description": "Date of the time block, DD.MM.YYYY. Use today if not given."},
        "scheduled_mw": {"type": "number", "description": "Scheduled injection (or drawal) in MW for the block"},
        "actual_mw": {"type": "number", "description": "Actual injection (or drawal) in MW for the block"},
        "frequency_hz": {"type": "number", "description": "Average grid frequency in the block (Hz)"},
        "reference_rate": {"type": "number", "description": "Reference charge rate RR in Rs/kWh (general seller, ESS)"},
        "contract_rate": {"type": "number", "description": "Contract rate in Rs/kWh (WS and MSW sellers)"},
        "normal_rate": {"type": "number", "description": "Normal Rate of charges for deviation NR in Rs/kWh (buyers, infirm power)"},
        "available_capacity_mw": {"type": "number", "description": "Available capacity in MW (WS sellers)"},
        "connectivity_mw": {"type": "number", "description": "Connectivity granted in MW (WS sellers), if stated"},
        "ws_type": {"type": "string", "enum": ["solar", "wind", "hybrid"]},
        "ws_route": {"type": "string", "enum": ["bid", "other"], "description": "WS project under bidding route or not"},
        "bid_date": {"type": "string", "description": "Tender/bid submission date, DD.MM.YYYY (WS, bid route)"},
        "cod_date": {"type": "string", "description": "Commercial operation date, DD.MM.YYYY (WS, other route)"},
        "x_percent": {"type": "number", "description": "Value of X in % only if the user states it"},
        "buyer_category": {"type": "string", "enum": list(BUYER_CATEGORIES)},
        "ess_mode": {"type": "string", "enum": ["discharging", "charging"]},
        "psp_section_62": {"type": "boolean", "description": "Pumped hydro whose tariff is under Section 62"},
        "energy_charge_rate": {"type": "number", "description": "Energy charge rate in Rs/kWh (Section 62 pumped hydro)"},
        "infirm_unit": {"type": "string", "enum": ["thermal", "ess", "other"]},
        "infirm_stage": {"type": "string", "enum": ["before_trial", "after_trial"]},
        "infirm_seller": {"type": "string", "enum": ["general", "ws"]},
    },
    "required": ["entity", "scheduled_mw", "actual_mw"],
}


def run_tool(args: dict, today: Date | None = None) -> dict:
    """Called when the LLM requests the tool. Never raises: errors come back as data the LLM can explain."""
    args = {k: v for k, v in dict(args or {}).items() if v is not None and k in TOOL_PARAMETERS["properties"]}
    args.setdefault("date", (today or Date.today()).isoformat())
    try:
        r = calculate(**args)
        return {"ok": True, "result": r.as_dict(), "summary": r.summary()}
    except OutOfScope as e:
        return {"ok": False, "error": "out_of_scope", "message": str(e), "field": e.field}
    except CalculatorError as e:
        return {"ok": False, "error": "missing_or_invalid_input", "message": str(e), "field": e.field}
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": "invalid_input", "message": str(e)}
