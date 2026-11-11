"""
Builds small PDFs that imitate the layouts of the real CERC documents, so the ingest code
can be tested without the real files. Text is taken from the real DSM Regulations 2024
(pages 7-8, as extracted in Phase 3).

    build_all(out_dir) -> {"principal": path, "amendment": path, "sor": path}
"""
from pathlib import Path

import pymupdf

W, H = 595, 842          # A4 in points
LM, RM = 60, 535         # left / right margins
MID = (LM + RM) / 2
FOOTER = "Central Electricity Regulatory Commission (Deviation Settlement Mechanism and Related Matters) Regulations, 2024 Page {n}"


def _write(page, y, text, x0=LM, x1=RM, size=9.5):
    """Write wrapped text in a box starting at y; returns the y after the text."""
    rect = pymupdf.Rect(x0, y, x1, y + 400)
    rc = page.insert_textbox(rect, text, fontsize=size, fontname="helv")
    used = 400 - rc if rc >= 0 else 400
    return y + used + 4


def _footer(page, n):
    page.insert_text((LM, H - 30), FOOTER.format(n=n), fontsize=7, fontname="helv")


TABLE_HEADER = ("Deviation by way of over injection (Receivable by the Seller)",
                "Deviation by way of under injection (Payable by the Seller)")

# Rows of Reg 8(1): ("row label",) or ("left cell", "right cell")
REG8_ROWS = [
    ("(I) For Deviation up to [10% DGS or 100 MW, whichever is less] and f within f band",),
    ("(i) @ RR when [49.97 Hz <= f <= 50.03 Hz]", "(iv) @ RR when [49.97 Hz <= f <= 50.03 Hz]"),
    ("(ii) When [50.03 Hz < f <= 50.05 Hz], for every increase in f by 0.01 Hz, charges for "
     "deviation for such seller shall be reduced by 25% of RR so that charges for deviation "
     "become 50% of RR when f = 50.05Hz",
     "(v) When [50.03 Hz < f <= 50.05 Hz], for every increase in f by 0.01 Hz, charges for "
     "deviation for such seller shall be reduced by 7.5% of RR so that charges for deviation "
     "become 85% of RR when f = 50.05Hz"),
    ("(iii) When [49.97 Hz > f >= 49.90 Hz], for every decrease in f by 0.01 Hz, charges for "
     "deviation for such seller shall be increased by 2.15% of RR so that charges for deviation "
     "become 115% of RR when f = 49.90Hz",
     "(vi) When [49.97 Hz > f >= 49.90 Hz], for every decrease in f by 0.01 Hz, charges for "
     "deviation for such seller shall be increased by 7.15% of RR so that charges for deviation "
     "becomes 150% of RR when f = 49.90Hz"),
    ("(II) For Deviation up to [10% DGS or 100 MW, whichever is less] and f outside f band",),
    ("(i) @ zero when [50.05 Hz < f < 50.10 Hz]: Provided that such seller shall pay @ 10% of RR "
     "when [f >= 50.10 Hz] (ii) @ 115 % of RR when [f < 49.90 Hz]",
     "(iii) @ 85 % of RR when [f > 50.05 Hz] (iv) @ 150 % of RR when [f < 49.90 Hz]"),
]

ROR_ROWS = [
    ("(i) @ RR for deviation up to [15% DGS or 150 MW, whichever is less]; (ii) @ Zero for "
     "deviation beyond [15% DGS or 150 MW, whichever is less]",
     "(iii) @ RR for deviation up to [15% DGS or 150 MW, whichever is less]; (iv) @ 105% of RR "
     "for deviation beyond [15% DGS or 150 MW, whichever is less] and up to [20% DGS or 200 MW, "
     "whichever is less]; (v) @ 110% of RR for deviation beyond [20% DGS or 200 MW, whichever is less]."),
]


def _table(page, y, rows, ruled: bool):
    """Two-column table. Row labels span both columns. Draws grid lines if ruled."""
    shape = page.new_shape() if ruled else None
    top = y
    all_rows = [TABLE_HEADER] + rows
    for row in all_rows:
        if len(row) == 1:
            h = 30
            page.insert_textbox(pymupdf.Rect(LM + 3, y + 3, RM - 3, y + h), row[0], fontsize=9, fontname="helv")
        else:
            h = max(32, 12 * (max(len(c) for c in row) // 38 + 2))
            page.insert_textbox(pymupdf.Rect(LM + 3, y + 3, MID - 6, y + h), row[0], fontsize=9, fontname="helv")
            page.insert_textbox(pymupdf.Rect(MID + 6, y + 3, RM - 3, y + h), row[1], fontsize=9, fontname="helv")
            if ruled:
                shape.draw_line((MID, y), (MID, y + h))
        if ruled:
            shape.draw_line((LM, y), (RM, y))
        y += h
    if ruled:
        shape.draw_line((LM, y), (RM, y))
        shape.draw_line((LM, top), (LM, y))
        shape.draw_line((RM, top), (RM, y))
        shape.finish(color=(0, 0, 0), width=0.8)
        shape.commit()
    return y + 8


def build_principal(path: Path, ruled: bool = True) -> Path:
    doc = pymupdf.open()

    # Page 1 — title, Hindi line, Reg 1, start of Reg 2 (definitions)
    p = doc.new_page(width=W, height=H)
    p.insert_text((LM, 60), "केंद्रीय विद्युत", fontsize=10)  # Hindi line (should be dropped; may render as boxes)
    y = _write(p, 80, "CENTRAL ELECTRICITY REGULATORY COMMISSION\nNOTIFICATION\nNo. L-1/260/2021/CERC Dated: 05.08.2024")
    y = _write(p, y, "1. Short title and commencement")
    y = _write(p, y, "(1) These regulations may be called the Central Electricity Regulatory Commission "
                     "(Deviation Settlement Mechanism and Related Matters) Regulations, 2024.")
    y = _write(p, y, "(2) These regulations shall come into force on such date as may be notified by the "
                     "Commission separately.")
    y = _write(p, y, "2. Definitions and Interpretation")
    y = _write(p, y, "(1) In these regulations, unless the context otherwise requires,-")
    letters = "abcdefghijklmnopqrstuvwxyz"
    defs = [f"({l}) 'Term {l.upper()}' means the defined concept number {i} used for scheduling, "
            f"metering and settlement of deviation by a regional entity, as specified in these "
            f"regulations and the Grid Code, as amended from time to time;" for i, l in enumerate(letters)]
    page_no = 1
    for d in defs:
        if y > H - 90:
            _footer(p, page_no); page_no += 1
            p = doc.new_page(width=W, height=H); y = 60
        y = _write(p, y, d)
    y = _write(p, y, "(2) Words and expressions used and not defined in these regulations shall have the "
                     "meaning assigned to them under the Act.")
    _footer(p, page_no)

    # Reg 8 page — ruled or unruled two-column table
    page_no += 1
    p = doc.new_page(width=W, height=H)
    y = _write(p, 60, "8. Charges for Deviation")
    y = _write(p, y, "(1) Charges for Deviation, in respect of a general seller (other than an RoR "
                     "generating station and a generating station based on municipal solid waste) shall be as under:")
    y = _table(p, y, REG8_ROWS, ruled)
    y = _write(p, y, "Note: System frequency = f and f band = [49.90Hz <= f <= 50.05 Hz]")
    _footer(p, page_no)

    # Next page — Reg 8(2) with an UNRULED table (tests the fallback), then Reg 9
    page_no += 1
    p = doc.new_page(width=W, height=H)
    y = _write(p, 60, "(2) Charges for Deviation, in respect of a general seller being an RoR generating "
                      "station, shall be without any linkage to grid frequency, as under:")
    y = _table(p, y, ROR_ROWS, ruled=False)
    y = _write(p, y, "(3) Charges for Deviation, in respect of a general seller being a generating station "
                     "based on municipal solid waste, shall be at the contract rate for deviation up to 20% DGS.")
    y = _write(p, y, "9. Deviation and Ancillary Service Pool Account")
    y = _write(p, y, "(1) The Deviation and Ancillary Service Pool Account shall be maintained and operated "
                     "by the concerned Regional Load Despatch Centre.")
    _footer(p, page_no)
    doc.save(path)
    return path


def build_amendment(path: Path) -> Path:
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    text = [
        "CENTRAL ELECTRICITY REGULATORY COMMISSION\nNo. L-1/260/2021/CERC Dated: 25.08.2026\nNOTIFICATION",
        "In exercise of the powers conferred under Section 178 of the Electricity Act, 2003, the Commission hereby "
        "makes the following regulations to amend the Principal Regulations, namely:-",
        "1. Short Title and Commencement",
        "1.1 These regulations may be called the Central Electricity Regulatory Commission (Deviation Settlement "
        "Mechanism and Related Matters) (Third Amendment) Regulations, 2026.",
        "1.2 These regulations shall come into force with effect from 31.08.2026.",
        "2. Insertion of a new clause after clause (4) of Regulation (8) of the Principal Regulations:",
        "2.1 A new clause shall be inserted after clause (4) of Regulation (8) of the Principal Regulations, as under:",
        "\"(4A) Notwithstanding the provisions of clause (4) of this Regulation, the charges for Deviation, in "
        "respect of a WS Seller, shall be treated at par with General Sellers in the following cases: (a) for "
        "projects under bidding route, with tendering or bid submission dates on or after 01.01.2027; and (b) for "
        "projects, other than those covered under sub-clause (a), having date of commercial operation on or after 01.01.2029.\"",
        "3. Amendment to Regulation 10 (Schedule of Payment of Charges for deviation) of the Principal Regulations:",
        "3.1 In clause (1) of Regulation 10 of the Principal Regulations, the words \"or within such timeline as "
        "stipulated in the Detailed Procedure\" shall be inserted.",
    ]
    y = 60
    for t in text:
        y = _write(p, y, t)
    doc.save(path)
    return path


TOPICS = ["the tolerance band for wind and solar sellers", "the normal rate of charges for deviation",
          "the frequency band used for general sellers", "aggregation of schedules at pooling stations",
          "payment timelines and late payment surcharge", "treatment of infirm power during trial runs",
          "charges for municipal solid waste based stations", "the pool account and its deficit procedure",
          "ancillary services and their recovery", "deviation limits for buyers in RE-rich States"]


def build_sor(path: Path) -> Path:
    """Statement of Reasons: numbered paragraphs, a running footer, and one sentence that
    repeats on every page in the MIDDLE of the page (must NOT be removed as a footer)."""
    doc = pymupdf.open()
    t = 0
    for n in range(1, 4):
        p = doc.new_page(width=W, height=H)
        y = 60
        for k in range(1, 6):
            topic = TOPICS[t % len(TOPICS)]; t += 1
            y = _write(p, y, f"{n}.{k} Stakeholders commented on {topic} under Regulation 8 of the draft "
                             f"regulations. Having considered the comments, the Commission has decided on "
                             f"{topic} as explained below, keeping grid security in view.")
            if k == 3:
                y = _write(p, y, "Commission's analysis and decision")
        _footer(p, n)
    doc.save(path)
    return path


def build_amendment_split_numbers(path: Path) -> Path:
    """Like the real Third Amendment: every paragraph number ("2.", "2.1") sits on its own
    line, with the text on the next line (seen in the owner's run on 2026-10-13)."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    items = [
        ("1.", "Short Title and Commencement"),
        ("1.1", "These regulations may be called the Central Electricity Regulatory Commission (Deviation "
                "Settlement Mechanism and Related Matters) (Third Amendment) Regulations, 2026."),
        ("1.2", "These regulations shall come into force with effect from 31.08.2026."),
        ("2.", "Insertion of a new clause after clause (4) of Regulation (8) of the Principal Regulations:"),
        ("2.1", "A new clause shall be inserted after clause (4) of Regulation (8) of the Principal Regulations, "
                "as under: \"(4A) Notwithstanding the provisions of clause (4) of this Regulation, the charges for "
                "Deviation, in respect of a WS Seller, shall be treated at par with General Sellers.\""),
        ("3.", "Insertion of a Note after clause (5) of Regulation (8) of the Principal Regulations:"),
        ("3.1", "A Note shall be inserted after clause (5) of Regulation (8) of the Principal Regulations, as under: "
                "\"Note: The charges for deviation in respect of a standalone ESS being a pumped hydro storage "
                "plant shall be computed at the energy charge rate.\""),
        ("4.", "Amendment to Regulation 10 (Schedule of Payment of Charges for deviation) of the Principal Regulations:"),
        ("4.1", "In clause (1) of Regulation 10 of the Principal Regulations, the words \"or within such timeline\" "
                "shall be inserted."),
    ]
    y = 60
    for num, text in items:
        p.insert_text((LM, y + 10), num, fontsize=9.5, fontname="helv")   # number alone on its line
        y = _write(p, y + 14, text)
    doc.save(path)
    return path


def build_principal_grid4(path: Path) -> Path:
    """Reg 8 table as the real PDF has it: a 4-column grid (two narrow, empty columns),
    'D' with subscript 'GS' in a row label, and the Note inside the last grid row."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    y = _write(p, 60, "8. Charges for Deviation")
    y = _write(p, y, "(1) Charges for Deviation, in respect of a general seller shall be as under:")
    cols = [LM, MID - 12, MID, RM - 12, RM]              # 4 grid columns: wide, narrow, wide, narrow
    shape = p.new_shape()
    top = y

    def cell(x0, x1, y0, h, text):
        p.insert_textbox(pymupdf.Rect(x0 + 3, y0 + 3, x1 - 3, y0 + h), text, fontsize=9, fontname="helv")

    def row_label_with_subscript(y0, h, prefix, rest):
        # "(I) For Deviation up to [10% D" + subscript "GS" + " or 100 MW, ...]"
        x = LM + 4
        p.insert_text((x, y0 + 13), prefix, fontsize=9, fontname="helv")
        x += pymupdf.get_text_length(prefix, "helv", 9)
        p.insert_text((x, y0 + 15), "GS", fontsize=6, fontname="helv")
        x += pymupdf.get_text_length("GS", "helv", 6)
        p.insert_text((x, y0 + 13), rest, fontsize=9, fontname="helv")

    layout = [
        ("head", 40, TABLE_HEADER),
        ("label", 22, ("(I) For Deviation up to [10% D", " or 100 MW, whichever is less] and f within f band")),
        ("cells", 34, ("(i) @ RR when [49.97 Hz <= f <= 50.03 Hz]", "(iv) @ RR when [49.97 Hz <= f <= 50.03 Hz]")),
        ("cells", 46, ("(ii) charges for deviation become 50% of RR when f = 50.05Hz",
                       "(v) charges for deviation become 85% of RR when f = 50.05Hz")),
        ("note", 22, "Note: System frequency = f and f band = [49.90Hz <= f <= 50.05 Hz]"),
    ]
    for kind, h, content in layout:
        shape.draw_line((LM, y), (RM, y))
        if kind == "head":
            cell(cols[0], cols[1], y, h, content[0]); cell(cols[2], cols[3], y, h, content[1])
            for x in cols[1:4]:
                shape.draw_line((x, y), (x, y + h))
        elif kind == "label":
            row_label_with_subscript(y, h, *content)
        elif kind == "cells":
            cell(cols[0], cols[1], y, h, content[0]); cell(cols[2], cols[3], y, h, content[1])
            for x in cols[1:4]:
                shape.draw_line((x, y), (x, y + h))
        else:   # note: sits in the narrow 2nd grid column's row but spans visually
            cell(cols[0], cols[3], y, h, content)
            for x in cols[1:4]:
                shape.draw_line((x, y), (x, y + h))
        y += h
    shape.draw_line((LM, y), (RM, y)); shape.draw_line((LM, top), (LM, y)); shape.draw_line((RM, top), (RM, y))
    shape.finish(color=(0, 0, 0), width=0.8); shape.commit()
    y = _write(p, y + 8, "(2) Charges for Deviation, in respect of a buyer, shall be as specified.")
    doc.save(path)
    return path


def build_all(out_dir: Path) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "principal": build_principal(out_dir / "principal_ruled.pdf", ruled=True),
        "principal_unruled": build_principal(out_dir / "principal_unruled.pdf", ruled=False),
        "principal_grid4": build_principal_grid4(out_dir / "principal_grid4.pdf"),
        "amendment": build_amendment(out_dir / "amendment.pdf"),
        "amendment_split": build_amendment_split_numbers(out_dir / "amendment_split.pdf"),
        "sor": build_sor(out_dir / "sor.pdf"),
    }
