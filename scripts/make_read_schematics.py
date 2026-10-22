#!/usr/bin/env python
"""Generate the schematic read-method diagrams for the results report.

One modular template for ALL arms — they are the same architecture family
(a stored bank + a memory channel the reader consumes); only the
MEMORY-ACCESS module and its read schedule differ:

  row 1  WRITE strip          identical in every arm (muted)
  row 2  READ row             identical anatomy: [memory chips?] query → answer
  row 3  MEMORY ACCESS frame  the swappable module (one-shot read, raw
                              context, per-token normalized / superposed
                              taps, interface chips with/without pass 1)

Visual vocabulary:
  teal squares    stored memory tokens m_{s,j} (the bank) / their reads
  blue circles    compressed segment keys k_s (Hopfield STORE)
  orange squares  learnable interface tokens M_0
  purple block    the folded linear-attention state S_r
  dashed arrows   per-layer state reads (their COUNT is the whole point)
  red X / text    the failure being removed / demonstrated

Run:  uv run python scripts/make_read_schematics.py
"""

import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

# palette
CTX, CTX_S = "#e8eaf0", "#9aa1b0"      # context / neutral boxes
MEM, MEM_S = "#7fd0c8", "#2e8f86"      # memory tokens
KEY = "#86a8e0"                        # segment keys
IFACE, IFACE_S = "#f8c893", "#c77b28"  # interface tokens M_0
STATE, STATE_S = "#cabdf0", "#6a4fa3"  # state S_r
ANS, ANS_S = "#cde8c9", "#4e8f4a"      # answer tokens
QRY, QRY_S = "#ffffff", "#707a8a"      # query tokens
INK, MUT = "#333a45", "#707a8a"
BAD, GOOD = "#c0392b", "#1e7d43"
READ_AR = "#2e8f86"                    # state-read arrows
ATT_AR = "#8a93a5"                     # attention arcs
MOD_BG, MOD_S = "#fafbfd", CTX_S       # memory-access frame


class SVG:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">',
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="16" y="24" font-size="15" font-weight="bold" fill="{INK}">{title}</text>',
            f'<text x="16" y="40" font-size="11" fill="{MUT}">{subtitle}</text>',
        ]

    def add(self, s):
        self.parts.append(s)

    def box(self, x, y, w, h, fill, stroke, label=None, fs=10.5, lc=INK, dash=None, rx=3, lw=1.2):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{lw}"{d} rx="{rx}"/>')
        if label:
            self.add(f'<text x="{x + w / 2}" y="{y + h / 2 + fs * 0.36}" font-size="{fs}" '
                     f'fill="{lc}" text-anchor="middle">{label}</text>')

    def text(self, x, y, s, fs=10.5, c=INK, anchor="start", bold=False):
        w = ' font-weight="bold"' if bold else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{c}" text-anchor="{anchor}"{w}>{s}</text>')

    def arrow(self, x1, y1, x2, y2, c=INK, dashed=False, lw=1.5):
        d = ' stroke-dasharray="5,4"' if dashed else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
                 f'stroke-width="{lw}"{d} marker-end="url(#arr{c[1:]})"/>')

    def elbow(self, pts, c=INK, dashed=False, lw=1.5, arrow=True):
        d = ' stroke-dasharray="5,4"' if dashed else ""
        p = " L ".join(f"{x} {y}" for x, y in pts)
        m = f' marker-end="url(#arr{c[1:]})"' if arrow else ""
        self.add(f'<path d="M {p}" fill="none" stroke="{c}" stroke-width="{lw}"{d}{m}/>')

    def arc(self, x1, x2, y, c=ATT_AR, label=None, drop=20, lw=1.3, lfs=9.5):
        self.add(f'<path d="M {x2} {y} C {x2 - (x2 - x1) * 0.25} {y - drop}, '
                 f'{x1 + (x2 - x1) * 0.25} {y - drop}, {x1} {y}" fill="none" '
                 f'stroke="{c}" stroke-width="{lw}" marker-end="url(#arr{c[1:]})"/>')
        if label:
            self.text((x1 + x2) / 2, y - drop - 4, label, fs=lfs, c=c, anchor="middle")

    # ---------------- template rows ---------------- #

    def write_strip(self, y):
        """Row 1 — identical in every arm (drawn muted: the fixed background)."""
        self.text(16, y - 6, "WRITE — identical in every arm:", fs=10.5, c=MUT, bold=True)
        for i in range(4):
            self.box(16 + i * 88, y, 80, 24, CTX, CTX_S, "!K:V! !K:V!", fs=9, lc=MUT)
        self.arrow(16 + 4 * 88 - 3, y + 12, 16 + 4 * 88 + 45, y + 12, c=MUT, lw=1.2)
        bx = 16 + 4 * 88 + 54
        self.text(bx, y - 6, "per-segment inner-loop write → bank", fs=10, c=MUT, bold=True)
        for g in range(4):
            for j in range(2):
                self.box(bx + g * 42 + j * 15, y + 2, 13, 13, MEM, MEM_S, "", fs=8)
            self.text(bx + g * 42 + 14, y + 28, f"m{g + 1}", fs=8.5, c=MEM_S, anchor="middle")

    def read_row(self, y, chips, n_q, with_ans=True, arc=True):
        """Row 2 — identical anatomy: [memory chips] query tokens [answer].

        chips: list of (label, fill, stroke, width). Returns chip/q/ans centres.
        """
        x = 16
        chip_xs = []
        for label, fill, stroke, w in chips:
            self.box(x, y, w, 26, fill, stroke, label, fs=9.5)
            chip_xs.append(x + w / 2)
            x += w + 5
        if chips:
            x += 6
        q_xs = []
        for i in range(n_q):
            self.box(x, y, 26, 26, QRY, CTX_S, f"q{i + 1}", fs=10)
            q_xs.append(x + 13)
            x += 31
        ans_x = None
        if with_ans:
            self.box(x, y, 26, 26, ANS, ANS_S, "V", fs=10)
            ans_x = x + 13
        if arc:
            start = chip_xs[0] if chip_xs else q_xs[0]
            self.arc(start, ans_x if with_ans else q_xs[-1], y - 8,
                     label="causal attention", drop=16)
        return chip_xs, q_xs, ans_x

    def module_frame(self, y0, h, note=None):
        """Row 3 — the swappable MEMORY-ACCESS module (header at the bottom
        so the top band stays free for read taps)."""
        self.box(16, y0, 648, h, MOD_BG, MOD_S, "", dash="7,5", lw=1.0)
        self.text(26, y0 + h - 8, "MEMORY ACCESS — the only component that changes between arms:",
                  fs=10, c=MUT, bold=True)
        if note:
            self.text(26, y0 + 15 + 0, note, fs=10, c=MUT)  # unused hook

    def badge(self, text, c=INK):
        self.text(864, 24, text, fs=10.5, c=c, anchor="end", bold=True)

    def state_block(self, x, y, scale=1.0, label="Sᵣ = Σ k vᵀ"):
        w = int(56 * scale)
        self.box(x, y, w, w, STATE, STATE_S, "", fs=10)
        for off in (0.34, 0.67):
            o = int(w * off)
            self.add(f'<line x1="{x}" y1="{y + o}" x2="{x + w}" y2="{y + o}" stroke="{STATE_S}" stroke-width="0.8"/>')
            self.add(f'<line x1="{x + o}" y1="{y}" x2="{x + o}" y2="{y + w}" stroke="{STATE_S}" stroke-width="0.8"/>')
        self.text(x + w / 2, y + w + 14, label, fs=10, c=STATE_S, anchor="middle", bold=True)

    def legend(self, y):
        self.box(16, y - 9, 12, 12, MEM, MEM_S)
        self.text(32, y + 1, "stored memory", fs=9, c=MUT)
        self.box(110, y - 9, 12, 12, IFACE, IFACE_S)
        self.text(126, y + 1, "interface M₀", fs=9, c=MUT)
        self.box(206, y - 9, 12, 12, STATE, STATE_S)
        self.text(222, y + 1, "folded state Sᵣ", fs=9, c=MUT)
        self.box(310, y - 9, 12, 12, CTX, CTX_S)
        self.text(326, y + 1, "blinded span", fs=9, c=MUT)
        self.box(410, y - 9, 12, 12, ANS, ANS_S)
        self.text(426, y + 1, "answer", fs=9, c=MUT)

    def save(self, name):
        for c in (INK, MEM_S, IFACE_S, STATE_S, ATT_AR, READ_AR, BAD, MUT, CTX_S):
            self.add(f'<defs><marker id="arr{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
                     f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                     f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker></defs>')
        self.parts.append("</svg>")
        path = os.path.join(OUT, name)
        os.makedirs(OUT, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self.parts))
        print("wrote", os.path.normpath(path))


ROW_TAP = dict(c=READ_AR, dashed=True, lw=1.6)   # per-layer read taps


# ------------------------------------------------------------------ (a)
def hopfield():
    s = SVG(880, 396,
            "(a) Hopfield one-shot — the SAME channel, read ONCE before the pass (EM 0.25)",
            "One compressed query scores every segment once; the mixture becomes a static "
            "prefix chip. Every answer position consumes the SAME o.")
    s.write_strip(66)
    y = 168
    chip_xs, q_xs, ans_x = s.read_row(y, chips=[("o", MEM, MEM_S, 34)], n_q=3)
    s.module_frame(232, 118)
    # the one read: query -> compress -> score -> mixture -> the chip
    s.arrow(q_xs[1], y + 26, 130, 282, c=INK)
    s.text(150, y + 44, "the whole query is compressed once", fs=9.5, c=MUT)
    s.box(36, 268, 170, 28, "#f5f6fa", CTX_S, "q = compress(query)", fs=10)
    s.arrow(208, 282, 240, 282, c=INK)
    s.box(242, 268, 190, 28, "#f5f6fa", CTX_S, "aₛ = softmax(β q·kₛ)", fs=10)
    s.arrow(434, 282, 466, 282, c=INK)
    s.box(468, 268, 160, 28, "#f5f6fa", CTX_S, "o = Σ aₛ vₛ (M·d)", fs=10)
    s.elbow([(548, 266), (548, 222), (33, 222), (33, y + 28)], c=INK, lw=1.6)
    s.text(560, 226, "fills the prefix chip — ONCE, before the pass", fs=9.5, c=INK)
    s.text(250, 322, "one compromise query · convex mixture: both values superposed in one M·d slot",
           fs=10, c=BAD)
    s.text(16, 372, "EM 0.24 / 0.25 / 0.26 — each value alone ≈ 0.5, joint ≈ 0.5 × 0.5.",
           fs=10.5, c=MUT)
    s.badge("reads: 1 · before the pass")
    s.legend(388)
    s.save("read_a_hopfield.svg")


# ------------------------------------------------------------------ (b)
def bank_prefix():
    s = SVG(880, 360,
            "(b) Bank-prefix control — the SAME row, but the chips ARE the raw bank (EM 0.02)",
            "No memory-access module at all: all stored tokens ride in the context and the "
            "frozen reader must retrieve by attention alone.")
    s.write_strip(66)
    y = 168
    chips = [("", MEM, MEM_S, 20)] * 8
    chip_xs, q_xs, ans_x = s.read_row(y, chips=chips, n_q=3)
    s.text(16, y + 42, "the bank m₁ … m₈ inlined as context — O(N) tokens", fs=9.5, c=MEM_S)
    s.module_frame(232, 96)
    s.text(36, 276, "— no module —", fs=11, c=MUT, bold=True)
    s.text(36, 294, "the raw tokens themselves are the interface; retrieval must emerge from the "
                    "frozen reader's attention.", fs=10, c=BAD)
    s.text(36, 308, "At training scale (4L / 128 d, from scratch) it fails — below even the one-shot control.",
           fs=10, c=BAD, bold=True)
    s.text(16, 340, "EM 0.024 / 0.024 / 0.023.", fs=10.5, c=MUT)
    s.badge("reads: 0 — raw context")
    s.legend(352)
    s.save("read_b_bank_prefix.svg")


# ------------------------------------------------------------------ (c)
def exact_softmax():
    s = SVG(880, 400,
            "(c) Exact-softmax heads — the SAME channel, re-read at every token × layer (EM 1.00)",
            "The bank stays unfolded as per-head keys/values; each read is normalized attention "
            "over N separate items — the payload is never mixed into one slot.")
    s.write_strip(66)
    y = 172
    chip_xs, q_xs, ans_x = s.read_row(y, chips=[], n_q=4)
    s.module_frame(236, 118)
    # unfolded bank + the per-token read
    for g in range(4):
        for j in range(2):
            s.box(40 + g * 38 + j * 14, 268, 12, 12, MEM, MEM_S, "", fs=8)
    s.text(40, 296, "Kᵣ, Vᵣ — bank kept UNFOLDED", fs=9.5, c=MEM_S)
    s.arrow(196, 282, 232, 282, c=INK)
    s.box(236, 268, 372, 28, "#f5f6fa", CTX_S,
          "per token t, layer ℓ, head r:  w = softmax(qₜ·Kᵣ);  μₜ = Σₙ wₙ Vₙ", fs=9.8)
    for cx in q_xs + [ans_x]:
        s.arrow(cx, 266, cx, y + 30, **ROW_TAP)
    s.text(16, 372, "Normalized per token — but the payload is separate tokens, so the reader never "
                    "demixes a slot. O(N) per read; fastest solver (all seeds 1.00 by 19–29k).",
           fs=10.5, c=GOOD)
    s.badge("reads: T·L·Hᵣ · normalized · in-pass", GOOD)
    s.legend(388)
    s.save("read_c_exact_softmax.svg")


# ------------------------------------------------------------------ (d)
def linear_all():
    s = SVG(880, 400,
            "(d) Linear-state read — the SAME channel, folded once, re-read per token × layer (EM 0.997)",
            "The bank is folded ONCE into per-head states; every position and layer taps the "
            "superposed state with its own fresh query.")
    s.write_strip(66)
    y = 172
    chip_xs, q_xs, ans_x = s.read_row(y, chips=[], n_q=4)
    s.module_frame(236, 118)
    for g in range(4):
        for j in range(2):
            s.box(40 + g * 38 + j * 14, 268, 12, 12, MEM, MEM_S, "", fs=8)
    s.text(40, 296, "the bank", fs=9.5, c=MEM_S)
    s.arrow(186, 282, 208, 282, c=INK)
    s.text(186, 262, "fold once:", fs=9.5, c=MUT, anchor="middle")
    s.state_block(214, 268, scale=0.85)
    s.arrow(268, 282, 300, 282, c=INK)
    s.box(304, 268, 344, 28, "#f5f6fa", CTX_S,
          "μₜ = qₜ,ᵣ @ Sᵣ  (unnormalized)", fs=9.8)
    for cx in q_xs + [ans_x]:
        s.arrow(cx, 266, cx, y + 30, **ROW_TAP)
    s.text(16, 372, "Unnormalized weights: a query can match BOTH segments at full strength — no convex mixture. "
                    "O(1) per read; EM 0.995–0.998 (one seed slow: 84k).",
           fs=10.5, c=GOOD)
    s.badge("reads: T·L·Hᵣ · superposed · in-pass", GOOD)
    s.legend(388)
    s.save("read_d_linear_all.svg")


# ------------------------------------------------------------------ (d')
def interface_cond():
    s = SVG(880, 436,
            "(d′) Interface tokens — the SAME taps, restricted to M chips, conditioned by pass 1 (EM 0.999)",
            "Only M = 2 learnable chips read the state (reads: M·L). Pass 1 lets the query "
            "condition those chips BEFORE the answering pass fills them per layer.")
    s.write_strip(66)
    y = 150
    chip_xs, q_xs, ans_x = s.read_row(y, chips=[("M₀", IFACE, IFACE_S, 30), ("M₀", IFACE, IFACE_S, 30)], n_q=3)
    s.module_frame(208, 176)
    # ---- pass 1: condition the chips
    s.box(32, 230, 520, 76, "#fbf9f4", IFACE_S, "", dash="6,4", lw=1.0)
    s.text(42, 248, "PASS 1 — condition the chips on the query:", fs=10.5, c=IFACE_S, bold=True)
    for i in range(3):
        s.box(52 + i * 31, 260, 26, 24, QRY if i == 0 else CTX, CTX_S, f"q{i + 1}", fs=9)
    for xx in (196, 228):
        s.box(xx, 260, 26, 24, IFACE, IFACE_S, "M₀", fs=9)
    s.arrow(142, 272, 192, 272, c=ATT_AR)
    s.text(52, 298, "q₂,q₃ blinded · M₀ attends the query", fs=8.5, c=MUT)
    s.text(268, 258, "the chips read Sᵣ in this pass too;", fs=9, c=READ_AR)
    s.text(268, 270, "their states become the row chips", fs=9, c=READ_AR)
    # ---- shared state, read by BOTH passes
    s.state_block(584, 250, scale=0.8)
    s.arrow(582, 266, 556, 266, c=READ_AR, dashed=True, lw=1.4)
    s.arrow(582, 330, 556, 330, c=READ_AR, dashed=True, lw=1.4)
    # ---- pass 2 = the read row
    s.box(32, 316, 520, 30, "#f5f6fa", CTX_S,
          "PASS 2 = the read row above:  μ = q @ Sᵣ at the chip positions (×M·L)", fs=9.8)
    # conditioned states ride up the left margin into chip 1
    s.elbow([(32, 268), (21, 268), (21, 196), (29, y + 28)], c=IFACE_S, lw=1.6)
    # per-layer taps at the chip positions
    for tx, cx in ((41, 36), (76, 71)):
        s.arrow(tx, 224, cx, y + 30, **ROW_TAP)
    s.text(96, 214, "×M·L", fs=9.5, c=READ_AR, bold=True)
    s.text(16, 414, "Query reaches memory ONLY through pass 1 — the chips are query-conditioned before they "
                    "answer. Read cost ≈ the one-shot control. Converges by ~31k on every seed.",
           fs=10.5, c=GOOD)
    s.badge("reads: M·L · 2-pass · conditioned", GOOD)
    s.save("read_dprime_interface_cond.svg")


# ------------------------------------------------------------------ (d'')
def interface_nocond():
    s = SVG(880, 452,
            "(d″) Interface tokens, query-blind — the SAME restricted taps, pass 1 removed (EM 0.012)",
            "read_state_interface_conditioning=false: the chips are the raw learnable M₀. "
            "They sit LEFT of the query and are causal — they can never see it.")
    s.write_strip(66)
    y = 168
    chip_xs, q_xs, ans_x = s.read_row(y, chips=[("M₀", IFACE, IFACE_S, 30), ("M₀", IFACE, IFACE_S, 30)], n_q=3)
    # the impossible conditioning, below the row, crossed out
    s.add(f'<path d="M {q_xs[0] - 8} {y + 28} C {q_xs[0] - 24} {y + 50}, {chip_xs[1] + 14} {y + 50}, '
          f'{chip_xs[1] + 6} {y + 28}" fill="none" stroke="{BAD}" stroke-width="1.4" '
          f'stroke-dasharray="4,3"/>')
    s.add(f'<text x="{(q_xs[0] + chip_xs[1]) / 2 + 4}" y="{y + 48}" font-size="13" fill="{BAD}" '
          f'text-anchor="middle">✕</text>')
    s.text(128, y + 46, "raw, learnable — never conditioned", fs=9.5, c=BAD)
    s.module_frame(232, 140)
    # pass-1 slot: removed
    s.box(32, 254, 600, 34, "white", BAD, "", dash="4,4", lw=1.0)
    s.text(42, 275, "PASS 1 — REMOVED (no query conditioning; the answering pass is the only pass)",
           fs=10, c=BAD)
    # state row
    s.state_block(60, 294, scale=0.8)
    s.arrow(120, 308, 152, 308, c=INK)
    s.box(156, 294, 400, 28, "#f5f6fa", CTX_S,
          "answering pass:  μ = q @ Sᵣ at the chip positions only", fs=9.8)
    # per-layer taps at the chip positions
    for tx, cx in ((41, 36), (76, 71)):
        s.arrow(tx, 248, cx, y + 30, **ROW_TAP)
    s.text(96, 240, "×M·L", fs=9.5, c=READ_AR, bold=True)
    s.text(16, 398, "The taps stay healthy (gate 0.18–0.25, ‖μ‖/‖x‖ and β like (d′)) — but the fixed M·d "
                    "channel cannot depend on the query:", fs=10.5, c=BAD)
    s.text(16, 411, "per-value EM ≈ 0.04–0.14, joint ≈ v₁·v₂ ≈ 0.01. The causality trap, demonstrated.",
           fs=10.5, c=BAD, bold=True)
    s.text(16, 428, "EM 0.003 / 0.020 / 0.013 vs 0.999 conditioned (same seeds).", fs=10.5, c=MUT)
    s.badge("reads: M·L · 1-pass · blind", BAD)
    s.save("read_dprime2_interface_nocond.svg")


if __name__ == "__main__":
    hopfield()
    bank_prefix()
    exact_softmax()
    linear_all()
    interface_cond()
    interface_nocond()
