"""Simple teaching pictures for the defence deck.

These are drawn, not photographed, and every quantity in them is read from the
dissertation rather than chosen for the picture. They exist because a
non-technical audience understands two red squares in a hundred faster than it
understands the number 0.01961.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import numpy as np
import os

GREEN, RED, GREY = "#1B4332", "#E4002B", "#565E6C"
CARD, PALE, AMBER, MID, MUTED = "#F5F7FA", "#E7ECE9", "#F2A900", "#2A5F45", "#9DB4A6"
FAINT = "#DDE3E8"
OUT = "dissertation"
plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def pic_rarity():
    """Two squares in a hundred. The panel positive rate is 1.961 per cent."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    hits = {44, 71}  # two of a hundred, placed apart so neither reads as a cluster
    for i in range(100):
        r, c = divmod(i, 10)
        on = i in hits
        ax.add_patch(FancyBboxPatch((c, 9 - r), 0.86, 0.86,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor=RED if on else FAINT,
                     edgecolor="white", linewidth=1.6))
    ax.set_xlim(-0.3, 10.2); ax.set_ylim(-0.3, 10.2)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_rarity.png")


def pic_coverage():
    """774 areas, 20 of them coverable in a week. 43 columns by 18 rows is 774."""
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    cols, rows = 43, 18
    rng = np.random.default_rng(7)
    picked = set(rng.choice(cols * rows, 20, replace=False).tolist())
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        on = i in picked
        ax.add_patch(Circle((c, rows - 1 - r), 0.40 if on else 0.30,
                    facecolor=RED if on else FAINT, edgecolor="none"))
    ax.set_xlim(-1, cols); ax.set_ylim(-1, rows)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_coverage.png")


def pic_reveal():
    """The brain tester answer. Recall at 20 is 0.196; a random list of 20 of
    774 areas reaches 20/774, which is 2.6 per cent. Lift 7.6 (Chapter Five)."""
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    rng = np.random.default_rng(3)
    for row, (label, n, colour) in enumerate(
            [("Picking 20 areas at random", 3, MUTED),
             ("The system's 20 areas", 20, RED)]):
        y = 1 - row * 1.55
        hits = set(rng.choice(100, n, replace=False).tolist())
        for i in range(100):
            rr, cc = divmod(i, 25)
            on = i in hits
            ax.add_patch(Circle((cc * 0.46, y - rr * 0.30), 0.17 if on else 0.12,
                        facecolor=colour if on else FAINT, edgecolor="none"))
        ax.text(-0.75, y - 0.45, label, ha="right", va="center",
                fontsize=12, color=GREEN, weight="bold")
        ax.text(11.9, y - 0.45, f"{n} in 100", ha="left", va="center",
                fontsize=17, color=colour, weight="bold")
    ax.set_xlim(-5.2, 14.2); ax.set_ylim(-1.9, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_reveal.png")


def pic_gap():
    """Where the reviewed work sits, and where this work sits. Rows from the
    related-works table: spatial unit against forecast horizon."""
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    studies = [
        ("Hou (2022)", 1, 2, 11, 0), ("Khalfa (2025)", 2, 1, 11, 0), ("Han (2023)", 2, 3, 11, 0),
        ("Rummens (2020)", 3, 3, 11, 0), ("Hegre (2019)", 3, 5.4, 11, 0),
        ("Brandt (2022)", 3, 3.85, -5, 54), ("D'Orazio (2022)", 3, 4.7, 11, 0),
        ("Mueller (2024)", 3.45, 4.2, 11, 0),
        ("Goodman (2024)", 4, 5, 11, 0), ("James & Ann (2025)", 5, 1.5, 11, 0),
    ]
    for name, x, y, dy, dx in studies:
        ax.scatter([x], [y], s=150, facecolor="white", edgecolor=GREY,
                   linewidth=1.6, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=8.2, color=GREY)
    ax.add_patch(Rectangle((1.55, 3.45), 0.9, 1.1, facecolor=RED, alpha=0.10,
                 edgecolor=RED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1))
    ax.scatter([2], [4], s=480, marker="*", facecolor=RED, edgecolor="white",
               linewidth=1.6, zorder=4)
    ax.annotate("This work", (2, 4), textcoords="offset points", xytext=(0, -24),
                ha="center", fontsize=11, color=RED, weight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Daily", "Weekly", "Monthly", "Yearly", "No\nforecast"], fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["One street", "One district", "One city", "One LGA",
                        "Region", "Whole country"], fontsize=10)
    ax.set_xlabel("How often the forecast is made", fontsize=11, color=GREEN, labelpad=10)
    ax.set_ylabel("How big an area it covers", fontsize=11, color=GREEN, labelpad=10)
    ax.set_xlim(0.4, 5.6); ax.set_ylim(0.2, 6.5)
    ax.grid(True, color=FAINT, linewidth=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(FAINT)
    ax.tick_params(colors=GREY, length=0)
    save(fig, "pic_gap.png")


def pic_judges():
    """The stack in plain words. Four base learners and one meta-learner."""
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    names = [("The scoring sheet", "adds up the clues"),
             ("The voting trees", "many yes-no questions"),
             ("The mistake fixer", "fixes the last try"),
             ("The neighbour watcher", "looks next door")]
    xs = [0.8, 3.5, 6.2, 8.9]
    W = 2.5
    for (t, sub), x in zip(names, xs):
        ax.add_patch(FancyBboxPatch((x, 3.05), W, 1.30,
                     boxstyle="round,pad=0.04,rounding_size=0.14",
                     facecolor=CARD, edgecolor=GREEN, linewidth=1.8))
        ax.text(x + W / 2, 3.98, t, ha="center", va="center", fontsize=9.2,
                color=GREEN, weight="bold")
        ax.text(x + W / 2, 3.48, sub, ha="center", va="center", fontsize=8.2, color=GREY)
        ax.annotate("", xy=(x + W / 2, 2.42), xytext=(x + W / 2, 3.01),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.5))
    ax.text(6.15, 5.00, "Four different ways of reading the same clues",
            ha="center", fontsize=11, color=MID, style="italic")
    ax.add_patch(FancyBboxPatch((0.8, 1.25), 10.6, 1.15,
                 boxstyle="round,pad=0.04,rounding_size=0.14",
                 facecolor=GREEN, edgecolor="none"))
    ax.text(6.10, 2.06, "THE HEAD JUDGE", ha="center", va="center",
            fontsize=10.6, color="white", weight="bold")
    ax.text(6.10, 1.60, "learns how much to trust each one", ha="center",
            va="center", fontsize=9.2, color=PALE)
    ax.annotate("", xy=(6.10, 0.52), xytext=(6.10, 1.21),
                arrowprops=dict(arrowstyle="-|>", color=GREY, linewidth=1.6))
    ax.add_patch(FancyBboxPatch((4.05, -0.30), 4.1, 0.80,
                 boxstyle="round,pad=0.04,rounding_size=0.14",
                 facecolor="white", edgecolor=RED, linewidth=2.0))
    ax.text(6.10, 0.12, "One risk score for each area",
            ha="center", va="center", fontsize=10.8, color=RED, weight="bold")
    ax.set_xlim(0.3, 11.9); ax.set_ylim(-0.8, 5.4)
    ax.axis("off")
    save(fig, "pic_judges.png")


def pic_crispdm():
    """The six phases, and the two places the work went backwards."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    steps = ["Understand\nthe problem", "Understand\nthe data", "Prepare\nthe data",
             "Build the\nmodels", "Test them\nhonestly", "Publish the\nforecast"]
    cx, cy, R = 0, 0, 2.05
    ang = np.linspace(90, -210, 6, endpoint=True) * np.pi / 180
    pts = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]
    for i in range(6):
        x2, y2 = pts[(i + 1) % 6]
        x1, y1 = pts[i]
        if i < 5:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                         connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
                         mutation_scale=15, color=MUTED, linewidth=1.5,
                         shrinkA=34, shrinkB=34, zorder=1))
    for (x, y), t in zip(pts, steps):
        ax.add_patch(Circle((x, y), 0.70, facecolor=GREEN, edgecolor="none", zorder=2))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.6,
                color="white", weight="bold", zorder=3, linespacing=1.35)
    for a, b in [(4, 3), (2, 1)]:
        ax.add_patch(FancyArrowPatch(pts[a], pts[b],
                     connectionstyle="arc3,rad=0.62", arrowstyle="-|>",
                     mutation_scale=20, color=RED, linewidth=2.6,
                     linestyle=(0, (5, 2.2)), shrinkA=36, shrinkB=36, zorder=4))
    ax.text(0, 0.22, "I went back twice", ha="center", fontsize=11,
            color=RED, weight="bold")
    ax.text(0, -0.32, "and that is normal", ha="center", fontsize=9.4, color=GREY)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_crispdm.png")


def pic_lowband():
    """Where 100 recorded attacks actually fell, by band. Table 3.15, share of
    all events: Severe 14.0, High 19.7, Elevated 24.8, Low 41.5."""
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bands = [("Severe", 14, RED), ("High", 20, "#C2410C"),
             ("Elevated", 25, AMBER), ("Low", 41, MUTED)]
    i = 0
    for name, n, colour in bands:
        for _ in range(n):
            r, c = divmod(i, 20)
            ax.add_patch(Circle((c * 0.5, -r * 0.5), 0.20,
                        facecolor=colour, edgecolor="none"))
            i += 1
    x = 0
    for name, n, colour in bands:
        ax.add_patch(Circle((x, -2.85), 0.18, facecolor=colour, edgecolor="none"))
        ax.text(x + 0.32, -2.85, f"{name}  {n}", va="center", fontsize=10.5,
                color=GREEN if name != "Low" else RED,
                weight="bold" if name == "Low" else "normal")
        x += 2.55
    ax.set_xlim(-0.6, 10.4); ax.set_ylim(-3.4, 0.6)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_lowband.png")


def pic_reporting():
    """Estimated detection, Kano 0.706 and Ondo 0.213 (Chapter Three)."""
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for col, (state, seen, rate) in enumerate(
            [("Kano", 7, "about 7 in 10 get recorded"),
             ("Ondo", 2, "about 2 in 10 get recorded")]):
        x0 = col * 6.3
        for i in range(10):
            on = i < seen
            ax.add_patch(Circle((x0 + i * 0.55, 0), 0.23,
                        facecolor=RED if on else "white",
                        edgecolor=RED if on else MUTED,
                        linewidth=1.8, linestyle="-" if on else (0, (2, 2))))
        ax.text(x0 + 2.47, 0.85, state, ha="center", fontsize=14,
                color=GREEN, weight="bold")
        ax.text(x0 + 2.47, -0.88, rate, ha="center", fontsize=10, color=GREY)
    ax.text(5.9, -1.75, "Filled means recorded.  Hollow means it happened and nobody wrote it down.",
            ha="center", fontsize=9.6, color=MID, style="italic")
    ax.set_xlim(-0.8, 12.6); ax.set_ylim(-2.2, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "pic_reporting.png")


def pic_folds():
    """How much of the record taught the models and how much tested them.

    Weeks per round are Table 3.7. Each round tests on 52 weeks, so five rounds
    cover 260 of the 655 weeks, which is the 201,240 held-out area-weeks
    reported in Chapter Four once multiplied by the 774 areas.
    """
    trains = [395, 447, 499, 551, 603]
    TEST, TOTAL = 52, 655
    SC = 10.0 / TOTAL
    fig, ax = plt.subplots(figsize=(10.6, 5.9))

    # --- the whole record, split once ---
    split = 395 * SC
    ax.add_patch(FancyBboxPatch((0, 5.30), split, 0.62,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor=PALE, edgecolor=MUTED, linewidth=1.2))
    ax.add_patch(FancyBboxPatch((split, 5.30), 10.0 - split, 0.62,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor=RED, edgecolor="none"))
    ax.text(split / 2, 5.61, "60%", ha="center", va="center",
            fontsize=17, color=GREEN, weight="bold")
    ax.text(split + (10 - split) / 2, 5.61, "40%", ha="center", va="center",
            fontsize=17, color="white", weight="bold")
    ax.text(split / 2, 4.94, "395 weeks, learning only",
            ha="center", fontsize=10.2, color=GREY)
    ax.text(split + (10 - split) / 2, 4.94, "260 weeks, each tested once",
            ha="center", fontsize=10.2, color=RED)
    ax.text(-0.18, 5.61, "All 655 weeks", ha="right", va="center",
            fontsize=11.5, color=GREEN, weight="bold")

    ax.plot([-2.9, 11.9], [4.48, 4.48], color=FAINT, linewidth=1.1)
    ax.text(-2.9, 4.16, "Inside one round, the split is about 90 to 10",
            ha="left", fontsize=11, color=MID, style="italic")

    # --- the five rounds ---
    for i, tr in enumerate(trains):
        y = 3.40 - i * 0.76
        w = tr * SC
        t = TEST * SC
        ax.add_patch(FancyBboxPatch((0, y - 0.24), w, 0.48,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     facecolor=PALE, edgecolor=MUTED, linewidth=1.0))
        ax.add_patch(FancyBboxPatch((w, y - 0.24), t, 0.48,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     facecolor=RED, edgecolor="none"))
        ax.text(-0.18, y, f"Round {i+1}", ha="right", va="center",
                fontsize=10.4, color=GREEN, weight="bold")
        pct = tr / (tr + TEST) * 100
        ax.text(w / 2, y, f"learned from {tr} weeks", ha="center", va="center",
                fontsize=9.2, color=GREY)
        ax.text(w + t + 0.22, y, f"{pct:.0f}% learn   {100-pct:.0f}% test",
                ha="left", va="center", fontsize=9.8, color=GREEN)
    ax.set_xlim(-2.95, 12.6)
    ax.set_ylim(-0.75, 6.25)
    ax.axis("off")
    save(fig, "pic_folds.png")


if __name__ == "__main__":
    pic_rarity(); pic_coverage(); pic_reveal(); pic_gap()
    pic_judges(); pic_crispdm(); pic_lowband(); pic_reporting(); pic_folds()
