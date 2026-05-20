"""
bonus.py
Partie Bonus — Algorithmes Rapides d'Estimation de Mouvement

Trois algorithmes comparés sur une séquence de frames :
  1. Full Search      (référence, exhaustif)
  2. Three-Step Search (TSS)
  3. Diamond Search    (EPZS simplifié)

Exécution indépendante — ne modifie pas evaluate.py ni les autres fichiers.
Usage : python bonus.py
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.fft import dctn, idctn

from part1_preprocessing import load_frames, preprocess_frame
from iframe import encode_iframe, decode_iframe, quantize, dequantize, Q_MATRIX
from pframe import decode_p_frame, MB_SIZE, GOP_SIZE, sad

os.makedirs("output", exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  PARAMÈTRES BONUS
# ══════════════════════════════════════════════════════════════

SEARCH_WIN = 16   # fenêtre ±S pour Full Search et TSS
QF         = 1.0

# Losanges Diamond Search
LDSP = [(0,0),(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)]  # grand
SDSP = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]                               # petit


# ══════════════════════════════════════════════════════════════
#  ALGORITHME 1 — FULL SEARCH (référence)
# ══════════════════════════════════════════════════════════════

def full_search(curr_block, ref_frame, x, y, S=SEARCH_WIN):
    """
    Recherche exhaustive : teste toutes les positions dans [-S, +S]².
    Précis mais lent — (2S+1)² comparaisons par macrobloc.
    """
    best_mv   = (0, 0)
    best_sad  = float('inf')
    h, w      = ref_frame.shape
    comparisons = 0

    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            ry, rx = y + dy, x + dx
            if ry < 0 or rx < 0 or ry + MB_SIZE > h or rx + MB_SIZE > w:
                continue
            s = sad(curr_block, ref_frame[ry:ry+MB_SIZE, rx:rx+MB_SIZE])
            comparisons += 1
            if s < best_sad:
                best_sad, best_mv = s, (dy, dx)

    return best_mv, comparisons


# ══════════════════════════════════════════════════════════════
#  ALGORITHME 2 — THREE-STEP SEARCH (TSS)
# ══════════════════════════════════════════════════════════════

def three_step_search(curr_block, ref_frame, x, y, S=SEARCH_WIN):
    """
    Three-Step Search :
      - Départ au centre, grand pas = S//2
      - Teste 9 positions (3×3), se déplace vers le minimum
      - Réduit le pas de moitié, répète jusqu'à pas = 1
    ~27 comparaisons au lieu de 1089 (S=16).
    """
    h, w        = ref_frame.shape
    best_pos    = [y, x]
    best_sad_v  = float('inf')
    comparisons = 0
    step        = S // 2

    def check(ry, rx):
        nonlocal best_sad_v, best_pos, comparisons
        if ry < 0 or rx < 0 or ry + MB_SIZE > h or rx + MB_SIZE > w:
            return
        s = sad(curr_block, ref_frame[ry:ry+MB_SIZE, rx:rx+MB_SIZE])
        comparisons += 1
        if s < best_sad_v:
            best_sad_v = s
            best_pos   = [ry, rx]

    while step >= 1:
        cy, cx = best_pos
        for dy in [-step, 0, step]:
            for dx in [-step, 0, step]:
                check(cy + dy, cx + dx)
        step //= 2

    mv = (best_pos[0] - y, best_pos[1] - x)
    return mv, comparisons


# ══════════════════════════════════════════════════════════════
#  ALGORITHME 3 — DIAMOND SEARCH (EPZS simplifié)
# ══════════════════════════════════════════════════════════════

def diamond_search(curr_block, ref_frame, x, y, S=SEARCH_WIN):
    """
    Diamond Search :
      Phase 1 : Grand losange (9 pts) jusqu'à convergence
      Phase 2 : Petit losange (5 pts) pour affiner
    ~13 comparaisons en moyenne — utilisé dans H.264.
    """
    h, w        = ref_frame.shape
    best_pos    = [y, x]
    best_sad_v  = float('inf')
    comparisons = 0

    def check(ry, rx):
        nonlocal best_sad_v, best_pos, comparisons
        ry = int(np.clip(ry, 0, h - MB_SIZE))
        rx = int(np.clip(rx, 0, w - MB_SIZE))
        s  = sad(curr_block, ref_frame[ry:ry+MB_SIZE, rx:rx+MB_SIZE])
        comparisons += 1
        if s < best_sad_v:
            best_sad_v = s
            best_pos   = [ry, rx]

    # Phase 1 : Grand losange jusqu'à convergence
    for dy, dx in LDSP:
        check(best_pos[0] + dy, best_pos[1] + dx)

    prev = None
    while prev != best_pos:
        prev = best_pos[:]
        for dy, dx in LDSP:
            check(prev[0] + dy, prev[1] + dx)

    # Phase 2 : Petit losange
    for dy, dx in SDSP:
        check(best_pos[0] + dy, best_pos[1] + dx)

    mv = (best_pos[0] - y, best_pos[1] - x)
    return mv, comparisons


# ══════════════════════════════════════════════════════════════
#  ENCODEUR GÉNÉRIQUE (résidus DCT)
# ══════════════════════════════════════════════════════════════

def encode_with_algorithm(curr_Y, ref_Y, algo_fn, S=SEARCH_WIN):
    """
    Encode une P-frame avec n'importe quel algorithme de recherche.
    Retourne : dict encodé + stats (temps, comparaisons totales)
    """
    h, w           = curr_Y.shape
    motion_vectors = []
    residuals      = []
    total_comp     = 0
    t0             = time.time()

    for y in range(0, h - MB_SIZE + 1, MB_SIZE):
        for x in range(0, w - MB_SIZE + 1, MB_SIZE):
            curr_block = curr_Y[y:y+MB_SIZE, x:x+MB_SIZE]
            mv, comp   = algo_fn(curr_block, ref_Y, x, y, S)
            total_comp += comp
            motion_vectors.append(mv)

            dy, dx     = mv
            pred_y     = int(np.clip(y + dy, 0, h - MB_SIZE))
            pred_x     = int(np.clip(x + dx, 0, w - MB_SIZE))
            prediction = ref_Y[pred_y:pred_y+MB_SIZE, pred_x:pred_x+MB_SIZE]
            residual   = curr_block.astype(np.int32) - prediction.astype(np.int32)

            q_subs = []
            for si in range(0, MB_SIZE, 8):
                for sj in range(0, MB_SIZE, 8):
                    sub     = residual[si:si+8, sj:sj+8].astype(np.float32)
                    dct_sub = dctn(sub, norm='ortho')
                    q_subs.append(quantize(dct_sub, QF))
            residuals.append(q_subs)

    elapsed = time.time() - t0
    enc = {
        "type":           "P",
        "qf":             QF,
        "shape":          (h, w),
        "motion_vectors": motion_vectors,
        "residuals":      residuals,
    }
    return enc, elapsed, total_comp


# ══════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════

def psnr(orig, rec):
    mse = np.mean((orig.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def run_comparison(curr_Y, ref_Y):
    """
    Lance les 3 algorithmes sur la même paire de frames et
    retourne un dict de résultats pour chacun.
    """
    algos = [
        ("Full Search",        full_search,        "#e74c3c"),
        ("Three-Step Search",  three_step_search,  "#3498db"),
        ("Diamond Search",     diamond_search,      "#27ae60"),
    ]
    results = {}
    for name, fn, color in algos:
        print(f"  [{name}] encodage ...", end=" ", flush=True)
        enc, elapsed, total_comp = encode_with_algorithm(curr_Y, ref_Y, fn)
        rec_Y, res_map = decode_p_frame(enc, ref_Y)
        q_val = psnr(curr_Y, rec_Y)
        mv_arr = np.array(enc["motion_vectors"])
        results[name] = {
            "enc":        enc,
            "rec_Y":      rec_Y,
            "res_map":    res_map,
            "time":       elapsed,
            "comparisons":total_comp,
            "psnr":       q_val,
            "mv":         mv_arr,
            "color":      color,
        }
        print(f"{elapsed:.2f}s | {total_comp:,} SAD | PSNR {q_val:.1f} dB")
    return results


# ══════════════════════════════════════════════════════════════
#  VISUALISATION BONUS
# ══════════════════════════════════════════════════════════════

def section_label(ax, text, color):
    ax.text(
        0.5, 0.97, text,
        transform=ax.transAxes,
        fontsize=9, fontweight='bold', color='white',
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                  edgecolor='none', alpha=0.92),
        zorder=5
    )


def visualize_bonus(curr_Y, ref_Y, results, curr_frame_bgr):
    """
    Figure bonus — 5 lignes :
      Ligne 0 : barres comparatives (temps / comparaisons / PSNR)
      Ligne 1 : pattern de recherche illustré pour chaque algo
      Ligne 2 : frames reconstruites (les 3 algos côte à côte)
      Ligne 3 : vecteurs de mouvement (les 3 algos)
      Ligne 4 : cartes des résidus (les 3 algos) + tableau récap
    """
    names  = list(results.keys())
    colors = [results[n]["color"] for n in names]

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle(
        'Bonus — Comparaison des algorithmes d\'estimation de mouvement',
        fontsize=17, fontweight='bold', y=0.985, color='#1a1a2e'
    )
    gs = gridspec.GridSpec(
        5, 5, figure=fig,
        hspace=0.60, wspace=0.35,
        top=0.955, bottom=0.04, left=0.04, right=0.97
    )

    # ── LIGNE 0 : barres comparatives ─────────────────────────
    bar_color = '#2c3e50'

    # Temps d'exécution
    ax = fig.add_subplot(gs[0, 0:2])
    times = [results[n]["time"] for n in names]
    bars  = ax.barh(names, times, color=colors, edgecolor='white',
                    linewidth=1.2, height=0.5)
    for bar, v in zip(bars, times):
        ax.text(v + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}s', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Temps (secondes)', fontsize=10)
    ax.set_title('Temps d\'encodage', fontsize=11, fontweight='bold',
                 color=bar_color, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, max(times) * 1.2)

    # Comparaisons SAD
    ax = fig.add_subplot(gs[0, 2:4])
    comps = [results[n]["comparisons"] for n in names]
    bars  = ax.barh(names, comps, color=colors, edgecolor='white',
                    linewidth=1.2, height=0.5)
    for bar, v in zip(bars, comps):
        ax.text(v + max(comps)*0.01, bar.get_y() + bar.get_height()/2,
                f'{v:,}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Nombre de comparaisons SAD totales', fontsize=10)
    ax.set_title('Comparaisons SAD totales', fontsize=11, fontweight='bold',
                 color=bar_color, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, max(comps) * 1.2)

    # PSNR
    ax = fig.add_subplot(gs[0, 4])
    psnrs = [results[n]["psnr"] for n in names]
    bars  = ax.bar(names, psnrs, color=colors, edgecolor='white',
                   linewidth=1.2, width=0.5)
    for bar, v in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f'{v:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=10)
    ax.set_title('Qualité PSNR', fontsize=11, fontweight='bold',
                 color=bar_color, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8, rotation=10)
    ax.set_ylim(0, max(psnrs) * 1.15)
    ref_line = psnrs[0]  # Full Search comme référence
    ax.axhline(y=ref_line, color='#e74c3c', linestyle='--',
               linewidth=1.2, alpha=0.6, label='Réf. Full Search')
    ax.legend(fontsize=8)

    # ── LIGNE 1 : patterns de recherche illustrés ─────────────
    pattern_color = '#6c3483'

    def draw_search_pattern(ax, title, points, color, S=SEARCH_WIN):
        """Dessine le pattern de recherche sur une grille 2D."""
        grid = np.zeros((2*S+1, 2*S+1))
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1,
                  extent=[-S-0.5, S+0.5, S+0.5, -S-0.5])
        ax.axhline(0, color='#cccccc', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='#cccccc', linewidth=0.5, alpha=0.5)
        xs = [p[1] for p in points]
        ys = [p[0] for p in points]
        ax.scatter(xs, ys, s=60, c=color, zorder=3, edgecolors='white',
                   linewidths=0.8)
        ax.scatter([0], [0], s=120, c='#f39c12', zorder=4,
                   marker='*', label='Centre')
        ax.set_xlim(-S-1, S+1)
        ax.set_ylim(S+1, -S-1)
        ax.set_xlabel('dx', fontsize=8)
        ax.set_ylabel('dy', fontsize=8)
        ax.tick_params(labelsize=7)
        section_label(ax, title, pattern_color)

    # Full Search : toutes les positions
    ax = fig.add_subplot(gs[1, 0])
    S = SEARCH_WIN
    all_pts = [(dy, dx) for dy in range(-S, S+1) for dx in range(-S, S+1)]
    draw_search_pattern(ax, f'Full Search — {len(all_pts)} positions', all_pts,
                        results["Full Search"]["color"])
    ax.text(0.5, 0.04, f'{len(all_pts)} tests/MB', transform=ax.transAxes,
            ha='center', fontsize=8, color='white',
            bbox=dict(facecolor='#e74c3c', alpha=0.8, boxstyle='round,pad=0.2'))

    # TSS : 3 niveaux de pas
    ax = fig.add_subplot(gs[1, 1])
    tss_pts = []
    step = S // 2
    pos = [0, 0]
    while step >= 1:
        for dy in [-step, 0, step]:
            for dx in [-step, 0, step]:
                tss_pts.append((pos[0]+dy, pos[1]+dx))
        step //= 2
    tss_pts = list(set(tss_pts))
    draw_search_pattern(ax, f'Three-Step Search — ~{len(tss_pts)} positions', tss_pts,
                        results["Three-Step Search"]["color"])
    ax.text(0.5, 0.04, f'~{len(tss_pts)} tests/MB', transform=ax.transAxes,
            ha='center', fontsize=8, color='white',
            bbox=dict(facecolor='#3498db', alpha=0.8, boxstyle='round,pad=0.2'))

    # Diamond : grand + petit losange
    ax = fig.add_subplot(gs[1, 2])
    diamond_pts = list(set(LDSP + SDSP))
    draw_search_pattern(ax, f'Diamond Search — ~{len(diamond_pts)} positions', diamond_pts,
                        results["Diamond Search"]["color"])
    ax.text(0.5, 0.04, f'~{len(diamond_pts)} tests/MB', transform=ax.transAxes,
            ha='center', fontsize=8, color='white',
            bbox=dict(facecolor='#27ae60', alpha=0.8, boxstyle='round,pad=0.2'))

    # Gain de vitesse (barres speedup)
    ax = fig.add_subplot(gs[1, 3:])
    ref_comp  = results["Full Search"]["comparisons"]
    speedups  = [ref_comp / results[n]["comparisons"] for n in names]
    bar_su    = ax.bar(names, speedups, color=colors, edgecolor='white',
                       linewidth=1.2, width=0.5)
    for bar, v in zip(bar_su, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.1f}×', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.axhline(y=1, color='#e74c3c', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Référence (Full Search)')
    ax.set_ylabel('Gain de vitesse (×)', fontsize=10)
    ax.set_title('Accélération vs Full Search', fontsize=11,
                 fontweight='bold', color=bar_color, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.legend(fontsize=8)

    # ── LIGNE 2 : frames reconstruites ────────────────────────
    rec_color = '#1a6b3a'
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(results[name]["rec_Y"], cmap='gray')
        section_label(ax, f'{name}  PSNR {results[name]["psnr"]:.1f} dB', rec_color)
        ax.axis('off')

    # Frame originale (référence visuelle)
    ax = fig.add_subplot(gs[2, 3])
    curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    ax.imshow(curr_gray, cmap='gray')
    section_label(ax, 'Frame originale (Y)', '#7f8c8d')
    ax.axis('off')

    # Différences Full Search vs Diamond
    ax = fig.add_subplot(gs[2, 4])
    diff_fd = np.abs(
        results["Full Search"]["rec_Y"].astype(np.int32) -
        results["Diamond Search"]["rec_Y"].astype(np.int32)
    ).astype(np.uint8)
    im = ax.imshow(diff_fd, cmap='hot', vmin=0, vmax=20)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    section_label(ax, 'Diff Full Search vs Diamond', '#c0392b')
    ax.axis('off')

    # ── LIGNE 3 : vecteurs de mouvement ───────────────────────
    mv_color = '#1a3a6b'
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[3, col])
        ax.imshow(curr_gray, cmap='gray', alpha=0.6)
        enc   = results[name]["enc"]
        mvs   = enc["motion_vectors"]
        h_f, w_f = enc["shape"]
        idx_mv = 0
        for yy in range(0, h_f - MB_SIZE + 1, MB_SIZE):
            for xx in range(0, w_f - MB_SIZE + 1, MB_SIZE):
                dy, dx = mvs[idx_mv]
                if dy != 0 or dx != 0:
                    ax.quiver(
                        xx + MB_SIZE//2, yy + MB_SIZE//2, dx, dy,
                        color=results[name]["color"],
                        scale=1, scale_units='xy', angles='xy',
                        width=0.004, headwidth=4, alpha=0.85
                    )
                idx_mv += 1
        section_label(ax, f'MV — {name}', mv_color)
        ax.axis('off')

    # Statistiques MV (amplitude moyenne)
    ax = fig.add_subplot(gs[3, 3:])
    mv_means = []
    mv_nonzero = []
    for name in names:
        mv_arr = results[name]["mv"]
        amps   = np.sqrt(mv_arr[:, 0]**2 + mv_arr[:, 1]**2)
        mv_means.append(float(np.mean(amps)))
        nz = int(np.sum(np.any(mv_arr != 0, axis=1)))
        mv_nonzero.append(100 * nz / len(mv_arr))

    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, mv_means,  width, color=colors, alpha=0.85,
           edgecolor='white', label='Amplitude moy. (px)')
    ax2b = ax.twinx()
    ax2b.bar(x_pos + width/2, mv_nonzero, width, color=colors, alpha=0.45,
             edgecolor='white', hatch='//', label='MV non-nuls (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Amplitude moyenne (px)', fontsize=9)
    ax2b.set_ylabel('MV non-nuls (%)', fontsize=9)
    ax.set_title('Statistiques vecteurs de mouvement', fontsize=11,
                 fontweight='bold', color=bar_color, pad=8)
    ax.spines[['top']].set_visible(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    # ── LIGNE 4 : cartes des résidus + tableau récap ──────────
    res_color = '#7d3c98'
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[4, col])
        im_r = ax.imshow(results[name]["res_map"],
                         cmap='RdBu', vmin=-50, vmax=50)
        plt.colorbar(im_r, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        section_label(ax, f'Résidus — {name}', res_color)
        ax.axis('off')

    # Tableau récapitulatif final
    ax = fig.add_subplot(gs[4, 3:])
    ax.axis('off')
    ref_time = results["Full Search"]["time"]
    ref_comp_val = results["Full Search"]["comparisons"]
    ref_psnr = results["Full Search"]["psnr"]
    table_data = [
        ['Algorithme', 'Temps (s)', 'SAD totaux', 'Gain ×', 'PSNR (dB)', 'Δ PSNR'],
    ]
    for name in names:
        r = results[name]
        gain   = ref_comp_val / r["comparisons"]
        delta  = r["psnr"] - ref_psnr
        sign   = '+' if delta >= 0 else ''
        table_data.append([
            name,
            f'{r["time"]:.2f}',
            f'{r["comparisons"]:,}',
            f'{gain:.1f}×',
            f'{r["psnr"]:.2f}',
            f'{sign}{delta:.2f}',
        ])

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center', cellLoc='center',
        bbox=[0.0, 0.05, 1.0, 0.90]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(colors[row - 1] + '22')  # couleur algo, très transparent
    ax.set_title('Récapitulatif comparatif', fontsize=11, fontweight='bold',
                 color=bar_color, pad=10)

    out_path = "output/bonus_motion_estimation.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n[OK] Figure bonus sauvegardée → {out_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  BONUS — Algorithmes rapides d'estimation de mouvement")
    print("=" * 60)

    # 1. Charger les frames
    frames = load_frames("frames")
    if len(frames) < 2:
        print("ERREUR : il faut au moins 2 frames dans 'frames/'")
        exit(1)
    preprocessed = [preprocess_frame(f) for f in frames]
    print(f"[OK] {len(frames)} frames chargées\n")

    # 2. I-frame 0 → référence
    print("[INFO] Encodage I-frame 0 (référence) ...")
    enc_i   = encode_iframe(preprocessed[0], qf=QF)
    ref_bgr = decode_iframe(enc_i)
    ref_Y   = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    print("[OK] I-frame encodée\n")

    # 3. Frame 1 → P-frame comparée avec les 3 algos
    curr_Y        = preprocessed[1]["Y"]
    curr_frame_bgr = frames[1]
    print(f"[INFO] Comparaison des 3 algorithmes sur frame 1 (S={SEARCH_WIN}) :")
    results = run_comparison(curr_Y, ref_Y)

    # 4. Résumé console
    print("\n── Résumé ───────────────────────────────────────────────")
    ref_t = results["Full Search"]["time"]
    ref_c = results["Full Search"]["comparisons"]
    for name, r in results.items():
        gain = ref_c / r["comparisons"]
        print(f"  {name:<22} {r['time']:.2f}s  "
              f"{r['comparisons']:>8,} SAD  "
              f"gain {gain:5.1f}×  "
              f"PSNR {r['psnr']:.2f} dB")
    print("─────────────────────────────────────────────────────────\n")

    # 5. Visualisation
    print("[INFO] Génération de la figure bonus ...")
    visualize_bonus(curr_Y, ref_Y, results, curr_frame_bgr)
    print("[OK] Partie bonus terminée !")
    print("     Partie classique : python evaluate.py")
    print("     Partie bonus     : python bonus.py")