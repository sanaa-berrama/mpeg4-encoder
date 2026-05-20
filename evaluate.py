"""
evaluate.py
Partie 5 — Évaluation & Visualisation
Lit output/video.bin et génère tous les graphes demandés.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import dctn, idctn

from part1_preprocessing import load_frames, preprocess_frame
from iframe import (decode_iframe, dct2d, quantize, dequantize, Q_MATRIX)
from pframe import decode_p_frame, MB_SIZE, GOP_SIZE
from entropy import decode_from_bin

os.makedirs("output", exist_ok=True)
BIN_PATH = "output/video.bin"


# ══════════════════════════════════════════════════════════════
#  5a — MÉTRIQUES
# ══════════════════════════════════════════════════════════════

def compute_psnr(orig, rec):
    mse = np.mean((orig.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def decode_all_frames(video_data):
    """
    Décode toutes les frames depuis video_data (issu de decode_from_bin).
    Retourne la liste des frames BGR reconstruites.
    """
    frames_encoded = video_data['frames']
    reconstructed  = []
    ref_Y          = None

    for enc in frames_encoded:
        if enc['type'] == 'I':
            bgr   = decode_iframe(enc)
            ref_Y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            reconstructed.append(bgr)
        else:
            rec_Y, _ = decode_p_frame(enc, ref_Y)
            ref_Y    = rec_Y
            # Reconstruire BGR approximatif depuis Y seul (Cb/Cr neutres)
            h, w = rec_Y.shape
            cb   = np.full((h, w), 128, dtype=np.uint8)
            cr   = np.full((h, w), 128, dtype=np.uint8)
            ycrcb = cv2.merge([rec_Y, cr, cb])
            bgr   = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            reconstructed.append(bgr)

    return reconstructed


def compute_metrics(original_frames, reconstructed_frames, bin_path):
    """Calcule taux de compression + PSNR par frame."""
    original_size   = sum(f.nbytes for f in original_frames)
    compressed_size = os.path.getsize(bin_path)
    ratio           = original_size / compressed_size

    psnr_values = []
    for orig, rec in zip(original_frames, reconstructed_frames):
        psnr_values.append(compute_psnr(orig, rec))

    print(f"\n── Métriques Partie 5 ────────────────────────────")
    print(f"  Taille originale    : {original_size:,} octets")
    print(f"  Taille compressée   : {compressed_size:,} octets")
    print(f"  Taux de compression : {ratio:.2f}x")
    print(f"  PSNR moyen          : {np.mean(psnr_values):.2f} dB")
    print(f"  PSNR min            : {np.min(psnr_values):.2f} dB")
    print(f"  PSNR max            : {np.max(psnr_values):.2f} dB")
    print(f"──────────────────────────────────────────────────\n")
    return ratio, psnr_values


# ══════════════════════════════════════════════════════════════
#  5b — VISUALISATION PIPELINE COMPLÈTE
# ══════════════════════════════════════════════════════════════

def visualize_pipeline(original_frames, reconstructed_frames,
                       video_data, psnr_values, ratio):
    """
    Figure principale — pipeline complet, mise en page aérée :

      Section A  (ligne 1, 5 cols) : frames orig. | recons. | diff | canaux Y/Cb/Cr
      Section B  (ligne 2, 5 cols) : pipeline DCT 4 étapes + histogramme coefficients
      Section C  (ligne 3, 5 cols) : P-frame : référence | courante | MV | résidu | recons.
      Section D  (ligne 4)         : courbe PSNR (3 cols) + tableau de bord (2 cols)
    """
    from matplotlib.patches import Patch

    frames_enc = video_data['frames']
    n          = len(original_frames)
    i_count    = sum(1 for e in frames_enc if e['type'] == 'I')
    p_count    = sum(1 for e in frames_enc if e['type'] == 'P')
    finite_psnr = [p if p != float('inf') else 60 for p in psnr_values]

    # ── Préparer les données ───────────────────────────────────
    prep0        = preprocess_frame(original_frames[0])
    Y0, Cb0, Cr0 = prep0['Y'], prep0['Cb_sub'], prep0['Cr_sub']

    # Bloc 8×8 au centre de Y
    h0, w0 = Y0.shape
    by = ((h0 // 2) // 8) * 8
    bx = ((w0 // 2) // 8) * 8
    raw_blk = Y0[by:by+8, bx:bx+8].astype(np.float32)
    dct_blk = dct2d(raw_blk)
    q_blk   = quantize(dct_blk, 1.0).astype(np.float32)
    dq_blk  = dequantize(q_blk.astype(np.int16), 1.0)
    rec_blk = np.clip(idctn(dq_blk, norm='ortho') + 128, 0, 255).astype(np.uint8)

    # Première P-frame
    p_enc  = next((e for e in frames_enc if e['type'] == 'P'), None)
    p_idx  = next((i for i, e in enumerate(frames_enc) if e['type'] == 'P'), 1)
    ref_Y  = cv2.cvtColor(decode_iframe(frames_enc[0]), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if p_enc:
        rec_Y_p, res_map = decode_p_frame(p_enc, ref_Y)
    else:
        rec_Y_p = ref_Y
        res_map = np.zeros_like(ref_Y, dtype=np.float32)

    # ── Créer la figure ────────────────────────────────────────
    # 4 lignes × 5 colonnes avec espacements généreux
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle(
        'Pipeline Encodeur MPEG-4 — Vue complète (Partie 5)',
        fontsize=17, fontweight='bold', y=0.985, color='#1a1a2e'
    )

    gs = gridspec.GridSpec(
        4, 5, figure=fig,
        hspace=0.60, wspace=0.35,
        top=0.955, bottom=0.04, left=0.04, right=0.97
    )

    # ── helper : titre de section ──────────────────────────────
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

    # ══════════════════════════════════════════════════════════
    #  LIGNE 0 — Frames & canaux couleur
    # ══════════════════════════════════════════════════════════
    row_color = '#2c3e50'

    # Frame originale 0
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(original_frames[0], cv2.COLOR_BGR2RGB))
    section_label(ax, f'Frame 0 original ({frames_enc[0]["type"]})', row_color)
    ax.axis('off')

    # Frame reconstruite 0
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cv2.cvtColor(reconstructed_frames[0], cv2.COLOR_BGR2RGB))
    section_label(ax, f'Frame 0 reconstruite  PSNR {psnr_values[0]:.1f} dB', row_color)
    ax.axis('off')

    # Différence
    ax = fig.add_subplot(gs[0, 2])
    diff0 = cv2.absdiff(original_frames[0], reconstructed_frames[0])
    ax.imshow(diff0)
    section_label(ax, 'Différence absolue (frame 0)', row_color)
    ax.axis('off')

    # Canal Y
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(Y0, cmap='gray')
    section_label(ax, f'Canal Y — luminance  {Y0.shape}', row_color)
    ax.axis('off')

    # Camembert I/P (dernière colonne, ligne 0)
    ax = fig.add_subplot(gs[0, 4])
    wedges, texts, autotexts = ax.pie(
        [i_count, p_count],
        labels=[f'I-frames\n({i_count})', f'P-frames\n({p_count})'],
        colors=['#e74c3c', '#3498db'],
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(linewidth=1.5, edgecolor='white'),
        textprops=dict(fontsize=9)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.text(0.5, 0.97, 'Répartition I / P',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color='#2c3e50', ha='center', va='top')

    # ══════════════════════════════════════════════════════════
    #  LIGNE 1 — Pipeline DCT (4 étapes) + histogramme coefficients
    # ══════════════════════════════════════════════════════════
    row_color = '#6c3483'

    dct_steps = [
        (raw_blk, 'Bloc 8×8 original', 'gray',    None,   None),
        (dct_blk, 'Coefficients DCT',  'seismic',  None,   None),
        (q_blk,   'Après quantification','seismic', None,   None),
        (rec_blk, 'Reconstruit (IDCT)', 'gray',    0,      255),
    ]
    for col, (data, title, cmap, vmin, vmax) in enumerate(dct_steps):
        ax = fig.add_subplot(gs[1, col])
        kwargs = {} if vmin is None else {'vmin': vmin, 'vmax': vmax}
        im = ax.imshow(data, cmap=cmap, **kwargs)
        section_label(ax, title, row_color)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        ax.axis('off')

    # Histogramme des coefficients DCT quantifiés
    ax = fig.add_subplot(gs[1, 4])
    zero_pct    = 100 * np.sum(q_blk == 0) / q_blk.size
    nonzero_pct = 100 - zero_pct
    bars = ax.bar(
        ['Non-nuls', 'Zéros'],
        [nonzero_pct, zero_pct],
        color=['#e74c3c', '#bdc3c7'],
        edgecolor='white', linewidth=1.5, width=0.55
    )
    for bar_obj, val in zip(bars, [nonzero_pct, zero_pct]):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                val + 1.5, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Proportion (%)', fontsize=9)
    ax.set_title('Coefficients DCT quantifiés\n(bloc 8×8 central)',
                 fontsize=9, fontweight='bold', color=row_color, pad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)

    # ══════════════════════════════════════════════════════════
    #  LIGNE 2 — P-frame complète
    # ══════════════════════════════════════════════════════════
    row_color = '#1a6b3a'

    if p_enc:
        curr_Y_gray = cv2.cvtColor(
            original_frames[p_idx], cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hp, wp = p_enc['shape']

        # Référence
        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(ref_Y, cmap='gray')
        section_label(ax, 'Frame de référence (Y)', row_color)
        ax.axis('off')

        # P-frame courante
        ax = fig.add_subplot(gs[2, 1])
        ax.imshow(curr_Y_gray, cmap='gray')
        section_label(ax, f'P-frame courante (Y) — frame {p_idx}', row_color)
        ax.axis('off')

        # Vecteurs de mouvement
        ax = fig.add_subplot(gs[2, 2])
        ax.imshow(curr_Y_gray, cmap='gray', alpha=0.7)
        mvs = p_enc['motion_vectors']
        idx_mv = 0
        for y in range(0, hp - MB_SIZE + 1, MB_SIZE):
            for x in range(0, wp - MB_SIZE + 1, MB_SIZE):
                dy, dx = mvs[idx_mv]
                if dy != 0 or dx != 0:
                    ax.quiver(
                        x + MB_SIZE // 2, y + MB_SIZE // 2, dx, dy,
                        color='#e74c3c', scale=1, scale_units='xy',
                        angles='xy', width=0.004, headwidth=4
                    )
                idx_mv += 1
        section_label(ax, 'Vecteurs de mouvement', row_color)
        ax.axis('off')

        # Résidu
        ax = fig.add_subplot(gs[2, 3])
        im_r = ax.imshow(res_map, cmap='RdBu', vmin=-50, vmax=50)
        section_label(ax, 'Carte des résidus', row_color)
        cb = plt.colorbar(im_r, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        ax.axis('off')

        # P-frame reconstruite
        ax = fig.add_subplot(gs[2, 4])
        ax.imshow(rec_Y_p, cmap='gray')
        section_label(
            ax,
            f'P-frame reconstruite  PSNR {psnr_values[p_idx]:.1f} dB',
            row_color
        )
        ax.axis('off')

    # ══════════════════════════════════════════════════════════
    #  LIGNE 3 — Courbe PSNR + tableau de bord
    # ══════════════════════════════════════════════════════════

    # Courbe PSNR (3 premières colonnes)
    ax = fig.add_subplot(gs[3, :3])
    colors_psnr = ['#e74c3c' if frames_enc[i]['type'] == 'I' else '#3498db'
                   for i in range(len(finite_psnr))]
    ax.bar(range(len(finite_psnr)), finite_psnr, color=colors_psnr, alpha=0.85,
           edgecolor='white', linewidth=0.4)
    mean_val = np.mean(finite_psnr)
    ax.axhline(y=mean_val, color='#e67e22', linestyle='--', linewidth=1.8,
               label=f'Moyenne : {mean_val:.1f} dB')
    ax.axhline(y=30, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.8,
               label='Seuil qualité : 30 dB')
    ax.set_xlabel('Index de frame', fontsize=10)
    ax.set_ylabel('PSNR (dB)', fontsize=10)
    ax.set_title('PSNR par frame', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_ylim(0, max(finite_psnr) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    legend_elements = [
        Patch(facecolor='#e74c3c', label='I-frame'),
        Patch(facecolor='#3498db', label='P-frame'),
        plt.Line2D([0], [0], color='#e67e22', linestyle='--',
                   linewidth=1.8, label=f'Moyenne : {mean_val:.1f} dB'),
        plt.Line2D([0], [0], color='#27ae60', linestyle=':', linewidth=1.5,
                   label='Seuil : 30 dB'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9,
              loc='upper right')

    # Tableau de bord (2 dernières colonnes)
    ax2 = fig.add_subplot(gs[3, 3:])
    ax2.axis('off')
    table_data = [
        ['Frames totales',    str(n)],
        ['I-frames',          str(i_count)],
        ['P-frames',          str(p_count)],
        ['GOP size',          str(video_data.get('gop', GOP_SIZE))],
        ['Résolution',        f"{video_data.get('width','?')}×{video_data.get('height','?')}"],
        ['FPS',               str(video_data.get('fps', 25))],
        ['Taux compression',  f"{ratio:.2f}×"],
        ['PSNR moyen',        f"{np.mean(finite_psnr):.2f} dB"],
        ['PSNR min',          f"{np.min(finite_psnr):.2f} dB"],
        ['PSNR max',          f"{np.max(finite_psnr):.2f} dB"],
    ]
    tbl = ax2.table(
        cellText=table_data,
        colLabels=['Paramètre', 'Valeur'],
        loc='center', cellLoc='center',
        bbox=[0.05, 0.0, 0.90, 1.0]   # [left, bottom, width, height] en coords axes
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    # Style : en-tête sombre, lignes alternées
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#eaf0f6')
        else:
            cell.set_facecolor('white')
    ax2.set_title('Tableau de bord', fontsize=11, fontweight='bold',
                  color='#2c3e50', pad=10)

    # ── Sauvegarder ───────────────────────────────────────────
    out_path = "output/part5_pipeline_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[OK] Visualisation sauvegardée dans {out_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Charger les frames originales
    print("[INFO] Chargement des frames originales ...")
    original_frames = load_frames("frames")
    print(f"[OK] {len(original_frames)} frames chargées")

    # 2. Décoder depuis le .bin
    print(f"\n[INFO] Décodage depuis {BIN_PATH} ...")
    video_data = decode_from_bin(BIN_PATH)

    # 3. Reconstruire toutes les frames
    print("\n[INFO] Reconstruction des frames ...")
    reconstructed_frames = decode_all_frames(video_data)
    print(f"[OK] {len(reconstructed_frames)} frames reconstruites")

    # 4. Calculer les métriques
    ratio, psnr_values = compute_metrics(
        original_frames, reconstructed_frames, BIN_PATH
    )

    # 5. Visualisation complète
    print("\n[INFO] Génération de la visualisation ...")
    visualize_pipeline(
        original_frames, reconstructed_frames,
        video_data, psnr_values, ratio
    )

    print("\n[OK] Partie 5 terminée !")
    print(f"     Fichier : output/part5_pipeline_visualization.png")