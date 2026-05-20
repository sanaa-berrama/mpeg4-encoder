"""
pframe.py
Partie 3 — Encodage/Decodage des P-frames
Estimation de mouvement : Full Search (exhaustif)
+ Calcul et codage des residus (DCT + quantification)
"""

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

from iframe import (
    Q_MATRIX, quantize, dequantize,
    split_into_blocks, reconstruct_from_blocks
)

# ─── PARAMETRES GOP ────────────────────────────────────────────
GOP_SIZE   = 10   # toutes les GOP_SIZE frames → I-frame
SEARCH_WIN = 16   # fenetre de recherche ±S pixels  (S=16 → 1089 compar./MB)
MB_SIZE    = 16   # taille d'un macrobloc (16×16)


# ══════════════════════════════════════════════════════════════
#  HELPERS GOP
# ══════════════════════════════════════════════════════════════

def get_frame_type(frame_index, gop_size=GOP_SIZE):
    """Retourne 'I' ou 'P' selon la position dans le GOP."""
    return 'I' if frame_index % gop_size == 0 else 'P'


def count_frame_types(n_frames, gop_size=GOP_SIZE):
    """Retourne le nombre de I-frames et P-frames pour n_frames."""
    i_count = sum(1 for i in range(n_frames) if get_frame_type(i, gop_size) == 'I')
    p_count = n_frames - i_count
    return i_count, p_count


# ══════════════════════════════════════════════════════════════
#  METRIQUE DE SIMILARITE
# ══════════════════════════════════════════════════════════════

def sad(block_a, block_b):
    """Sum of Absolute Differences — mesure la similarite entre deux blocs."""
    return int(np.sum(np.abs(block_a.astype(np.int32) - block_b.astype(np.int32))))


# ══════════════════════════════════════════════════════════════
#  FULL SEARCH (algorithme principal — Partie 3)
# ══════════════════════════════════════════════════════════════

def full_search(curr_block, ref_frame, x, y, S=SEARCH_WIN):
    """
    Recherche exhaustive (Full Search / Brute Force).

    Pour chaque macrobloc 16x16 de la frame courante, on teste TOUTES les
    positions (dy, dx) dans la fenetre [-S, +S] x [-S, +S] de la frame
    de reference et on garde celle qui minimise la SAD.

    Nombre de comparaisons : (2*S+1)^2 par macrobloc.
      S=8  →  289 comparaisons/MB
      S=16 → 1089 comparaisons/MB   ← valeur par defaut ici

    C'est l'algorithme le plus precis mais aussi le plus lent.
    Sur une frame 640x480 avec S=16 :
      1200 MB x 1089 comparaisons = ~1,3 million de SAD calcules par frame.
    """
    best_mv  = (0, 0)
    best_sad = float('inf')
    h, w     = ref_frame.shape

    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            ry = y + dy
            rx = x + dx
            # Verifier que le bloc candidat reste dans la frame
            if ry < 0 or rx < 0:
                continue
            if ry + MB_SIZE > h or rx + MB_SIZE > w:
                continue
            candidate = ref_frame[ry:ry + MB_SIZE, rx:rx + MB_SIZE]
            s = sad(curr_block, candidate)
            if s < best_sad:
                best_sad = s
                best_mv  = (dy, dx)

    return best_mv   # (dy, dx) = vecteur de mouvement optimal


# ══════════════════════════════════════════════════════════════
#  ENCODEUR P-FRAME
# ══════════════════════════════════════════════════════════════

def encode_p_frame(curr_frame_Y, ref_frame_Y, qf=1.0, S=SEARCH_WIN):
    """
    Encode une P-frame complete (canal Y uniquement).

    Pipeline pour chaque macrobloc 16x16 :
      1. Full Search → vecteur de mouvement (dy, dx)
      2. Prediction  → bloc de reference deplace par (dy, dx)
      3. Residu      → difference entre bloc courant et prediction
      4. DCT + Quantification sur le residu (4 sous-blocs 8x8)

    Parametres :
        curr_frame_Y : canal Y de la frame courante (uint8)
        ref_frame_Y  : canal Y de la frame de reference RECONSTRUITE (uint8)
        qf           : Quality Factor pour la quantification des residus
        S            : demi-taille de la fenetre de recherche (defaut 16)

    Retourne un dict avec toutes les donnees necessaires au decodeur.
    """
    h, w = curr_frame_Y.shape
    motion_vectors = []
    residuals      = []

    n_mb = ((h - MB_SIZE) // MB_SIZE + 1) * ((w - MB_SIZE) // MB_SIZE + 1)
    n_comp = (2 * S + 1) ** 2
    print(f"  Full Search : {n_mb} macroblocs x {n_comp} comparaisons = "
          f"~{n_mb * n_comp:,} SAD calcules")

    mb_idx = 0
    for y in range(0, h - MB_SIZE + 1, MB_SIZE):
        for x in range(0, w - MB_SIZE + 1, MB_SIZE):
            curr_block = curr_frame_Y[y:y + MB_SIZE, x:x + MB_SIZE]

            # 1. Estimation de mouvement par Full Search
            mv = full_search(curr_block, ref_frame_Y, x, y, S)
            motion_vectors.append(mv)

            # 2. Prediction : bloc de reference deplace
            dy, dx = mv
            pred_y = np.clip(y + dy, 0, h - MB_SIZE)
            pred_x = np.clip(x + dx, 0, w - MB_SIZE)
            prediction = ref_frame_Y[pred_y:pred_y + MB_SIZE,
                                     pred_x:pred_x + MB_SIZE]

            # 3. Residu = difference (peut etre negatif)
            residual = curr_block.astype(np.int32) - prediction.astype(np.int32)

            # 4. DCT + Quantification sur les 4 sous-blocs 8x8 du macrobloc
            q_subblocks = []
            for si in range(0, MB_SIZE, 8):
                for sj in range(0, MB_SIZE, 8):
                    sub     = residual[si:si+8, sj:sj+8].astype(np.float32)
                    dct_sub = dctn(sub, norm='ortho')
                    q_sub   = quantize(dct_sub, qf)
                    q_subblocks.append(q_sub)
            residuals.append(q_subblocks)

            mb_idx += 1

    return {
        "type":           "P",
        "qf":             qf,
        "shape":          (h, w),
        "motion_vectors": motion_vectors,
        "residuals":      residuals,
        "algorithm":      "full",
        "search_window":  S,
    }


# ══════════════════════════════════════════════════════════════
#  DECODEUR P-FRAME
# ══════════════════════════════════════════════════════════════

def decode_p_frame(encoded_p, ref_frame_Y):
    """
    Decode une P-frame a partir des vecteurs de mouvement et residus encodes.

    Pour chaque macrobloc :
      1. Prediction  → bloc de reference deplace par le vecteur de mouvement
      2. Residu rec. → de-quantification + IDCT sur les 4 sous-blocs 8x8
      3. Reconstruction = prediction + residu

    Retourne :
        reconstructed_Y : canal Y reconstruit (uint8)
        residual_map    : carte des residus (float32, pour visualisation)
    """
    h, w      = encoded_p["shape"]
    qf        = encoded_p["qf"]
    mvs       = encoded_p["motion_vectors"]
    residuals = encoded_p["residuals"]

    reconstructed = np.zeros((h, w), dtype=np.float32)
    residual_map  = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for y in range(0, h - MB_SIZE + 1, MB_SIZE):
        for x in range(0, w - MB_SIZE + 1, MB_SIZE):
            dy, dx = mvs[idx]

            # 1. Prediction depuis la frame de reference
            pred_y     = np.clip(y + dy, 0, h - MB_SIZE)
            pred_x     = np.clip(x + dx, 0, w - MB_SIZE)
            prediction = ref_frame_Y[pred_y:pred_y + MB_SIZE,
                                     pred_x:pred_x + MB_SIZE].astype(np.float32)

            # 2. Decoder le residu (dequantification + IDCT sur 4 sous-blocs)
            q_subblocks = residuals[idx]
            residual_mb = np.zeros((MB_SIZE, MB_SIZE), dtype=np.float32)
            sub_idx = 0
            for si in range(0, MB_SIZE, 8):
                for sj in range(0, MB_SIZE, 8):
                    dct_sub = dequantize(q_subblocks[sub_idx], qf)
                    sub_rec = idctn(dct_sub, norm='ortho')
                    residual_mb[si:si+8, sj:sj+8] = sub_rec
                    sub_idx += 1

            # 3. Reconstruction = prediction + residu
            reconstructed[y:y + MB_SIZE, x:x + MB_SIZE] = prediction + residual_mb
            residual_map [y:y + MB_SIZE, x:x + MB_SIZE] = residual_mb

            idx += 1

    reconstructed_Y = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed_Y, residual_map


# ══════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════

def visualize_pframe(curr_frame_bgr, ref_frame_Y, encoded_p,
                     reconstructed_Y, residual_map):
    """
    Affiche 6 sous-figures claires sur 2 lignes :
      Ligne 1 : frame de reference | frame courante | vecteurs de mouvement
      Ligne 2 : carte des residus  | frame reconstruite | erreur de reconstruction
    """
    curr_Y = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    h, w   = encoded_p["shape"]
    mvs    = encoded_p["motion_vectors"]
    S      = encoded_p["search_window"]
    qf     = encoded_p["qf"]
    n_comp = (2 * S + 1) ** 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Partie 3 — P-frame  |  Full Search  |  S={S}  ({n_comp} compar./MB)  |  QF={qf}",
        fontsize=14, fontweight='bold', y=1.01
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.12)

    # [0,0] Frame de reference
    axes[0, 0].imshow(ref_frame_Y, cmap='gray')
    axes[0, 0].set_title("① Frame de reference (Y)", fontsize=11, pad=8)
    axes[0, 0].axis('off')

    # [0,1] Frame courante
    axes[0, 1].imshow(curr_Y, cmap='gray')
    axes[0, 1].set_title("② Frame courante (Y)", fontsize=11, pad=8)
    axes[0, 1].axis('off')

    # [0,2] Vecteurs de mouvement
    axes[0, 2].imshow(curr_Y, cmap='gray', alpha=0.75)
    idx = 0
    for y in range(0, h - MB_SIZE + 1, MB_SIZE):
        for x in range(0, w - MB_SIZE + 1, MB_SIZE):
            dy, dx = mvs[idx]
            if dy != 0 or dx != 0:
                axes[0, 2].annotate(
                    "",
                    xy=(x + MB_SIZE//2 + dx, y + MB_SIZE//2 + dy),
                    xytext=(x + MB_SIZE//2,   y + MB_SIZE//2),
                    arrowprops=dict(arrowstyle="-|>", color='red',
                                   lw=1.0, mutation_scale=8)
                )
            idx += 1
    axes[0, 2].set_title("③ Vecteurs de mouvement", fontsize=11, pad=8)
    axes[0, 2].axis('off')

    # [1,0] Carte des residus
    im_res = axes[1, 0].imshow(residual_map, cmap='RdBu', vmin=-50, vmax=50)
    axes[1, 0].set_title("④ Carte des residus", fontsize=11, pad=8)
    axes[1, 0].axis('off')
    cb = fig.colorbar(im_res, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb.set_label("Amplitude residu", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # [1,1] Frame reconstruite
    axes[1, 1].imshow(reconstructed_Y, cmap='gray')
    axes[1, 1].set_title("⑤ Frame reconstruite (Y)", fontsize=11, pad=8)
    axes[1, 1].axis('off')

    # [1,2] Erreur de reconstruction
    diff    = np.abs(curr_Y.astype(np.int32) - reconstructed_Y.astype(np.int32))
    im_diff = axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=30)
    axes[1, 2].set_title("⑥ Erreur de reconstruction", fontsize=11, pad=8)
    axes[1, 2].axis('off')
    cb2 = fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cb2.set_label("Erreur absolue (niveaux)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    # Barre de metriques en bas
    mse_val  = np.mean(diff.astype(np.float64) ** 2)
    psnr_val = 10 * np.log10(255.0**2 / mse_val) if mse_val > 0 else float('inf')
    mvs_arr  = np.array(mvs)
    nonzero  = int(np.sum(np.any(mvs_arr != 0, axis=1)))
    mv_mean  = float(np.mean(np.sqrt(mvs_arr[:, 0]**2 + mvs_arr[:, 1]**2)))

    fig.text(
        0.5, -0.01,
        f"PSNR : {psnr_val:.2f} dB   |   "
        f"MV non-nuls : {nonzero}/{len(mvs)} ({100*nonzero/len(mvs):.0f}%)   |   "
        f"Amplitude moy. MV : {mv_mean:.2f} px   |   "
        f"Resolution : {w}x{h}",
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8', edgecolor='#bbb')
    )

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_path = "output/part3_pframe_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Visualisation sauvegardee -> {out_path}")
    plt.show()


def print_stats(encoded_p, curr_Y, reconstructed_Y):
    """Affiche les statistiques de la partie 3."""
    mse  = np.mean((curr_Y.astype(float) - reconstructed_Y.astype(float)) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float('inf')

    mvs     = np.array(encoded_p["motion_vectors"])
    nonzero = int(np.sum(np.any(mvs != 0, axis=1)))
    mv_mean = float(np.mean(np.sqrt(mvs[:, 0]**2 + mvs[:, 1]**2)))
    h, w    = encoded_p["shape"]
    S       = encoded_p["search_window"]
    n_comp  = (2 * S + 1) ** 2

    print("\n-- Statistiques Partie 3 — P-frame ----------------")
    print(f"  Resolution           : {w} x {h}")
    print(f"  Algorithme           : Full Search")
    print(f"  Fenetre de recherche : +-{S} px")
    print(f"  Comparaisons / MB    : {n_comp}  (= (2x{S}+1)^2)")
    print(f"  Total SAD calcules   : ~{len(encoded_p['motion_vectors']) * n_comp:,}")
    print(f"  Quality Factor (QF)  : {encoded_p['qf']}")
    print(f"  Macroblocs 16x16     : {len(encoded_p['motion_vectors'])}")
    print(f"  MV non-nuls          : {nonzero} ({100*nonzero/len(mvs):.1f}%)")
    print(f"  Amplitude moy. MV    : {mv_mean:.2f} px")
    print(f"  PSNR (Y)             : {psnr:.2f} dB")
    print("----------------------------------------------------\n")


# ══════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from part1_preprocessing import load_frames, preprocess_frame
    from iframe import encode_iframe, decode_iframe

    os.makedirs("output", exist_ok=True)

    # ── Chargement des frames ──────────────────────────────────
    frames = load_frames("frames")
    if len(frames) < 2:
        print("ERREUR : il faut au moins 2 frames dans le dossier 'frames/'")
        exit(1)

    preprocessed = [preprocess_frame(f) for f in frames]

    QF = 1.0
    S  = SEARCH_WIN   # 16 par defaut → 1089 comparaisons/MB

    # ── Frame 0 → I-frame (reference) ─────────────────────────
    print(f"[INFO] Encodage I-frame 0 (QF={QF}) ...")
    enc_i   = encode_iframe(preprocessed[0], qf=QF)
    ref_bgr = decode_iframe(enc_i)
    ref_Y   = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    print("[OK] I-frame 0 encodee et decodee")

    # ── Frame 1 → P-frame (Full Search S=16) ──────────────────
    curr_Y = preprocessed[1]["Y"]
    print(f"\n[INFO] Encodage P-frame 1 — Full Search (S={S}) ...")
    t0      = time.time()
    enc_p   = encode_p_frame(curr_Y, ref_Y, qf=QF, S=S)
    elapsed = time.time() - t0
    print(f"[OK] Encodage termine en {elapsed:.2f}s")

    print("[INFO] Decodage P-frame ...")
    rec_Y, res_map = decode_p_frame(enc_p, ref_Y)
    print("[OK] Decodage termine")

    print_stats(enc_p, curr_Y, rec_Y)
    visualize_pframe(frames[1], ref_Y, enc_p, rec_Y, res_map)

    # ── Encodage de toute la sequence avec GOP ─────────────────
    print(f"\n[INFO] Encodage sequence complete (GOP={GOP_SIZE}, S={S}) ...")
    all_encoded = []
    ref_Y_seq   = None
    t_total     = time.time()

    for idx, prep in enumerate(preprocessed):
        ftype = get_frame_type(idx, GOP_SIZE)
        if ftype == 'I':
            enc         = encode_iframe(prep, qf=QF)
            dec_bgr     = decode_iframe(enc)
            ref_Y_seq   = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            all_encoded.append(enc)
            print(f"  Frame {idx:03d} -> I-frame")
        else:
            t_frame     = time.time()
            enc         = encode_p_frame(prep["Y"], ref_Y_seq, qf=QF, S=S)
            rec_Y_seq, _ = decode_p_frame(enc, ref_Y_seq)
            ref_Y_seq   = rec_Y_seq
            all_encoded.append(enc)
            print(f"  Frame {idx:03d} -> P-frame  ({time.time()-t_frame:.2f}s)")

    i_count, p_count = count_frame_types(len(frames), GOP_SIZE)
    print(f"\n[OK] {len(all_encoded)} frames encodees : "
          f"{i_count} I-frames, {p_count} P-frames")
    print(f"[OK] Temps total : {time.time()-t_total:.1f}s")
    print("[OK] Pret pour la Partie 4 (codage entropique)")