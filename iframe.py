
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

from part1_preprocessing import load_frames, preprocess_frame

Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

 
def split_into_blocks(channel, block_size=8):
    """
    Découpe un canal (Y, Cb ou Cr) en blocs de block_size × block_size.
    Ajoute du padding (valeur 128) si la taille n'est pas un multiple de block_size.
 
    Retourne :
        blocks      : liste de blocs numpy (block_size, block_size)
        padded_shape: (h_padded, w_padded) — nécessaire pour la reconstruction
    """
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
 
    padded = np.pad(
        channel,
        ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=128
    )
 
    blocks = []
    for i in range(0, padded.shape[0], block_size):
        for j in range(0, padded.shape[1], block_size):
            blocks.append(padded[i:i + block_size, j:j + block_size])
 
    return blocks, padded.shape
 
 
def reconstruct_from_blocks(blocks, padded_shape, original_shape, block_size=8):
    """
    Reconstruit un canal depuis une liste de blocs.
    Retire le padding pour revenir à la taille originale.
    """
    h_pad, w_pad = padded_shape
    canvas = np.zeros((h_pad, w_pad), dtype=np.float32)
 
    idx = 0
    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            canvas[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1

    h, w = original_shape
    return np.clip(canvas[:h, :w], 0, 255).astype(np.uint8)
 
def dct2d(block):
    """
    Applique la DCT 2D sur un bloc 8×8.
    On centre d'abord les valeurs autour de 0 (soustrait 128).
    """
    block_centered = block.astype(np.float32) - 128.0
    return dctn(block_centered, norm='ortho')
 
 
def idct2d(dct_block):
    """
    Applique l'IDCT 2D (inverse DCT).
    Remet les valeurs dans [0, 255].
    """
    reconstructed = idctn(dct_block, norm='ortho')
    return np.clip(reconstructed + 128.0, 0, 255).astype(np.uint8)
 

def quantize(dct_block, qf=1.0):
    """
    Quantifie les coefficients DCT.
    qf (Quality Factor) : 
        qf > 1 → plus de perte, meilleure compression
        qf < 1 → moins de perte, meilleure qualité
    """
    return np.round(dct_block / (Q_MATRIX * qf)).astype(np.int16)
 
 
def dequantize(q_block, qf=1.0):
    """
    Dé-quantifie les coefficients (multiplie par la matrice Q * qf).
    Opération inverse de quantize — utilisée au décodeur.
    """
    return (q_block * (Q_MATRIX * qf)).astype(np.float32)
 

def encode_channel(channel, qf=1.0, block_size=8):
    """
    Encode un canal (Y, Cb ou Cr) complet :
      split → DCT → quantize sur chaque bloc.
 
    Retourne :
        q_blocks     : liste de blocs quantifiés (int16)
        padded_shape : forme padded (pour reconstruction)
    """
    blocks, padded_shape = split_into_blocks(channel, block_size)
    q_blocks = []
    for block in blocks:
        dct_block = dct2d(block)
        q_block = quantize(dct_block, qf)
        q_blocks.append(q_block)
    return q_blocks, padded_shape
 
 
def encode_iframe(preprocessed, qf=1.0):
    """
    Encode une I-frame complète (Y + Cb_sub + Cr_sub).
 
    Paramètre :
        preprocessed : dict retourné par preprocess_frame()
        qf           : Quality Factor (défaut 1.0)
 
    Retourne un dict avec toutes les données encodées.
    """
    Y      = preprocessed["Y"]
    Cb_sub = preprocessed["Cb_sub"]
    Cr_sub = preprocessed["Cr_sub"]
    shape  = preprocessed["shape"]
 
    q_Y,  pad_Y  = encode_channel(Y,      qf)
    q_Cb, pad_Cb = encode_channel(Cb_sub, qf)
    q_Cr, pad_Cr = encode_channel(Cr_sub, qf)
 
    return {
        "type":     "I",
        "qf":       qf,
        "shape":    shape,     
        "Y":        q_Y,        
        "Cb":       q_Cb,
        "Cr":       q_Cr,
        "pad_Y":    pad_Y,           
        "pad_Cb":   pad_Cb,
        "pad_Cr":   pad_Cr,
        "Cb_shape": Cb_sub.shape,   
        "Cr_shape": Cr_sub.shape,
    }
 

 
def decode_channel(q_blocks, padded_shape, original_shape, qf=1.0, block_size=8):
    """
    Décode un canal :
      dequantize → IDCT → reconstruct sur chaque bloc.
    """
    recon_blocks = []
    for q_block in q_blocks:
        dct_block = dequantize(q_block, qf)
        block     = idct2d(dct_block)
        recon_blocks.append(block)
    return reconstruct_from_blocks(recon_blocks, padded_shape, original_shape, block_size)
 
 
def decode_iframe(encoded):
    """
    Décode une I-frame encodée et retourne l'image BGR reconstruite.
    """
    qf    = encoded["qf"]
    h, w  = encoded["shape"]

    Y_rec  = decode_channel(encoded["Y"],  encoded["pad_Y"],  (h, w),                    qf)
    Cb_rec = decode_channel(encoded["Cb"], encoded["pad_Cb"], encoded["Cb_shape"],        qf)
    Cr_rec = decode_channel(encoded["Cr"], encoded["pad_Cr"], encoded["Cr_shape"],        qf)

    Cb_up = cv2.resize(Cb_rec, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_rec, (w, h), interpolation=cv2.INTER_LINEAR)
 
    ycbcr = cv2.merge([Y_rec, Cr_up, Cb_up])  
    bgr   = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    return bgr
 
 
def compute_psnr(original, reconstructed):
    """Calcule le PSNR entre l'image originale et reconstruite (en dB)."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)
 
 
def compute_compression_ratio(q_blocks_Y, q_blocks_Cb, q_blocks_Cr, original_shape):
    """Estime le ratio de compression basé sur le nombre de zéros (coefficients nuls)."""
    h, w = original_shape
    original_bytes = h * w * 3
 
    total_coeffs    = sum(b.size for b in q_blocks_Y + q_blocks_Cb + q_blocks_Cr)
    nonzero_coeffs  = sum(np.count_nonzero(b) for b in q_blocks_Y + q_blocks_Cb + q_blocks_Cr)
    zero_ratio      = 1 - nonzero_coeffs / total_coeffs
 
    # Estimation : les zéros se compriment très bien
    estimated_bytes = max(1, int(total_coeffs * 2 * (1 - zero_ratio * 0.9)))
    ratio = original_bytes / estimated_bytes
    return ratio, zero_ratio * 100
 
 
def visualize_iframe(original_frame, encoded, reconstructed_frame):
    """
    Affiche une figure matplotlib montrant les 4 étapes du pipeline I-frame :
      1. Bloc 8×8 original
      2. Coefficients DCT
      3. Coefficients quantifiés
      4. Bloc reconstruit
    + comparaison frame complète originale vs reconstruite.
    """
    h, w  = encoded["shape"]
    cy, cx = h // 2, w // 2
   
    by = (cy // 8) * 8
    bx = (cx // 8) * 8
 
    Y_channel  = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    raw_block  = Y_channel[by:by + 8, bx:bx + 8].astype(np.float32)
    dct_block  = dct2d(raw_block)
    q_block    = quantize(dct_block, encoded["qf"]).astype(np.float32)
    dq_block   = dequantize(q_block.astype(np.int16), encoded["qf"])
    recon_block = idct2d(dq_block).astype(np.float32)
 
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Partie 2 — I-frame DCT (QF={encoded['qf']})", fontsize=14, fontweight='bold')
 
   
    im0 = axes[0, 0].imshow(raw_block, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Bloc 8×8 original (pixels)")
    plt.colorbar(im0, ax=axes[0, 0])
 
    im1 = axes[0, 1].imshow(dct_block, cmap='seismic')
    axes[0, 1].set_title("Coefficients DCT")
    plt.colorbar(im1, ax=axes[0, 1])
 
    im2 = axes[0, 2].imshow(q_block, cmap='seismic')
    axes[0, 2].set_title("Après quantification")
    plt.colorbar(im2, ax=axes[0, 2])
 
    im3 = axes[0, 3].imshow(recon_block, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title("Bloc reconstruit (IDCT)")
    plt.colorbar(im3, ax=axes[0, 3])
 
    # Ligne 2 : comparaison frame complète
    axes[1, 0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Frame originale")
 
    axes[1, 1].imshow(cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Frame reconstruite")
 
    diff = cv2.absdiff(original_frame, reconstructed_frame)
    axes[1, 2].imshow(diff)
    axes[1, 2].set_title("Différence (erreur)")
 
    psnr = compute_psnr(original_frame, reconstructed_frame)
    ratio, zero_pct = compute_compression_ratio(
        encoded["Y"], encoded["Cb"], encoded["Cr"], encoded["shape"]
    )
    info = (
        f"PSNR     : {psnr:.2f} dB\n"
        f"Zéros    : {zero_pct:.1f}%\n"
        f"Ratio est.: {ratio:.2f}x\n"
        f"QF       : {encoded['qf']}"
    )
    axes[1, 3].text(0.1, 0.5, info, transform=axes[1, 3].transAxes,
                    fontsize=13, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 3].set_title("Métriques")
    axes[1, 3].axis('off')
 
    for ax in axes.flat:
        ax.axis('off')
 
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/part2_iframe_visualization.png", dpi=150, bbox_inches='tight')
    print("[OK] Visualisation sauvegardée dans output/part2_iframe_visualization.png")
    plt.show()
 
 
def print_stats(encoded, original_frame, reconstructed_frame):
    """Affiche les statistiques de la partie 2."""
    psnr = compute_psnr(original_frame, reconstructed_frame)
    ratio, zero_pct = compute_compression_ratio(
        encoded["Y"], encoded["Cb"], encoded["Cr"], encoded["shape"]
    )
    h, w = encoded["shape"]
    nb_blocks_Y  = len(encoded["Y"])
    nb_blocks_Cb = len(encoded["Cb"])
 
    print("\n── Statistiques Partie 2 — I-frame ────────────")
    print(f"  Résolution           : {w} x {h}")
    print(f"  Quality Factor (QF)  : {encoded['qf']}")
    print(f"  Blocs 8×8 (Y)        : {nb_blocks_Y}")
    print(f"  Blocs 8×8 (Cb/Cr)    : {nb_blocks_Cb}")
    print(f"  Coefficients nuls    : {zero_pct:.1f}%")
    print(f"  Ratio compression    : ~{ratio:.2f}x")
    print(f"  PSNR                 : {psnr:.2f} dB")
    print("─────────────────────────────────────────────────\n")
 
 
 
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
 
   
    frames = load_frames("frames")
    preprocessed_frames = [preprocess_frame(f) for f in frames]
 
    
    QF = 1.0  
    print(f"[INFO] Encodage I-frame avec QF={QF} ...")
    encoded = encode_iframe(preprocessed_frames[0], qf=QF)
    print("[OK] Encodage terminé")
 
    print("[INFO] Décodage ...")
    reconstructed = decode_iframe(encoded)
    print("[OK] Décodage terminé")
 
    print_stats(encoded, frames[0], reconstructed)
 
    visualize_iframe(frames[0], encoded, reconstructed)
 
    print("[INFO] Encodage de toutes les frames ...")
    all_encoded = [encode_iframe(p, qf=QF) for p in preprocessed_frames]
    all_decoded = [decode_iframe(e) for e in all_encoded]
    print(f"[OK] {len(all_encoded)} I-frame(s) encodée(s) — prêtes pour la Partie 3")