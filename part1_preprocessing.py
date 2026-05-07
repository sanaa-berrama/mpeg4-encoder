

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# ─── PARAMÈTRES ───────────────────────────────
FRAMES_DIR = "frames"
# ──────────────────────────────────────────────


def load_frames(folder):
    """Charge toutes les images PNG/JPG d'un dossier."""
    exts = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            frames.append(img)
    print(f"[OK] {len(frames)} frame(s) chargée(s)")
    return frames


def bgr_to_ycbcr(frame_bgr):
    """
    Convertit une frame BGR en YCbCr.
    Retourne les 3 canaux séparés : Y, Cb, Cr
    """
    ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)  # OpenCV: ordre YCrCb
    return Y, Cb, Cr


def subsample_420(Cb, Cr):
    """
    Sous-échantillonnage 4:2:0 :
    On garde 1 pixel sur 2 horizontalement ET verticalement.
    Divise la taille de Cb et Cr par 4.
    """
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]
    return Cb_sub, Cr_sub


def upsample_420(Cb_sub, Cr_sub, target_shape):
    """
    Ré-échantillonnage : remet Cb et Cr à la taille originale.
    Nécessaire pour reconstruire l'image.
    """
    h, w = target_shape
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    return Cb_up, Cr_up


def ycbcr_to_bgr(Y, Cb, Cr):
    """
    Reconstruit une frame BGR depuis Y, Cb, Cr.
    """
    ycbcr = cv2.merge([Y, Cr, Cb])  # OpenCV attend YCrCb
    bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    return bgr


def preprocess_frame(frame_bgr):
    """
    Pipeline complet pour une frame :
    1. Conversion BGR → YCbCr
    2. Sous-échantillonnage 4:2:0 de Cb et Cr
    Retourne un dict avec toutes les données.
    """
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    Cb_sub, Cr_sub = subsample_420(Cb, Cr)

    return {
        "Y":      Y,        # canal luminance (taille originale)
        "Cb":     Cb,       # canal bleu original
        "Cr":     Cr,       # canal rouge original
        "Cb_sub": Cb_sub,   # Cb sous-échantillonné (1/4 taille)
        "Cr_sub": Cr_sub,   # Cr sous-échantillonné (1/4 taille)
        "shape":  frame_bgr.shape[:2]  # (hauteur, largeur)
    }


def reconstruct_frame(data):
    """
    Reconstruit une frame BGR depuis les données compressées.
    """
    h, w = data["shape"]
    Cb_up, Cr_up = upsample_420(data["Cb_sub"], data["Cr_sub"], (h, w))
    return ycbcr_to_bgr(data["Y"], Cb_up, Cr_up)


def visualize(frame_bgr, data):
    """
    Affiche une figure matplotlib avec :
    - Image originale
    - Canal Y (luminance)
    - Canal Cb (chrominance bleue)
    - Canal Cr (chrominance rouge)
    - Image reconstruite
    - Différence (erreur)
    """
    reconstructed = reconstruct_frame(data)
    diff = cv2.absdiff(frame_bgr, reconstructed)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Partie 1 — Pré-traitement YCbCr", fontsize=14, fontweight='bold')

    # Ligne 1
    axes[0, 0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Frame originale (BGR)")

    axes[0, 1].imshow(data["Y"], cmap='gray')
    axes[0, 1].set_title(f"Canal Y — Luminance\n{data['Y'].shape}")

    axes[0, 2].imshow(data["Cb"], cmap='Blues_r')
    axes[0, 2].set_title(f"Canal Cb — Chrominance bleue\n{data['Cb'].shape}")

    # Ligne 2
    axes[1, 0].imshow(data["Cr"], cmap='Reds_r')
    axes[1, 0].set_title(f"Canal Cr — Chrominance rouge\n{data['Cr'].shape}")

    axes[1, 1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Frame reconstruite")

    axes[1, 2].imshow(diff)
    axes[1, 2].set_title("Différence (erreur de reconstruction)")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("output/part1_visualisation.png", dpi=150, bbox_inches='tight')
    print("[OK] Visualisation sauvegardée dans output/part1_visualisation.png")
    plt.show()


def print_stats(frame_bgr, data):
    """Affiche les statistiques de compression de la partie 1."""
    h, w = data["shape"]
    original_size = h * w * 3  # 3 canaux BGR, 1 octet chacun

    # Taille après conversion YCbCr + sous-échantillonnage
    Y_size  = data["Y"].size           # h * w
    Cb_size = data["Cb_sub"].size      # (h/2) * (w/2)
    Cr_size = data["Cr_sub"].size      # (h/2) * (w/2)
    compressed_size = Y_size + Cb_size + Cr_size

    ratio = original_size / compressed_size

    print("\n── Statistiques Partie 1 ──────────────────")
    print(f"  Résolution          : {w} x {h}")
    print(f"  Taille originale    : {original_size:,} octets")
    print(f"  Taille Y            : {Y_size:,} octets  ({w}x{h})")
    print(f"  Taille Cb réduit    : {Cb_size:,} octets  ({w//2}x{h//2})")
    print(f"  Taille Cr réduit    : {Cr_size:,} octets  ({w//2}x{h//2})")
    print(f"  Taille compressée   : {compressed_size:,} octets")
    print(f"  Ratio compression   : {ratio:.2f}x")
    print("────────────────────────────────────────────\n")


# ─── PROGRAMME PRINCIPAL ──────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # 1. Charger les frames
    frames = load_frames(FRAMES_DIR)

    # 2. Traiter la première frame (pour la démo)
    frame = frames[0]
    data  = preprocess_frame(frame)

    # 3. Afficher les stats
    print_stats(frame, data)

    # 4. Visualiser
    visualize(frame, data)

    # 5. Traiter TOUTES les frames (pour le pipeline complet)
    all_preprocessed = [preprocess_frame(f) for f in frames]
    print(f"[OK] {len(all_preprocessed)} frame(s) pré-traitée(s) — prêtes pour la Partie 2")