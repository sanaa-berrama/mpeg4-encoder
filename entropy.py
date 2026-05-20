"""
entropy.py
Partie 4 — Codage entropique
Serialisation + compression zlib → fichier .bin
Decodeur correspondant
"""

import pickle
import zlib
import struct
import os
import time


# ══════════════════════════════════════════════════════════════
#  ENCODEUR — ecriture du fichier .bin
# ══════════════════════════════════════════════════════════════

def encode_to_bin(video_data, output_path, zlib_level=9):
    """
    Compresse et sauvegarde toutes les donnees video dans un fichier .bin.

    Etapes :
      1. Serialisation avec pickle  → bytes bruts
      2. Compression avec zlib      → bytes comprimes (algo DEFLATE = Huffman + LZ77)
      3. Ecriture dans le .bin      → en-tete 4 octets + donnees comprimees

    Parametres :
        video_data  : dict contenant toutes les frames encodees
                      {
                        'width'  : int,
                        'height' : int,
                        'fps'    : int,
                        'gop'    : int,
                        'frames' : [ {...}, {...}, ... ]
                      }
        output_path : chemin du fichier .bin de sortie
        zlib_level  : niveau de compression zlib (1=rapide, 9=max)

    Retourne : (taille_originale, taille_comprimee) en octets
    """
    print(f"[INFO] Serialisation des donnees ...")
    raw_bytes  = pickle.dumps(video_data)
    print(f"  Donnees serialisees : {len(raw_bytes):,} octets")

    print(f"[INFO] Compression zlib (niveau {zlib_level}) ...")
    t0         = time.time()
    compressed = zlib.compress(raw_bytes, level=zlib_level)
    elapsed    = time.time() - t0
    print(f"  Donnees comprimees  : {len(compressed):,} octets  ({elapsed:.2f}s)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        # En-tete : taille originale sur 4 octets (big-endian unsigned int)
        f.write(struct.pack('>I', len(raw_bytes)))
        f.write(compressed)

    ratio = len(raw_bytes) / len(compressed)
    print(f"[OK] Fichier .bin ecrit : {output_path}")
    print(f"     Taille originale   : {len(raw_bytes):,} octets")
    print(f"     Taille comprimee   : {len(compressed):,} octets")
    print(f"     Ratio zlib         : {ratio:.2f}x")

    return len(raw_bytes), len(compressed)


# ══════════════════════════════════════════════════════════════
#  DECODEUR — lecture du fichier .bin
# ══════════════════════════════════════════════════════════════

def decode_from_bin(input_path):
    """
    Lit et decompresse un fichier .bin genere par encode_to_bin().

    Etapes :
      1. Lecture de l'en-tete (taille originale attendue)
      2. Decompression zlib
      3. Verification de la taille (detection de corruption)
      4. Deserialisation pickle → video_data

    Retourne : video_data (meme structure que ce qui a ete encode)
    """
    print(f"[INFO] Lecture du fichier {input_path} ...")
    with open(input_path, 'rb') as f:
        original_size = struct.unpack('>I', f.read(4))[0]
        compressed    = f.read()

    print(f"  Taille fichier .bin : {os.path.getsize(input_path):,} octets")
    print(f"  Taille attendue     : {original_size:,} octets")

    print(f"[INFO] Decompression zlib ...")
    t0        = time.time()
    raw_bytes = zlib.decompress(compressed)
    elapsed   = time.time() - t0

    assert len(raw_bytes) == original_size, \
        f"ERREUR : fichier corrompu ! ({len(raw_bytes)} != {original_size})"

    print(f"[OK] Decompression reussie en {elapsed:.2f}s")

    video_data = pickle.loads(raw_bytes)
    print(f"[OK] {len(video_data['frames'])} frames chargees depuis {input_path}")
    return video_data


# ══════════════════════════════════════════════════════════════
#  HELPER — construire video_data depuis les frames encodees
# ══════════════════════════════════════════════════════════════

def build_video_data(all_encoded, original_frames, fps=25, gop=10):
    """
    Construit le dict video_data a partir de la liste des frames encodees.

    Parametres :
        all_encoded     : liste de dicts retournes par encode_iframe / encode_p_frame
        original_frames : liste des frames BGR originales (pour recuperer w/h/fps)
        fps             : frequence d'images de la video
        gop             : taille du GOP utilise

    Retourne : dict video_data pret pour encode_to_bin()
    """
    h, w = original_frames[0].shape[:2]
    return {
        'width':  w,
        'height': h,
        'fps':    fps,
        'gop':    gop,
        'frames': all_encoded,
    }


# ══════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    import numpy as np
    from part1_preprocessing import load_frames, preprocess_frame
    from iframe import encode_iframe, decode_iframe
    from pframe import (encode_p_frame, decode_p_frame,
                        get_frame_type, count_frame_types,
                        GOP_SIZE, SEARCH_WIN)

    os.makedirs("output", exist_ok=True)
    BIN_PATH = "output/video.bin"
    QF       = 1.0

    # ── 1. Charger les frames ──────────────────────────────────
    frames       = load_frames("frames")
    preprocessed = [preprocess_frame(f) for f in frames]
    print(f"[INFO] {len(frames)} frames chargees")

    # ── 2. Encoder toute la sequence (I + P frames) ────────────
    print(f"\n[INFO] Encodage sequence (GOP={GOP_SIZE}, S={SEARCH_WIN}, QF={QF}) ...")
    all_encoded = []
    ref_Y_seq   = None
    t_total     = time.time()

    for idx, prep in enumerate(preprocessed):
        ftype = get_frame_type(idx, GOP_SIZE)
        if ftype == 'I':
            enc       = encode_iframe(prep, qf=QF)
            dec_bgr   = decode_iframe(enc)
            ref_Y_seq = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            all_encoded.append(enc)
            print(f"  Frame {idx:03d} -> I-frame")
        else:
            t0  = time.time()
            enc = encode_p_frame(prep["Y"], ref_Y_seq, qf=QF, S=SEARCH_WIN)
            rec_Y_seq, _ = decode_p_frame(enc, ref_Y_seq)
            ref_Y_seq    = rec_Y_seq
            all_encoded.append(enc)
            print(f"  Frame {idx:03d} -> P-frame  ({time.time()-t0:.1f}s)")

    i_count, p_count = count_frame_types(len(frames), GOP_SIZE)
    print(f"\n[OK] Encodage termine en {time.time()-t_total:.1f}s")
    print(f"     {i_count} I-frames  |  {p_count} P-frames")

    # ── 3. Codage entropique → fichier .bin ────────────────────
    print(f"\n[INFO] Codage entropique -> {BIN_PATH}")
    video_data = build_video_data(all_encoded, frames, fps=25, gop=GOP_SIZE)
    original_size, compressed_size = encode_to_bin(video_data, BIN_PATH)

    # Taux de compression global (pixels bruts vs .bin)
    raw_pixels_bytes = frames[0].shape[0] * frames[0].shape[1] * 3 * len(frames)
    ratio_global = raw_pixels_bytes / compressed_size
    print(f"\n[OK] Taux de compression global : {ratio_global:.2f}x")
    print(f"     (pixels bruts : {raw_pixels_bytes:,} octets)")

    # ── 4. Verification : decoder depuis le .bin ──────────────
    print(f"\n[INFO] Verification du decodage ...")
    video_data_decoded = decode_from_bin(BIN_PATH)
    print(f"[OK] Verification reussie — {len(video_data_decoded['frames'])} frames decodees")
    print(f"[OK] Partie 4 terminee — pret pour la Partie 5 (evaluation)")
