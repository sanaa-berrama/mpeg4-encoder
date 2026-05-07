"""
extract_frames.py
Extrait les frames d'une vidéo et les sauvegarde dans /frames/
"""

import cv2
import os

# ─── PARAMÈTRES ───────────────────────────────
VIDEO_PATH  = "video_test.mp4"   # ton fichier vidéo
OUTPUT_DIR  = "frames"           # dossier de sortie
MAX_FRAMES  = 50                 # nombre max de frames à extraire
# ──────────────────────────────────────────────

def extract_frames(video_path, output_dir, max_frames=50):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERREUR : impossible d'ouvrir '{video_path}'")
        return

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Vidéo : {total} frames | {fps:.1f} FPS | {width}x{height}")
    print(f"Extraction de {min(max_frames, total)} frames...")

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        filename = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)
        count += 1

    cap.release()
    print(f"[OK] {count} frames sauvegardées dans '{output_dir}/'")

if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, MAX_FRAMES)