# mpeg4-encoder

Mini projet multimédia — Encodeur vidéo simplifié type MPEG-4, réalisé dans le cadre du module Systèmes Multimédias.

**M1 IL — Groupe 1 — USTHB, 2026**

---

## Ce que fait le projet

On prend une séquence d'images (frames extraites d'une vidéo), on les compresse en passant par les mêmes grandes étapes qu'un vrai codec vidéo, et on obtient un fichier `.bin` qu'on peut redécoder pour retrouver les images.

Le pipeline complet :

1. **Prétraitement** — conversion BGR → YCbCr + sous-échantillonnage chrominance 4:2:0
2. **I-frames** — compression spatiale par DCT 8×8 + quantification
3. **P-frames** — estimation de mouvement (Full Search) + codage des résidus DCT
4. **Codage entropique** — sérialisation pickle + compression zlib → fichier `.bin`
5. **Évaluation** — PSNR, taux de compression, visualisation du pipeline complet

---

## Résultats obtenus

Testé sur 50 frames (640×360) :

- Taux de compression : **49.26×**
- PSNR moyen : **19.78 dB**
- Taille compressée : 701 532 octets (contre 34 560 000 octets bruts)

---

## Bonus

Comparaison de trois algorithmes d'estimation de mouvement :

| Algorithme | Temps | Gain | PSNR |
|---|---|---|---|
| Full Search | 13.54s | 1× | 22.11 dB |
| Three-Step Search | 0.80s | 29.4× | 22.11 dB |
| Diamond Search | 1.61s | 41.8× | 22.11 dB |

---

## Lancer le projet

```bash
# 0. Extraire les frames
python extract_frames.py

# 1 à 4. Encoder (prétraitement + I/P-frames + entropie)
python entropy.py

# 5. Évaluer + visualiser
python evaluate.py

# Bonus
python bonus.py
```

## Dépendances

```bash
pip install opencv-python numpy scipy matplotlib
```

---

## Fichiers

```
├── frames/                  # frames d'entrée
├── output/
│   └── video.bin            # fichier compressé
├── extract_frames.py
├── part1_preprocessing.py
├── iframe.py
├── pframe.py
├── entropy.py
├── evaluate.py
└── bonus.py
```
