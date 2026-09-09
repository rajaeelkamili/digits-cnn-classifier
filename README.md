# Digits CNN classifier

Un CNN construit de zéro avec PyTorch pour classifier des images de chiffres écrits à la main (0-9).

## Objectif

Comprendre les couches convolutionnelles, l'entraînement par epochs, et la backpropagation.

## Données

[`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/datasets/toy_dataset.html) — 1797 images 8x8 en niveaux de gris (version compacte de MNIST).

## Architecture

Un CNN simple à 2 couches convolutionnelles (16 puis 32 filtres) suivies de 2 couches denses (fully connected) pour la classification finale sur 10 classes.

## Approche

- Normalisation des pixels (0-16 → 0-1)
- Split 80/20 stratifié
- Entraînement sur 15 epochs avec `Adam` optimizer et `CrossEntropyLoss`
- Backpropagation : `loss.backward()` + `optimizer.step()` à chaque batch

## Résultats

| Métrique | Score |
|----------|-------|
| Accuracy (test) | 0.969 |
| Loss finale (train) | ~0.087 |

## Lancer le projet

```bash
pip install torch scikit-learn
cd src
python train.py
```

## Prochaines améliorations possibles

- Tester sur le vrai MNIST (28x28, 60000 images) pour voir l'effet de l'échelle
- Ajouter du dropout pour limiter l'overfitting
- Visualiser les filtres appris par la première couche convolutionnelle
