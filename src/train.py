"""
Projet 2 — Deep Learning : construire un CNN de zéro avec PyTorch.
Objectif : classifier des images de chiffres écrits à la main (0-9).

Contrairement au projet 1 (régression logistique, une seule couche de calcul),
ici le modèle a plusieurs couches empilées qui apprennent des représentations
de plus en plus abstraites de l'image (bords -> formes -> chiffres).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# 1. LES DONNEES
# load_digits : 1797 images de chiffres, 8x8 pixels en niveaux de gris.
# (Version compacte du célèbre dataset MNIST, mais intégrée à scikit-learn.)
digits = load_digits()
X = digits.images  # shape (1797, 8, 8) -> chaque image est une grille 8x8
y = digits.target  # le vrai chiffre (0-9)

print(f"Dataset : {X.shape[0]} images de {X.shape[1]}x{X.shape[2]} pixels")
print(f"Classes : {np.unique(y)}\n")

# 2. PREPROCESSING
# Un CNN attend un tenseur de forme (batch, channels, hauteur, largeur).
# Nos images sont en niveaux de gris -> 1 seul channel.
X = X.reshape(-1, 1, 8, 8).astype(np.float32)
X = X / 16.0  # normalisation : pixels entre 0 et 16 -> entre 0 et 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Conversion en tenseurs PyTorch
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# 3. LE MODELE (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d détecte des motifs locaux (bords, courbes) dans l'image
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # réduit la taille, garde l'essentiel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Couches denses (fully connected) : combinent les motifs détectés pour classifier
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 sorties = 10 chiffres possibles
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 8x8 -> 4x4
        x = self.pool(self.relu(self.conv2(x)))  # 4x4 -> 2x2
        x = x.view(x.size(0), -1)  # aplatir pour les couches denses
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # adaptée à la classification multi-classes
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. ENTRAINEMENT
# Contrairement au projet 1 (une seule étape .fit()), ici on entraîne
# sur plusieurs "epochs" (passages complets sur les données).
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()          # remet les gradients à zéro
        outputs = model(batch_X)       # forward pass : prédiction
        loss = criterion(outputs, batch_y)  # calcule l'erreur
        loss.backward()                # backpropagation : calcule les gradients
        optimizer.step()               # met à jour les poids du modèle
        total_loss += loss.item()

    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/len(train_loader):.4f}")

# 5. EVALUATION
model.eval()
with torch.no_grad():  # pas besoin de calculer les gradients pour évaluer
    outputs = model(X_test_t)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test_t).float().mean().item()

print(f"\nAccuracy sur le test : {accuracy:.3f}")

# Exemple concret
print(f"\nExemple : vrai={y_test_t[0].item()}, prédit={predictions[0].item()}")
