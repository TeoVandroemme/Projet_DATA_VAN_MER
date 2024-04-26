import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Définir votre classe de dataset personnalisé
class DatasetPersonnalise(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, sep=';')  # Lire le fichier CSV dans un DataFrame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :-1]  # Exclure la colonne de la cible
        label = self.data.iloc[idx, -1]  # Récupérer la cible
        # Convertissez l'échantillon et la cible en tenseurs PyTorch si nécessaire
        sample = torch.tensor(sample.values.astype(float))
        label = torch.tensor(label).long()  # Supposant que votre cible est de type entier (long)
        return sample, label

# Chemin vers votre dataset CSV personnalisé
fichier_csv = "new_data/labelled_dataset.csv"

# Créez une instance de votre dataset personnalisé
dataset_personnalise = DatasetPersonnalise(fichier_csv)

# Exemple d'utilisation du dataset personnalisé
# Ici, nous allons parcourir le dataset en utilisant un DataLoader
# et imprimer les premiers échantillons
taille_lot = 32
chargeur_donnees = DataLoader(dataset=dataset_personnalise, batch_size=taille_lot, shuffle=True)

# Définir votre modèle
class MonModele(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MonModele, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Convertir les données en simple précision
        x = x.float()

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Paramètres du modèle
input_size = len(dataset_personnalise[0][0])  # Taille de l'entrée du modèle (nombre de fonctionnalités)
hidden_size = 128  # Taille de la couche cachée
num_classes = 2  # Nombre de classes de sortie

# Créer une instance de votre modèle
model = MonModele(input_size, hidden_size, num_classes)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
num_epochs = 10  # Nombre d'époques d'entraînement
for epoch in range(num_epochs):
    for data, labels in chargeur_donnees:
        # Passage avant
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Rétropropagation et mise à jour des poids
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Imprimer la perte à la fin de chaque époque
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
