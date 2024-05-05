import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
        label = torch.tensor(label - 1).long()
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

# for batch_idx, (samples, labels) in enumerate(chargeur_donnees):
#    print("Batch:", batch_idx)
#    print("Samples (features):", samples)
#    print("Labels (targets):", labels)
 #   print("Batch size:", len(labels))
  #  print()

class MonCNN(nn.Module):
    def __init__(self):
        super(MonCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1).double()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).double()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1).double()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 9, 128).double()  # Ajuster la taille en fonction de la sortie de la dernière couche de convolution
        self.fc2 = nn.Linear(128, 10).double()  # 10 classes pour la classification des activités

    def forward(self, x):
        x = x.unsqueeze(1)  # Ajouter une dimension pour le canal (batch_size, 1, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 9)  # Ajuster la taille en fonction de la sortie de la dernière couche de convolution
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1).double()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64, 128).double()  # Calculé pour correspondre à la taille de l'entrée
        self.fc2 = nn.Linear(128, 11).double()  # 11 classes pour la classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Ajouter une dimension pour le canal (batch_size, 1, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 39936)  # Ajuster la taille en fonction de la sortie de la dernière couche de convolution
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Créer une instance de votre modèle CNN
model = SimpleCNN()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for samples, labels in chargeur_donnees:
        outputs = model(samples)
        loss = criterion(outputs, labels)

        # Rétropropagation et mise à jour des poids
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Imprimer la perte à la fin de chaque époque
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

