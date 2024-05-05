import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
class DatasetPersonnalise(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, sep=';')  # Lire le fichier CSV dans un DataFrame
        # Diviser les données en ensembles d'entraînement et de test
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.2, random_state=42)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :-1]  # Exclure la colonne de la cible
        label = self.data.iloc[idx, -1]  # Récupérer la cible
        # Convertissez l'échantillon et la cible en tenseurs PyTorch si nécessaire
        sample = torch.tensor(sample.values.astype(float))
        label = torch.tensor(label - 1).long()
        return sample, label

    def set_train(self, train=True):
        self.train = train
# Chemin vers votre dataset CSV personnalisé
fichier_csv = "new_data/labelled_dataset.csv"

# Créez une instance de votre dataset personnalisé
dataset_personnalise = DatasetPersonnalise(fichier_csv)

# Exemple d'utilisation du dataset personnalisé
# Ici, nous allons parcourir le dataset en utilisant un DataLoader
# et imprimer les premiers échantillons

# Définir l'ensemble de données d'entraînement
dataset_personnalise.set_train(True)
taille_lot = 64
chargeur_donnees_train = DataLoader(dataset=dataset_personnalise, batch_size=taille_lot, shuffle=True)
dataset_personnalise.set_train(False)
# Exemple d'utilisation du dataset personnalisé pour les données de test
chargeur_donnees_test = DataLoader(dataset=dataset_personnalise, batch_size=taille_lot, shuffle=True)

chargeur_donnees = DataLoader(dataset=dataset_personnalise, batch_size=taille_lot, shuffle=True)

# Calculer le nombre d'itérations pour l'ensemble d'entraînement
nb_iterations_train = math.ceil(len(dataset_personnalise.data_train) / taille_lot)
# Calculer le nombre d'itérations pour l'ensemble de test
nb_iterations_test = math.ceil(len(dataset_personnalise.data_test) / taille_lot)

for batch_idx, (samples, labels) in enumerate(chargeur_donnees_test):
    print("Batch:", batch_idx)
    print("Samples (features):", samples)
    print("Labels (targets):", labels)
    print("Batch size:", len(labels))
    print()
    if batch_idx + 1 == nb_iterations_test:
        break
class MonCNN(nn.Module):
    def __init__(self):
        super(MonCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1).double()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).double()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1).double()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 9, 128).double()
        self.fc2 = nn.Linear(128, 10).double()  # 10 classes pour la classification des activités

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1).double()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 64, 128).double()  
        self.fc2 = nn.Linear(128, 11).double()  # 11 classes pour la classification

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 39936)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Créer une instance de votre modèle CNN
model1 = SimpleCNN()
model = MonCNN()

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100


model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(chargeur_donnees_train, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Afficher la perte moyenne pour l'ensemble de l'époque
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(chargeur_donnees_train)))
    running_loss = 0.0

print('Finished Training')

# Test du modèle
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for data in chargeur_donnees_test:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test datas: %d %%' % (
    100 * correct / total))



