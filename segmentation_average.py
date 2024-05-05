import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Charger le fichier merged_data.csv
base = pd.read_csv('new_data/merged_data.csv', sep=',')

# Convertir la colonne 'Time' en datetime et ajuster le fuseau horaire à 'UTC+01:00'
base['Time'] = pd.to_datetime(base['Time']).dt.tz_convert('UTC+01:00')

# Vérifier les noms et les formats des colonnes
print(base.dtypes)

# Charger le fichier 'activites.xlsx' et extraire la feuille 'Done so far'
activity_df = pd.read_excel('fichiers_data/activites.xlsx', sheet_name='Done so far')

# Convertir les colonnes 'Started' et 'Ended' en datetime et ajuster le fuseau horaire à 'UTC+01:00'
activity_df['Started'] = pd.to_datetime(activity_df['Started']).dt.tz_localize('UTC').dt.tz_convert('UTC+01:00')
activity_df['Ended'] = pd.to_datetime(activity_df['Ended']).dt.tz_localize('UTC').dt.tz_convert('UTC+01:00')

# Supprimer les colonnes non nécessaires et les valeurs NaN
new_activity_df = activity_df.drop(
    columns=['Comments', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7']).dropna()

# Supprimer les lignes contenant des valeurs NaN
new_activity_df = new_activity_df.dropna(ignore_index=True)

# Vérifier à nouveau les types de données dans le DataFrame
print("Types de données dans le dataframe des activités après nettoyage :")
print(new_activity_df.dtypes)

# Vérifier les dimensions du DataFrame après nettoyage
print("\nDimensions du dataframe des activités après nettoyage :")
print(new_activity_df.shape)

# Créer des listes pour stocker les segments de chaque activité
Saber_grouped = []
Aera_grouped = []
Nett_grouped = []
Asp_grouped = []
AS1_grouped = []
Bougie_grouped = []
SdB_grouped = []
BricoP_grouped = []
BricoC_grouped = []
Oeuf_grouped = []

# Créer des variables pour suivre la longueur de chaque activité
len_Saber = 0
len_Aera = 0
len_Nett = 0
len_Asp = 0
len_AS1 = 0
len_Bougie = 0
len_SdB = 0
len_BricoP = 0
len_BricoC = 0
len_Oeuf = 0

# Création du jeu de données étiqueté
labelled_dataset = pd.DataFrame()  # Initialiser un dataframe pour stocker le jeu de données étiqueté
# Parcourir les activités dans le calendrier
for idact, act in enumerate(new_activity_df['activity']):
    start = new_activity_df['Started'][idact]
    end = new_activity_df['Ended'][idact]
    # Segmenter la base de données en fonction de chaque activité
    if act == 'Saber':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 6
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Saber += len(segment)
        Saber_grouped.append(segment)
    elif act == 'Aera':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 8
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Aera += len(segment)
        Aera_grouped.append(segment)
    elif act == 'Nett':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 5
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Nett += len(segment)
        Nett_grouped.append(segment)
    elif act == 'Asp':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 4
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Asp += len(segment)
        Asp_grouped.append(segment)
    elif act == 'AS1':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 1
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_AS1 += len(segment)
        AS1_grouped.append(segment)
    elif act == 'Bougie':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 7
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Bougie += len(segment)
        Bougie_grouped.append(segment)
    elif act == 'SdB':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 3
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_SdB += len(segment)
        SdB_grouped.append(segment)
    elif act == 'BricoP':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 9
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_BricoP += len(segment)
        BricoP_grouped.append(segment)
    elif act == 'BricoC':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 10
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_BricoC += len(segment)
        BricoC_grouped.append(segment)
    elif act == 'Oeuf':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(
            by='Time').drop(columns='Time')
        segment['label'] = 2
        labelled_dataset = pd.concat([labelled_dataset, segment], ignore_index=True)
        len_Oeuf += len(segment)
        Oeuf_grouped.append(segment)
    # Ajoutez des clauses elif pour chaque activité supplémentaire

# Enregistrer le jeu de données étiqueté dans un fichier CSV
labelled_dataset.to_csv('new_data/labelled_dataset.csv', index=False, sep=';')


# Vérifier la longueur de chaque activité
print("Longueur de Saber :", len_Saber)
print("Longueur de Aera :", len_Aera)
print("Longueur de Nett :", len_Nett)
print("Longueur de Asp :", len_Asp)
print("Longueur de AS1 :", len_AS1)
print("Longueur de Bougie :", len_Bougie)
print("Longueur de SdB :", len_SdB)
print("Longueur de BricoP :", len_BricoP)
print("Longueur de BricoC :", len_BricoC)
print("Longueur de Oeuf :", len_Oeuf)


def averageSignature(activity_instances, num_instances):
    min_length = min(
        len(instance) for instance in activity_instances)  # Trouver la longueur minimale parmi toutes les instances

    avg_signature = []  # Initialiser la signature moyenne

    # Parcourir chaque ligne jusqu'à la longueur minimale
    for i in range(min_length):
        row_sum = np.zeros(activity_instances[0].shape[1])  # Initialiser la somme des valeurs de chaque colonne

        instance_count = 0  # Initialiser le nombre d'instances comptées pour cette ligne

        # Parcourir chaque instance d'activité
        for instance in activity_instances:
            row_sum += instance.iloc[i].values  # Ajouter les valeurs de la ligne à la somme
            instance_count += 1  # Incrémenter le nombre d'instances comptées

        # Calculer la moyenne des valeurs de chaque colonne pour cette ligne
        avg_row = row_sum / instance_count
        avg_signature.append(avg_row)  # Ajouter la moyenne au tableau de signature moyenne

    return pd.DataFrame(avg_signature, columns=activity_instances[0].columns)


# Utilisation de la fonction pour calculer la signature moyenne de chaque activité
avg_Saber = averageSignature(Saber_grouped, len(Saber_grouped))
avg_Aera = averageSignature(Aera_grouped, len(Aera_grouped))
avg_Nett = averageSignature(Nett_grouped, len(Nett_grouped))
avg_Asp = averageSignature(Asp_grouped, len(Asp_grouped))
avg_AS1 = averageSignature(AS1_grouped, len(AS1_grouped))
avg_Bougie = averageSignature(Bougie_grouped, len(Bougie_grouped))
avg_SdB = averageSignature(SdB_grouped, len(SdB_grouped))
avg_BricoP = averageSignature(BricoP_grouped, len(BricoP_grouped))
avg_BricoC = averageSignature(BricoC_grouped, len(BricoC_grouped))
avg_Oeuf = averageSignature(Oeuf_grouped, len(Oeuf_grouped))

import matplotlib.pyplot as plt


def plotSignature(activity_name, avg_signature):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_signature, marker='o', linestyle='-')
    plt.title("Average response of '{}'".format(activity_name))
    plt.xlabel("Samples")
    plt.ylabel("Sensors' measurements")
    plt.grid(True)
    plt.show()


# Utilisation de la fonction pour tracer la signature moyenne de chaque activité
plotSignature("Saber", avg_Saber)
plotSignature("Aera", avg_Aera)
plotSignature("Nett", avg_Nett)
plotSignature("Asp", avg_Asp)
plotSignature("AS1", avg_AS1)
plotSignature("Bougie", avg_Bougie)
plotSignature("SdB", avg_SdB)
plotSignature("BricoP", avg_BricoP)
plotSignature("BricoC", avg_BricoC)
plotSignature("Oeuf", avg_Oeuf)




