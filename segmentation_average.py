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
new_activity_df = activity_df.drop(columns=['Comments', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7']).dropna()

# Supprimer les lignes contenant des valeurs NaN
new_activity_df = new_activity_df.dropna(ignore_index = True)

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
# Ajoutez autant de listes que nécessaire pour chaque activité

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
# Ajoutez autant de variables que nécessaire pour chaque activité

# Parcourir les activités dans le calendrier
for idact, act in enumerate(new_activity_df['activity']):
    start = new_activity_df['Started'][idact]
    end = new_activity_df['Ended'][idact]

    # Segmenter la base de données en fonction de chaque activité
    if act == 'Saber':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Saber += len(segment)
        Saber_grouped.append(segment)
    elif act == 'Aera':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Aera += len(segment)
        Aera_grouped.append(segment)
    elif act == 'Nett':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Nett += len(segment)
        Nett_grouped.append(segment)
    elif act == 'Asp':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Asp += len(segment)
        Asp_grouped.append(segment)
    elif act == 'AS1':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_AS1 += len(segment)
        AS1_grouped.append(segment)
    elif act == 'Bougie':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Bougie += len(segment)
        Bougie_grouped.append(segment)
    elif act == 'SdB':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_SdB += len(segment)
        SdB_grouped.append(segment)
    elif act == 'BricoP':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_BricoP += len(segment)
        BricoP_grouped.append(segment)
    elif act == 'BricoC':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_BricoC += len(segment)
        BricoC_grouped.append(segment)
    elif act == 'Oeuf':
        segment = base[(base['Time'] >= start) & (base['Time'] <= end)].reset_index(drop=True).sort_values(by='Time').drop(columns='Time')
        len_Oeuf += len(segment)
        Oeuf_grouped.append(segment)
    # Ajoutez des clauses elif pour chaque activité supplémentaire

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
