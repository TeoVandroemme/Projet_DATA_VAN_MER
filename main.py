import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Chemin vers les fichiers de données pour mod1 et mod2
mod1_files = ["fichiers_data/Libelium New/part1/mod1.txt",
              "fichiers_data/Libelium New/part2/mod1.txt",
              "fichiers_data/Libelium New/part3/mod1.txt",
              "fichiers_data/Libelium New/part4/mod1.txt",
              "fichiers_data/Libelium New/part5/mod1.txt",
              "fichiers_data/Libelium New/part6/mod1.txt",
              "fichiers_data/Libelium New/part7/mod1.txt",
              "fichiers_data/Libelium New/part8/mod1.txt"]

mod2_files = ["fichiers_data/Libelium New/part1/mod2.txt",
              "fichiers_data/Libelium New/part2/mod2.txt",
              "fichiers_data/Libelium New/part3/mod2.txt",
              "fichiers_data/Libelium New/part4/mod2.txt",
              "fichiers_data/Libelium New/part5/mod2.txt",
              "fichiers_data/Libelium New/part6/mod2.txt",
              "fichiers_data/Libelium New/part7/mod2.txt",
              "fichiers_data/Libelium New/part8/mod2.txt"]

# Lire et concaténer les fichiers pour mod1
mod1_concatenated = pd.concat([pd.read_csv(file, sep="\t", header=None, names=(
    "Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620")) for file in
                               mod1_files])

# Lire et concaténer les fichiers pour mod2
mod2_concatenated = pd.concat([pd.read_csv(file, sep="\t", header=None, names=(
    "Time", "RH", "Temperature", "TGS4161", "MICS2714", "TGS2442", "MICS5524", "TGS2602", "TGS2620")) for file in
                               mod2_files])

# Convertir la colonne 'Time' en datetime et définir le fuseau horaire
mod1_concatenated["Time"] = pd.to_datetime(mod1_concatenated['Time'], format="%d/%m/%Y %H:%M:%S").dt.tz_localize(
    'UTC+01:00', ambiguous='infer')
mod2_concatenated["Time"] = pd.to_datetime(mod2_concatenated['Time'], format="%d/%m/%Y %H:%M:%S").dt.tz_localize(
    'UTC+01:00', ambiguous='infer')

# Supprimer les doublons
mod1_concatenated.drop_duplicates(inplace=True)
mod2_concatenated.drop_duplicates(inplace=True)

# Sauvegarder les données concaténées dans des fichiers .txt
mod1_concatenated.to_csv("new_data/mod1_concatenated.txt", sep='\t', index=False)
mod2_concatenated.to_csv("new_data/mod2_concatenated.txt", sep='\t', index=False)

# Compter le nombre de lignes dans mod1_concatenated
nb_lignes_mod1_concatenated = mod1_concatenated.shape[0]

# Afficher le nombre de lignes mod1
print("Le nombre de lignes dans mod1_concatenated est :", nb_lignes_mod1_concatenated)

# Compter le nombre de colonnes dans mod1_concatenated
nb_colonnes_mod1_concatenated = mod1_concatenated.shape[1]

# Afficher le nombre de colonnes mod1
print("Le nombre de colonnes dans mod1_concatenated est :", nb_colonnes_mod1_concatenated)

# Compter le nombre de lignes dans mod2_concatenated
nb_lignes_mod2_concatenated = mod2_concatenated.shape[0]

# Afficher le nombre de lignes mod2
print("Le nombre de lignes dans mod2_concatenated est :", nb_lignes_mod2_concatenated)

# Compter le nombre de colonnes dans mod2_concatenated
nb_colonnes_mod2_concatenated = mod2_concatenated.shape[1]

# Afficher le nombre de colonnes mod2
print("Le nombre de colonnes dans mod2_concatenated est :", nb_colonnes_mod2_concatenated)

pod85_paths = [
    "fichiers_data/PODs/14_nov-22_nov-Pods/POD200085.csv",
    "fichiers_data/PODs/23_nov-12_dec-Pods/POD200085.csv",
    "fichiers_data/PODs/fevrier_mars_pods/POD200085.csv"
]

# Concaténer les fichiers pour chaque module POD85
pod85 = pd.concat([pd.read_csv(file, sep=";", skiprows=[1, 2, 3, 4]) for file in pod85_paths])

# Supprimer les doublons
pod85.drop_duplicates(inplace=True)

# Supprimer les colonnes inutiles et renommer la colonne 'date' en 'Time'
pod85 = pod85.drop(columns=['element', 'aqi', 'Unnamed: 0'])
pod85 = pod85.rename(columns={'date': 'Time'})

# Convertir la colonne 'Time' en datetime
pod85['Time'] = pd.to_datetime(pod85['Time'])

# Sauvegarder les données traitées dans un fichier CSV unique pour chaque module POD85
pod85.to_csv(f"new_data/POD_85_concatenated.csv", index=False)

pod86_paths = [
    "fichiers_data/PODs/14_nov-22_nov-Pods/POD200086.csv",
    "fichiers_data/PODs/23_nov-12_dec-Pods/POD200086.csv",
    "fichiers_data/PODs/fevrier_mars_pods/POD200086.csv"
]

# Concaténer les fichiers pour chaque module POD86
pod86 = pd.concat([pd.read_csv(file, sep=";", skiprows=[1, 2, 3, 4]) for file in pod86_paths])

# Supprimer les doublons
pod86.drop_duplicates(inplace=True)

# Supprimer les colonnes inutiles et renommer la colonne 'date' en 'Time'
pod86 = pod86.drop(columns=['element', 'aqi', 'Unnamed: 0'])
pod86 = pod86.rename(columns={'date': 'Time'})

# Convertir la colonne 'Time' en datetime
pod86['Time'] = pd.to_datetime(pod86['Time'])

# Sauvegarder les données traitées dans un fichier CSV unique pour chaque module POD86
pod86.to_csv(f"new_data/POD_86_concatenated.csv", index=False)

pod88_paths = [
    "fichiers_data/PODs/14_nov-22_nov-Pods/POD200088.csv",
    "fichiers_data/PODs/23_nov-12_dec-Pods/POD200088.csv",
    "fichiers_data/PODs/fevrier_mars_pods/POD200088.csv"
]

# Concaténer les fichiers pour chaque module POD88
pod88 = pd.concat([pd.read_csv(file, sep=";", skiprows=[1, 2, 3, 4]) for file in pod88_paths])

# Supprimer les doublons
pod88.drop_duplicates(inplace=True)

# Supprimer les colonnes inutiles et renommer la colonne 'date' en 'Time'
pod88 = pod88.drop(columns=['element', 'aqi', 'Unnamed: 0'])
pod88 = pod88.rename(columns={'date': 'Time'})

# Convertir la colonne 'Time' en datetime
pod88['Time'] = pd.to_datetime(pod88['Time'])

# Sauvegarder les données traitées dans un fichier CSV unique pour chaque module POD88
pod88.to_csv(f"new_data/POD_88_concatenated.csv", index=False)

# Compter le nombre de lignes dans pod85_concatenated
nb_lignes_pod85_concatenated = pod85.shape[0]

# Afficher le nombre de lignes pour le module POD85
print("Le nombre de lignes dans pod85_concatenated est :", nb_lignes_pod85_concatenated)

# Compter le nombre de colonnes dans pod85_concatenated
nb_colonnes_pod85_concatenated = pod85.shape[1]

# Afficher le nombre de colonnes pour le module POD85
print("Le nombre de colonnes dans pod85_concatenated est :", nb_colonnes_pod85_concatenated)

# Compter le nombre de lignes dans pod86_concatenated
nb_lignes_pod86_concatenated = pod86.shape[0]

# Afficher le nombre de lignes pour le module POD86
print("Le nombre de lignes dans pod86_concatenated est :", nb_lignes_pod86_concatenated)

# Compter le nombre de colonnes dans pod86_concatenated
nb_colonnes_pod86_concatenated = pod86.shape[1]

# Afficher le nombre de colonnes pour le module POD86
print("Le nombre de colonnes dans pod86_concatenated est :", nb_colonnes_pod86_concatenated)

# Compter le nombre de lignes dans pod88_concatenated
nb_lignes_pod88_concatenated = pod88.shape[0]

# Afficher le nombre de lignes pour le module POD88
print("Le nombre de lignes dans pod88_concatenated est :", nb_lignes_pod88_concatenated)

# Compter le nombre de colonnes dans pod88_concatenated
nb_colonnes_pod88_concatenated = pod88.shape[1]

# Afficher le nombre de colonnes pour le module POD88
print("Le nombre de colonnes dans pod88_concatenated est :", nb_colonnes_pod88_concatenated)

# File paths for PICO data
pico_paths = [
    "fichiers_data/Piano/14_nov-22_nov-Piano/IMT_PICO.csv",
    "fichiers_data/Piano/23_nov-12_dec-Piano/IMT_PICO.csv",
    "fichiers_data/Piano/fevrier_mars_2023_piano/IMT_PICO.csv"
]

# Concatenate and process PICO data
for i, pico_path in enumerate(pico_paths, start=1):
    # Read data, skip unnecessary rows, rename columns, and convert 'date' column to datetime
    pico_data = pd.read_csv(pico_path, sep=";", skiprows=[1, 2, 3, 4])
    pico_data = pico_data.drop(columns=[col for col in pico_data.columns if
                                        'aqi' in col or 'qai' in col or 'iaq' in col or col == 'element' or 'Unnamed' in col])
    pico_data = pico_data.rename(columns={'date': 'Time'})
    pico_data['Time'] = pd.to_datetime(pico_data['Time'])

    # Remove duplicates
    pico_data.drop_duplicates(inplace=True)

    # Save processed data to CSV
    pico_data.to_csv(f"new_data/PICO_{i}_concatenated.csv", index=False)

# Compter le nombre de lignes dans pico_data_concatenated
nb_lignes_pico_concatenated = pico_data.shape[0]

# Afficher le nombre de lignes pour le module PICO
print("Le nombre de lignes dans pico_data_concatenated est :", nb_lignes_pico_concatenated)

# Compter le nombre de colonnes dans pico_data_concatenated
nb_colonnes_pico_concatenated = pico_data.shape[1]

# Afficher le nombre de colonnes pour le module PICO
print("Le nombre de colonnes dans pico_data_concatenated est :", nb_colonnes_pico_concatenated)


# The code:
# plt.figure(figsize=(10, 2))  # Plot overview of the files
# date_format = mdates.DateFormatter('%d-%b')

# plt.subplot(3, 2, 1)  # MOD1
# plt.subplots_adjust(top=8)
# plt.title("MOD1 (Temperature x Time)")
# plt.plot(mod1_concatenated['Time'], mod1_concatenated['Temperature'])
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.xticks(rotation=45)

# plt.subplot(3, 2, 2)  # MOD2
# plt.subplots_adjust(top=8)
# plt.title("MOD2 (Temperature x Time)")
# plt.plot(mod2_concatenated['Time'], mod2_concatenated['Temperature'])
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.xticks(rotation=45)

# Plot pour POD 200085
# plt.subplot(3, 1, 1)
# plt.title("POD 200085 (Temperature x Time)")
# plt.plot(POD_85_concatenated['Time'], POD_85_concatenated['Temperature'])
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.xticks(rotation=45)

# Afficher les graphiques
# plt.show()
