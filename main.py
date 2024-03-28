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
mod1_concatenated = pd.concat([pd.read_csv(file, sep="\t", header=None, names=("Time","RH","Temperature","TGS4161","MICS2714","TGS2442","MICS5524","TGS2602","TGS2620")) for file in mod1_files])

# Lire et concaténer les fichiers pour mod2
mod2_concatenated = pd.concat([pd.read_csv(file, sep="\t", header=None, names=("Time","RH","Temperature","TGS4161","MICS2714","TGS2442","MICS5524","TGS2602","TGS2620")) for file in mod2_files])

# Convertir la colonne 'Time' en datetime et définir le fuseau horaire
mod1_concatenated["Time"] = pd.to_datetime(mod1_concatenated['Time'], format="%d/%m/%Y %H:%M:%S").dt.tz_localize('UTC+01:00', ambiguous='infer')
mod2_concatenated["Time"] = pd.to_datetime(mod2_concatenated['Time'], format="%d/%m/%Y %H:%M:%S").dt.tz_localize('UTC+01:00', ambiguous='infer')


# Supprimer les doublons
mod1_concatenated.drop_duplicates(inplace=True)
mod2_concatenated.drop_duplicates(inplace=True)

# Sauvegarder les données concaténées dans des fichiers .txt
mod1_concatenated.to_csv("new_data/mod1_concatenated.txt", sep='\t', index=False)
mod2_concatenated.to_csv("new_data/mod2_concatenated.txt", sep='\t', index=False)

# Compter le nombre de lignes dans mod1_concatenated
nb_lignes_mod1_concatenated = mod1_concatenated.shape[0]

# Afficher le nombre de lignes
print("Le nombre de lignes dans mod1_concatenated est :", nb_lignes_mod1_concatenated)

# Compter le nombre de colonnes dans mod1_concatenated
nb_colonnes_mod1_concatenated = mod1_concatenated.shape[1]

# Afficher le nombre de colonnes
print("Le nombre de colonnes dans mod1_concatenated est :", nb_colonnes_mod1_concatenated)



# The code:
plt.figure(figsize=(10, 2))  # Plot overview of the files
date_format = mdates.DateFormatter('%d-%b')

plt.subplot(3,2,1) # MOD1
plt.subplots_adjust(top=8)
plt.title("MOD1 (Temperature x Time)")
plt.plot(mod1_concatenated['Time'], mod1_concatenated['Temperature'])
plt.gca().xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45)

# plt.subplot(3,2,2) # MOD2
# plt.subplots_adjust(top=8)
# plt.title("MOD2 (Temperature x Time)")
# plt.plot(mod2_concatenated['Time'], mod2_concatenated['Temperature'])
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.xticks(rotation=45)

# Afficher les graphiques
plt.show()
