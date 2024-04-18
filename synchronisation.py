import pandas as pd

# Charger les données pour les modules mod1, mod2, POD85, POD86, POD88, PICO, THIN et THICK
mod1_concatenated = pd.read_csv("new_data/mod1_concatenated.txt", sep="\t")
mod2_concatenated = pd.read_csv("new_data/mod2_concatenated.txt", sep="\t")
pod85_concatenated = pd.read_csv("new_data/POD_85_concatenated.csv")
pod86_concatenated = pd.read_csv("new_data/POD_86_concatenated.csv")
pod88_concatenated = pd.read_csv("new_data/POD_88_concatenated.csv")
pico_concatenated = pd.read_csv("new_data/PICO_concatenated.csv")
thin_concatenated = pd.read_csv("new_data/THIN_concatenated.csv")
thick_concatenated = pd.read_csv("new_data/THICK_concatenated.csv")

# Convertir la colonne 'Time' en type datetime64
mod1_concatenated["Time"] = pd.to_datetime(mod1_concatenated["Time"])
mod2_concatenated["Time"] = pd.to_datetime(mod2_concatenated["Time"])
pod85_concatenated["Time"] = pd.to_datetime(pod85_concatenated["Time"])
pod86_concatenated["Time"] = pd.to_datetime(pod86_concatenated["Time"])
pod88_concatenated["Time"] = pd.to_datetime(pod88_concatenated["Time"])
pico_concatenated["Time"] = pd.to_datetime(pico_concatenated["Time"])
thin_concatenated["Time"] = pd.to_datetime(thin_concatenated["Time"])
thick_concatenated["Time"] = pd.to_datetime(thick_concatenated["Time"])

# Pour les modules mod1 et mod2
mod1_concatenated["Time"] = mod1_concatenated["Time"].dt.round('10 s')
mod2_concatenated["Time"] = mod2_concatenated["Time"].dt.round('10 s')

# Regrouper les données des modules mod1 et mod2 par timestamp et calculer la moyenne pour chaque groupe
mod1_grouped = mod1_concatenated.groupby('Time').mean().reset_index()
mod2_grouped = mod2_concatenated.groupby('Time').mean().reset_index()

# Fusionner les données de tous les modules en utilisant la colonne 'Time' comme référence
merged_data = pd.merge(pod85_concatenated, pod86_concatenated, on='Time', suffixes=('_POD85', '_POD86'))
merged_data = pd.merge(merged_data, pod88_concatenated, on='Time')
merged_data = pd.merge(merged_data, pico_concatenated, on='Time')
merged_data = pd.merge(merged_data, thin_concatenated, on='Time')
merged_data = pd.merge(merged_data, thick_concatenated, on='Time')
merged_data = pd.merge(merged_data, mod1_grouped, on='Time', how='outer', suffixes=('_POD88', '_PICO'))
merged_data = pd.merge(merged_data, mod2_grouped, on='Time', how='outer', suffixes=('_THIN', '_THICK'))

# Informations sur le DataFrame final
nb_lignes_merged_data, nb_colonnes_merged_data = merged_data.shape
print(f"DataFrame final => Colonnes : {nb_colonnes_merged_data} \t Lignes : {nb_lignes_merged_data}")



# Enregistrer les données fusionnées dans un fichier CSV
merged_data.to_csv("new_data/merged_data.csv", index=False)
