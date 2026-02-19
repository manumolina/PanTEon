import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from Bio import SeqIO

diccionario = {
    "CLASSI/DIRS/DIRS": ["DIRS"],
    "CLASSII/CRYPTON/CRYPTON": ["CRYPTON"],
    "CLASSII/HELITRON/HELITRON": ["HELITRON", "RC"],
    "CLASSII/MAVERICK/MAVERICK": ["MAVERICK"],
    "CLASSII/TIR/CACTA": ["CACTA", "CMC"],
    "CLASSII/TIR/HAT": ["HAT"],
    "CLASSII/TIR/MERLIN": ["MERLIN"],
    "CLASSII/TIR/MULE": ["MULE", "MUTATOR"],
    "CLASSII/TIR/P": ["P"],
    "CLASSII/TIR/PIFHARBINGER": ["PIFHARBINGER", "HARBINGER", "PIF", "PIF-HARBINGER"],
    "CLASSII/TIR/PIGGYBAC": ["PIGGYBAC"],
    "CLASSII/TIR/TC1MARINER": ["TC1MARINER", "TC1-MARINER", "TCMAR"],
    "CLASSII/TIR/DADA": ["DADA"],
    "CLASSII/TIR/KOLOBOK": ["KOLOBOK"],
    "CLASSII/TIR/TRANSIB": ["TRANSIB"],
    "CLASSII/TIR/TIR": ["TIR", "DNA", "NMITE"],
    "CLASSII/MITE/MITE": ["MITE"],
    "CLASSI/LINE/I": ["I", "JOCKEY", "TAD1", "R1"],
    "CLASSI/LINE/L1": ["L1", "L1_L2"],
    "CLASSI/LINE/LINE": ["LINE"],
    "CLASSI/LINE/R2": ["R2"],
    "CLASSI/LINE/RTE": ["RTE", "PROTO2"],
    "CLASSI/LINE/CR1": ["CR1", "L2", "REX1", "REX-BABAR"],
    "CLASSI/LINE/CRE": ["CRE"],
    "CLASSI/LTR/BELPAO": ["BELPAO", "BEL-PAO", "BEL", "PAO"],
    "CLASSI/LTR/COPIA": ["COPIA"],
    "CLASSI/LTR/ERV": ["ERV", "CAULIMOVIRUS", "RETROVIRUS"],
    "CLASSI/LTR/GYPSY": ["GYPSY"],
    "CLASSI/LTR/LTR": ["LTR"],
    "CLASSI/PLE/PLE": ["PLE", "PENELOPE"],
    "CLASSI/SINE/5S": ["5S"],
    "CLASSI/SINE/SINE": ["SINE"],
    "CLASSI/SINE/7S": ["7SL"],
    "CLASSI/SINE/tRNA": ["SINE2/TRNA", "TRNA"],
    "CLASSI/SINE/U": ["U"],
    "CLASSII/TIR/ACADEM-1": ["ACADEM"],
    "CLASSI/CLASSI/CLASSI": ["CLASSI"],
    "ClASSII/ClASSII/ClASSII": ["CLASSII"],
    "CLASSI/UNKNOWN/UNKNOWN": ["NLTR"],
    "UNKNOWN/UNKNOWN/UNKNOWN": ["UNKNOWN"]

}

df = pd.read_excel('classification_normalized.xlsx')
df = df.fillna("UNKNOWN")

modelos = ['CREATE', 'TERL', 'NeuralTE', 'DeepTE', 'TEClass2', 'Terrier', 'ClassifyTE']
niveles = ['class', 'order', 'SF']  # SF = Superfamily

# Original columns
original_cols = {
    'class': 'Original_class',
    'order': 'Original_order',
    'SF': 'Original_SF'
}

# Step 1: Generate the base mapping
mapping = {}
for key, variantes in diccionario.items():
    partes = key.split('/')
    if len(partes) == 3:
        clase, orden, superfam = partes
    else:
        clase = orden = superfam = "Not found"
    for v in variantes:
        mapping[v] = [clase, orden, superfam]

for modelo in modelos:
    df[modelo+"_SF_std"] = [mapping[str(df.loc[x, modelo+"_SF"]).upper()][2] for x in range(df.shape[0])]

print(df)
df.to_excel('classification_normalized_v2.xlsx', index=False)

resumen = []

for modelo in modelos:
    for nivel in niveles:
        if nivel == "SF":
            print(df[f'{modelo}_{nivel}_std'].value_counts())
            y_true = df[original_cols[nivel]]
            y_pred = df[f'{modelo}_{nivel}_std'].str.upper()
        else:
            print(df[f'{modelo}_{nivel}'].value_counts())
            y_true = df[original_cols[nivel]]
            y_pred = df[f'{modelo}_{nivel}'].str.upper()

        if len(y_true) == 0:
            print(f'There are no valid data for {modelo} at level {nivel}')
            continue

        resumen.append({
            'Model': modelo,
            'Level': nivel,
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        })

        print(f'\n--- Model metrics {modelo} at  level {nivel} ---')
        print(classification_report(y_true, y_pred, zero_division=0))

        all_classes = sorted(set(list(y_true) + list(y_pred)))
        snn_cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        num_classes = len(pd.concat([y_true, y_pred]).unique())

        snn_df_cm = pd.DataFrame(snn_cm, index=all_classes, columns=all_classes)

        plt.figure(figsize=(20, 14))
        plt.imshow(snn_df_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{modelo} {nivel}')
        plt.colorbar()

        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_marks, all_classes, fontsize=12)

        for i in range(snn_cm.shape[0]):
            for j in range(snn_cm.shape[1]):
                plt.text(j, i, snn_cm[i, j],
                         ha="center", va="center",
                         color="white" if snn_cm[i, j] > snn_cm.max() / 2. else "black",
                         fontsize=12)

        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'confusionMatrix_{modelo}_{nivel}.png', bbox_inches='tight', dpi=500)
        plt.close()

df_resumen = pd.DataFrame(resumen)
df_resumen.to_excel('metrics_brief_models_levels_all.xlsx', index=False)

# metrics by each kingdom
fungi = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_fungi.fasta", "fasta")]
plants = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_plants.fasta", "fasta")]
animals = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_animals.fasta", "fasta")]

resumen = []
df_fungi = df[df["TE_ID"].isin(fungi)]
for modelo in modelos:
    for nivel in niveles:
        if nivel == "SF":
            print(df_fungi[f'{modelo}_{nivel}_std'].value_counts())
            y_true = df_fungi[original_cols[nivel]]
            y_pred = df_fungi[f'{modelo}_{nivel}_std'].str.upper()
        else:
            print(df_fungi[f'{modelo}_{nivel}'].value_counts())
            y_true = df_fungi[original_cols[nivel]]
            y_pred = df_fungi[f'{modelo}_{nivel}'].str.upper()

        if len(y_true) == 0:
            print(f'There is no valid data for {modelo} at level {nivel}')
            continue

        resumen.append({
            'Model': modelo,
            'Level': nivel,
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        })

        print(f'\n--- Model metrics {modelo} at level {nivel} Fungi ---')
        print(classification_report(y_true, y_pred, zero_division=0))

        all_classes = sorted(set(list(y_true) + list(y_pred)))
        snn_cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        num_classes = len(pd.concat([y_true, y_pred]).unique())

        snn_df_cm = pd.DataFrame(snn_cm, index=all_classes, columns=all_classes)

        plt.figure(figsize=(20, 14))
        plt.imshow(snn_df_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{modelo} {nivel}')
        plt.colorbar()

        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_marks, all_classes, fontsize=12)

        for i in range(snn_cm.shape[0]):
            for j in range(snn_cm.shape[1]):
                plt.text(j, i, snn_cm[i, j],
                         ha="center", va="center",
                         color="white" if snn_cm[i, j] > snn_cm.max() / 2. else "black",
                         fontsize=12)

        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'confusionMatrix_{modelo}_{nivel}_fungi.png', bbox_inches='tight', dpi=500)
        plt.close()

df_resumen = pd.DataFrame(resumen)
df_resumen.to_excel('metrics_brief_models_levels_fungi.xlsx', index=False)

resumen = []
df_plants = df[df["TE_ID"].isin(plants)]
for modelo in modelos:
    for nivel in niveles:
        if nivel == "SF":
            print(df_plants[f'{modelo}_{nivel}_std'].value_counts())
            y_true = df_plants[original_cols[nivel]]
            y_pred = df_plants[f'{modelo}_{nivel}_std'].str.upper()
        else:
            print(df_plants[f'{modelo}_{nivel}'].value_counts())
            y_true = df_plants[original_cols[nivel]]
            y_pred = df_plants[f'{modelo}_{nivel}'].str.upper()

        if len(y_true) == 0:
            print(f'There is no valid data for {modelo} level {nivel}')
            continue

        resumen.append({
            'Model': modelo,
            'Level': nivel,
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        })

        print(f'\n--- Model metrics {modelo} at level {nivel} Plants ---')
        print(classification_report(y_true, y_pred, zero_division=0))

        all_classes = sorted(set(list(y_true) + list(y_pred)))
        snn_cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        num_classes = len(pd.concat([y_true, y_pred]).unique())

        snn_df_cm = pd.DataFrame(snn_cm, index=all_classes, columns=all_classes)

        plt.figure(figsize=(20, 14))
        plt.imshow(snn_df_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{modelo} {nivel}')
        plt.colorbar()

        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_marks, all_classes, fontsize=12)

        for i in range(snn_cm.shape[0]):
            for j in range(snn_cm.shape[1]):
                plt.text(j, i, snn_cm[i, j],
                         ha="center", va="center",
                         color="white" if snn_cm[i, j] > snn_cm.max() / 2. else "black",
                         fontsize=12)

        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'confusionMatrix_{modelo}_{nivel}_plants.png', bbox_inches='tight', dpi=500)
        plt.close()

df_resumen = pd.DataFrame(resumen)
df_resumen.to_excel('metrics_brief_models_levels_plants.xlsx', index=False)

resumen = []
df_animals = df[df["TE_ID"].isin(animals)]
for modelo in modelos:
    for nivel in niveles:
        if nivel == "SF":
            print(df_animals[f'{modelo}_{nivel}_std'].value_counts())
            y_true = df_animals[original_cols[nivel]]
            y_pred = df_animals[f'{modelo}_{nivel}_std'].str.upper()
        else:
            print(df_animals[f'{modelo}_{nivel}'].value_counts())
            y_true = df_animals[original_cols[nivel]]
            y_pred = df_animals[f'{modelo}_{nivel}'].str.upper()

        if len(y_true) == 0:
            print(f'There is no valid data for {modelo} level {nivel}')
            continue

        resumen.append({
            'Model': modelo,
            'Level': nivel,
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        })

        print(f'\n--- Model metrics {modelo} at level {nivel} Animals ---')
        print(classification_report(y_true, y_pred, zero_division=0))

        all_classes = sorted(set(list(y_true) + list(y_pred)))
        snn_cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        num_classes = len(pd.concat([y_true, y_pred]).unique())

        snn_df_cm = pd.DataFrame(snn_cm, index=all_classes, columns=all_classes)

        plt.figure(figsize=(20, 14))
        plt.imshow(snn_df_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{modelo} {nivel}')
        plt.colorbar()

        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_marks, all_classes, fontsize=12)

        for i in range(snn_cm.shape[0]):
            for j in range(snn_cm.shape[1]):
                plt.text(j, i, snn_cm[i, j],
                         ha="center", va="center",
                         color="white" if snn_cm[i, j] > snn_cm.max() / 2. else "black",
                         fontsize=12)

        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'confusionMatrix_{modelo}_{nivel}_animals.png', bbox_inches='tight', dpi=500)
        plt.close()

df_resumen = pd.DataFrame(resumen)
df_resumen.to_excel('metrics_brief_models_levels_animals.xlsx', index=False)