import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_excel('clasificacion_normalizada_simon_v2.xlsx')
df = df.fillna("UNKNOWN")

# train/test (60/40) split
df_train, df_test = train_test_split(df, test_size=0.4, stratify=df['Original_SF'], random_state=42)

modelos = ['CREATE', 'TERL', 'NeuralTE', 'DeepTE', 'TEClass2', 'Terrier', 'ClassifyTE']
niveles = ['class', 'order', 'SF']

original_cols = {
    'class': 'Original_class',
    'order': 'Original_order',
    'SF': 'Original_SF'
}

# Majority voting system
def majority_vote_class(row):
    predicciones = row[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]
    voto = Counter(predicciones).most_common(1)[0][0]
    return voto

def majority_vote_order(row):
    predicciones = row[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]
    voto = Counter(predicciones).most_common(1)[0][0]
    return voto

def majority_vote_SF(row):
    predicciones = row[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]
    voto = Counter(predicciones).most_common(1)[0][0]
    return voto

df_test['maj_class'] = df_test.apply(majority_vote_class, axis=1)
df_test['maj_order'] = df_test.apply(majority_vote_order, axis=1)
df_test['maj_SF'] = df_test.apply(majority_vote_SF, axis=1)

# Weighted voting system
pesos = {
    'ClassifyTE': 1.0,
    'CREATE': 1.3,
    'DeepTE': 1.1,
    'NeuralTE': 1.5,
    'TEClass2': 1.2,
    'TERL': 1.0,
    'Terrier': 1.5
}

def weighted_vote_class(row):
    votos = {}
    for tool, peso in pesos.items():
        clase = row[f'{tool}_class']
        votos[clase] = votos.get(clase, 0) + peso
    return max(votos.items(), key=lambda x: x[1])[0]

def weighted_vote_order(row):
    votos = {}
    for tool, peso in pesos.items():
        clase = row[f'{tool}_order']
        votos[clase] = votos.get(clase, 0) + peso
    return max(votos.items(), key=lambda x: x[1])[0]

def weighted_vote_SF(row):
    votos = {}
    for tool, peso in pesos.items():
        clase = row[f'{tool}_SF_std']
        votos[clase] = votos.get(clase, 0) + peso
    return max(votos.items(), key=lambda x: x[1])[0]

df_test['wei_class'] = df_test.apply(weighted_vote_class, axis=1)
df_test['wei_order'] = df_test.apply(weighted_vote_order, axis=1)
df_test['wei_SF'] = df_test.apply(weighted_vote_SF, axis=1)

# Stacking method
le = LabelEncoder()
y_train = le.fit_transform(df_train['Original_class'])
y_test = le.transform(df_test['Original_class'])

X_train = df_train[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]
X_test = df_test[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]

for col in X_train.columns:
    le_tool = LabelEncoder()
    le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
    X_train[col] = le_tool.transform(X_train[col])
    X_test[col] = le_tool.transform(X_test[col])

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
df_test['stack_class'] = le.inverse_transform(y_pred)

print("F1-score total Class:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
print("Full Report Class:\n", classification_report(y_test, y_pred, zero_division=0))

le = LabelEncoder()
y_train = le.fit_transform(df_train['Original_order'])
y_test = le.transform(df_test['Original_order'])

X_train = df_train[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]
X_test = df_test[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]

for col in X_train.columns:
    le_tool = LabelEncoder()
    le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
    X_train[col] = le_tool.transform(X_train[col])
    X_test[col] = le_tool.transform(X_test[col])

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
df_test['stack_order'] = le.inverse_transform(y_pred)

print("F1-score total Order:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
print("Full Report Order:\n", classification_report(y_test, y_pred, zero_division=0))

le = LabelEncoder()
y_train = le.fit_transform(df_train['Original_SF'])
y_test = le.transform(df_test['Original_SF'])

X_train = df_train[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]
X_test = df_test[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]

for col in X_train.columns:
    le_tool = LabelEncoder()
    le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
    X_train[col] = le_tool.transform(X_train[col])
    X_test[col] = le_tool.transform(X_test[col])

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
df_test['stack_SF'] = le.inverse_transform(y_pred)

print("F1-score total SF:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
print("Full Report SF:\n", classification_report(y_test, y_pred, zero_division=0))

print(df_test)

resumen = []
modelos = ["maj", "wei", "stack"]
niveles = ["class", "order", "SF"]

for modelo in modelos:
    for nivel in niveles:
        print(df_test[f'{modelo}_{nivel}'].value_counts())
        y_true = df_test[original_cols[nivel]]
        y_pred = df_test[f'{modelo}_{nivel}'].str.upper()

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

        print(f'\n--- Metric for model {modelo} at level {nivel} ---')
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
df_resumen.to_excel('metrics_brief_ensemble_all.xlsx', index=False)

fungi = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_fungi.fasta", "fasta")]
plants = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_plants.fasta", "fasta")]
animals = [te.id.split(" ")[0] for te in SeqIO.parse("dataset/PanTEon_DB_subset_animals.fasta", "fasta")]

resumen = []

for kingdom in ["animals", "plants", "fungi"]:

    print("########## Doing metrics for: "+kingdom)
    if kingdom == "animals":
        df_i = df[df["TE_ID"].isin(animals)]
    elif kingdom == "plants":
        df_i = df[df["TE_ID"].isin(plants)]
    elif kingdom == "fungi":
        df_i = df[df["TE_ID"].isin(fungi)]

    df_i = (df_i
        .groupby('Original_SF')
        .filter(lambda g: len(g) >= 2)
        .reset_index(drop=True)
    )
    df_train, df_test = train_test_split(df_i, test_size=0.4, stratify=df_i['Original_SF'], random_state=42)

    modelos = ['CREATE', 'TERL', 'NeuralTE', 'DeepTE', 'TEClass2', 'Terrier', 'ClassifyTE']
    niveles = ['class', 'order', 'SF']  # SF = Superfamily

    original_cols = {
        'class': 'Original_class',
        'order': 'Original_order',
        'SF': 'Original_SF'
    }

    def majority_vote_class(row):
        predicciones = row[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]
        voto = Counter(predicciones).most_common(1)[0][0]
        return voto

    def majority_vote_order(row):
        predicciones = row[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]
        voto = Counter(predicciones).most_common(1)[0][0]
        return voto

    def majority_vote_SF(row):
        predicciones = row[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]
        voto = Counter(predicciones).most_common(1)[0][0]
        return voto

    df_test['maj_class'] = df_test.apply(majority_vote_class, axis=1)
    df_test['maj_order'] = df_test.apply(majority_vote_order, axis=1)
    df_test['maj_SF'] = df_test.apply(majority_vote_SF, axis=1)

    pesos = {
        'ClassifyTE': 1.0,
        'CREATE': 1.3,
        'DeepTE': 1.1,
        'NeuralTE': 1.5,
        'TEClass2': 1.2,
        'TERL': 1.0,
        'Terrier': 1.5
    }

    def weighted_vote_class(row):
        votos = {}
        for tool, peso in pesos.items():
            clase = row[f'{tool}_class']
            votos[clase] = votos.get(clase, 0) + peso
        return max(votos.items(), key=lambda x: x[1])[0]

    def weighted_vote_order(row):
        votos = {}
        for tool, peso in pesos.items():
            clase = row[f'{tool}_order']
            votos[clase] = votos.get(clase, 0) + peso
        return max(votos.items(), key=lambda x: x[1])[0]

    def weighted_vote_SF(row):
        votos = {}
        for tool, peso in pesos.items():
            clase = row[f'{tool}_SF_std']
            votos[clase] = votos.get(clase, 0) + peso
        return max(votos.items(), key=lambda x: x[1])[0]

    df_test['wei_class'] = df_test.apply(weighted_vote_class, axis=1)
    df_test['wei_order'] = df_test.apply(weighted_vote_order, axis=1)
    df_test['wei_SF'] = df_test.apply(weighted_vote_SF, axis=1)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['Original_class'])
    y_test = le.transform(df_test['Original_class'])

    X_train = df_train[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]
    X_test = df_test[['CREATE_class', 'TERL_class', 'NeuralTE_class', 'DeepTE_class', 'Terrier_class', 'TEClass2_class', 'ClassifyTE_class']]

    for col in X_train.columns:
        le_tool = LabelEncoder()
        le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
        X_train[col] = le_tool.transform(X_train[col])
        X_test[col] = le_tool.transform(X_test[col])

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['stack_class'] = le.inverse_transform(y_pred)

    print("F1-score total Class:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Full Report Class:\n", classification_report(y_test, y_pred, zero_division=0))

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['Original_order'])
    y_test = le.transform(df_test['Original_order'])

    X_train = df_train[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]
    X_test = df_test[['CREATE_order', 'TERL_order', 'NeuralTE_order', 'DeepTE_order', 'Terrier_order', 'TEClass2_order', 'ClassifyTE_order']]

    for col in X_train.columns:
        le_tool = LabelEncoder()
        le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
        X_train[col] = le_tool.transform(X_train[col])
        X_test[col] = le_tool.transform(X_test[col])

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['stack_order'] = le.inverse_transform(y_pred)

    print("F1-score total Order:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Full Report Order:\n", classification_report(y_test, y_pred, zero_division=0))

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['Original_SF'])
    y_test = le.transform(df_test['Original_SF'])

    X_train = df_train[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]
    X_test = df_test[['CREATE_SF_std', 'TERL_SF_std', 'NeuralTE_SF_std', 'DeepTE_SF_std', 'Terrier_SF_std', 'TEClass2_SF_std', 'ClassifyTE_SF_std']]

    for col in X_train.columns:
        le_tool = LabelEncoder()
        le_tool.fit(X_train[col].tolist() + X_test[col].tolist())
        X_train[col] = le_tool.transform(X_train[col])
        X_test[col] = le_tool.transform(X_test[col])

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_test['stack_SF'] = le.inverse_transform(y_pred)

    print("F1-score total SF:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Full Report SF:\n", classification_report(y_test, y_pred, zero_division=0))

    resumen = []
    modelos = ["maj", "wei", "stack"]
    niveles = ["class", "order", "SF"]

    for modelo in modelos:
        for nivel in niveles:
            print(df_test[f'{modelo}_{nivel}'].value_counts())
            y_true = df_test[original_cols[nivel]]
            y_pred = df_test[f'{modelo}_{nivel}'].str.upper()

            if len(y_true) == 0:
                print(f'⚠️ There is no valid data for {modelo} level {nivel}')
                continue

            resumen.append({
                'Model': modelo,
                'Level': nivel,
                'Accuracy': accuracy_score(y_true, y_pred),
                'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            })

            print(f'\n--- Model metrics {modelo} at level {nivel} ---')
            print(classification_report(y_true, y_pred, zero_division=0))

            # Create the confusion matrix
            all_classes = sorted(set(list(y_true) + list(y_pred)))
            snn_cm = confusion_matrix(y_true, y_pred, labels=all_classes)
            num_classes = len(pd.concat([y_true, y_pred]).unique())


            # Create a DataFrame from the confusion matrix
            snn_df_cm = pd.DataFrame(snn_cm, index=all_classes, columns=all_classes)

            # Visualize directly with Matplotlib
            plt.figure(figsize=(20, 14))
            plt.imshow(snn_df_cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{modelo} {nivel}')
            plt.colorbar()

            # Label the X and Y axes
            tick_marks = np.arange(len(all_classes))
            plt.xticks(tick_marks, all_classes, rotation=45, ha='right', fontsize=12)
            plt.yticks(tick_marks, all_classes, fontsize=12)

            # Add annotations to each cell
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
    df_resumen.to_excel(f'metrics_brief_ensemble_{kingdom}.xlsx', index=False)