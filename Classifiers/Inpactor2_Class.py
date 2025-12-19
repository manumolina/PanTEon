# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
import random
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tqdm import tqdm
import seaborn as sn
import time
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from itertools import product

# ====================
# CONFIGURACIÓN GPU
# ====================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# Superfamily dict
"""superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}"""

superf_dict = {'CLASSI/LINE/L1': 0, 'CLASSI/DIRS/DIRS': 1, 'CLASSI/LTR/LTR': 2, 'CLASSI/LTR/GYPSY': 3,
'CLASSI/LTR/LARD': 4, 'CLASSII/TIR/TIR': 5, 'CLASSI/LINE/CR1': 6, 'CLASSI/LINE/RTE': 7, 'CLASSII/HELITRON/HELITRON': 8,
'CLASSII/TIR/P': 9, 'CLASSII/TIR/PIFHARBINGER': 10, 'CLASSII/TIR/MULE': 11, 'CLASSI/LTR/COPIA': 12, 'CLASSII/TIR/HAT': 13,
'CLASSII/TIR/TC1MARINER': 14, 'CLASSI/LINE/I': 15, 'CLASSII/TIR/CACTA': 16, 'CLASSI/LINE/LINE': 17}

# ====================
# MÉTRICAS PERSONALIZADAS
# ====================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_data(fasta_path, mode="T"):
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    k_range = (1, 6)

    all_kmer_set = set()
    k_min, k_max = k_range
    a = tuple(("A", "C", "G", "T"))  # tupla para acceso rápido
    for k in range(k_min, k_max + 1):
        for tup in product(a, repeat=k):
            all_kmer_set.add(''.join(tup))
    all_kmers = sorted(all_kmer_set)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    if mode == "T":
        counts_matrix = np.zeros((len(sequences), len(all_kmers) + 1), dtype=int)
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting k-mers")):
            classification = seq.id.split(" ")[0].split("#")[1]
            if classification in superf_dict:
                order = superf_dict[classification]
                counts_matrix[seq_idx, 0] = order
                seq_str = str(seq.seq)
                for k in range(k_range[0], k_range[1] + 1):
                    for i in range(len(seq_str) - k + 1):
                        kmer = seq_str[i:i + k].upper()
                        if kmer in kmer_index:
                            counts_matrix[seq_idx, kmer_index[kmer] + 1] += 1
            else:
                print(f"[ERROR] {classification} not found in our superfamily dictionary")

        Y = counts_matrix[:, 0]
        X = counts_matrix[:, 1:]

        return X, Y
    elif mode == "P":
        counts_matrix = np.zeros((len(sequences), len(all_kmers)), dtype=int)
        labels_TEs = []
        for seq_idx, seq in enumerate(tqdm(sequences, desc="Contando k-mers")):
            labels_TEs.append(seq.id)
            seq_str = str(seq.seq)
            for k in range(k_range[0], k_range[1] + 1):
                for i in range(len(seq_str) - k):
                    kmer = seq_str[i:i + k].upper()
                    if kmer in kmer_index:
                        counts_matrix[seq_idx, kmer_index[kmer]] += 1
        return counts_matrix, labels_TEs
    else:
        return None, None


def get_model(shape, num_classes):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(shape,))
    x = tf.keras.layers.Dense(200, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[f1_m])
    return model


def run_experiment(model, X_train, Y_train, labels, X_dev, Y_dev, batch_size, num_epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)
    # w = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    # class_weight = {i: w_i for i, w_i in enumerate(w)}
    # class_weight=class_weight,
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(X_dev, Y_dev), callbacks=[lr_scheduler, early_stopping], verbose=1)
    return history


def metrics(Y_validation, predictions, num_classes):

    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #Creamos la matriz de confusión
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusión
    snn_df_cm = pd.DataFrame(snn_cm, range(num_classes), range(num_classes))
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.savefig('confusionMatrix_DeepTE.png', bbox_inches='tight', dpi=500)


def plot_training_metrics(history):
    # plot metrics
    plt.figure()
    plt.plot(history.history['val_f1_m'])
    plt.plot(history.history['f1_m'])
    plt.legend(['val_f1_m', 'train_f1_m'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('f1_m')
    plt.title('Epoch vs f1_m')
    plt.savefig('Train_Curve_Inp2_Class.png', bbox_inches='tight', dpi=500)

    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.legend(['val_loss', 'train_loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Epoch vs Loss')

    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.legend(['val_loss', 'train_loss'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Epoch vs loss')
    plt.savefig('Train_Curve_los_Inp2_Class.png', bbox_inches='tight', dpi=500)


# ====================
# MAIN
# ====================
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"[ERROR] Parameter TE_library.fasta is required.")
        print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta [script_mode]")
        sys.exit(1)
    else:
        TE_library = sys.argv[1]
        if len(sys.argv) > 2:
            script_mode = sys.argv[2].upper()
            if script_mode not in ['T', 'P']:
                print(f"[ERROR] script_mode should be T or P, found {script_mode} instead.")
                print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta [script_mode]")
                sys.exit(1)
        else:
            print("[INFO] Using training script mode by default")
            script_mode = "T"

    if script_mode == "T":
        start_all = time.time()
        print("### Step 0: Starting to load and transform the dataset......")
        start = time.time()
        os.makedirs("trained_models/", exist_ok=True)
        X, Y = load_data(TE_library)

        ##########################
        # 0. Save the data
        os.makedirs("data_for_training/", exist_ok=True)
        np.save("data_for_training/X.npy", X)
        np.save("data_for_training/Y.npy", Y)

        """X = np.load("data_for_training/X.npy")
        Y = np.load("data_for_training/Y.npy")"""

        num_classes = int(np.max(Y) + 1)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. data split: 80% train, 10% dev and 10% test
        print("### Step 1: Starting the dataset splitting ......")
        start = time.time()

        validation_size = 0.2
        seed = 7
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y)
        X_dev, X_test, Y_dev, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed, stratify=Y_temp)

        # Space optimization
        X_train = X_train.astype('float32')
        X_dev = X_dev.astype('float32')
        X_test = X_test.astype('float32')

        print("\nDataset shapes:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_dev shape: {X_dev.shape}")
        print(f"X_test shape: {X_test.shape}")

        print("\nLabel information:")
        print(f"Shape of Y_train: {Y_train.shape}")
        print(f"Shape of Y_dev: {Y_dev.shape}")
        print(f"Shape of Y_test: {Y_test.shape}")

        print(f"\nNumber of unique classes in Y_train: {len(np.unique(Y_train))}")
        print(f"Number of unique classes in Y_dev: {len(np.unique(Y_dev))}")
        print(f"Number of unique classes in Y_test: {len(np.unique(Y_test))}")

        print("\nClasses distribution in Y_train:")
        unique, counts = np.unique(Y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")

        print("\nClasses distribution in Y_dev:")
        unique, counts = np.unique(Y_dev, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")

        print("\nClasses distribution in Y_test:")
        unique, counts = np.unique(Y_test, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Class {cls}: {count} samples")

        # Optimizing memory usage
        X = None
        Y = None

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ##########################
        # 2. Preprocess input data
        print("### Step 2: Starting the features preprocessing steps ......")
        start = time.time()

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, "trained_models/scaler.pkl")

        pca = decomposition.PCA(n_components=0.96, svd_solver='full')
        pca.fit(X_scaled)
        X_train_pca = pca.transform(X_scaled)
        X_dev_pca = pca.transform(X_dev_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        joblib.dump(pca, "trained_models/pca.pkl")

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")
        ###########################
        # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
        print("### Step 3: Starting the labels preprocessing steps ......")
        start = time.time()

        Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)
        Y_dev_cat = tf.keras.utils.to_categorical(Y_dev, num_classes)
        Y_test_cat = tf.keras.utils.to_categorical(Y_test, num_classes)

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        batch_size = 512
        num_epochs = 200
        model = get_model(X_train_pca.shape[1], num_classes)
        tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        history = run_experiment(model, X_train_pca, Y_train_cat, Y_train, X_dev_pca, Y_dev_cat, batch_size=batch_size, num_epochs=num_epochs)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        model.save('trained_models/Inpactor2_Classify_retrained_model.h5')

        end = time.time()
        print(f"### Step 5 Done !! [{end - start}]......")

        ###########################
        # 6. Training report
        print("### Step 6: Creating the training reports ......")
        start = time.time()

        plot_training_metrics(history)

        end = time.time()
        print(f"### Step 6 Done !! [{end - start}]......")

        ###########################
        # 7. Testing report
        print("### Step 7: Creating the testing reports ......")
        start = time.time()

        predicted_classes = model.predict(X_test_pca)
        predicted_classes = np.argmax(predicted_classes, axis=1)
        metrics(Y_test, predicted_classes, num_classes)

        end = time.time()
        print(f"### Step 7 Done !! [{end - start}]......")

        end_all = time.time()
        print(f"[INFO] Training process successfully complete. Total time={end_all - start_all} seconds. ")

    elif script_mode == "P":

        ##########################
        # 0. Load and transform the data
        start_all = time.time()
        print("### Step 0: Starting to load and transform the dataset......")
        start = time.time()
        X, labels = load_data(TE_library, mode=script_mode)
        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data
        print("### Step 1: Starting the preprocessing step......")
        start = time.time()
        scaler = joblib.load("trained_models/scaler.pkl")
        X_scaled = scaler.transform(X)

        pca = joblib.load("trained_models/pca.pkl")
        X_scaled_pca = pca.transform(X_scaled)

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        model = load_model("trained_models/Inpactor2_Classify_retrained_model.h5")

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Predict the labels
        print("### Step 3: Starting to predict the TE classification......")
        start = time.time()

        y_preds_probs = model.predict(X_scaled_pca)

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Save results in fasta and in csv
        print("### Step 4: Starting to save the results......")
        start = time.time()

        inv_superf_dict = {value: key for key, value in superf_dict.items()}
        y_pred_idx = y_preds_probs.argmax(axis=1)
        y_pred_label = [inv_superf_dict[i] for i in y_pred_idx]
        prob_of_pred = y_preds_probs[np.arange(len(y_pred_idx)), y_pred_idx]
        df = pd.DataFrame(
            {
                "id": labels,
                "predicted_class": y_pred_label,
                "probability": prob_of_pred,
            }
        )
        df.to_csv("classification_prediction.csv", index=False)

        min_prob = 0.0
        final_seqs = []
        for TE in SeqIO.parse(TE_library, "fasta"):
            # remove previous classification if any
            original_name = TE.id.split("#")[0]
            position = labels.index(TE.id)
            new_classification = y_pred_label[position] if prob_of_pred[position] >= min_prob else "Unknown"
            TE.id = original_name + "#" + new_classification
            if len(TE.description.split(" ")) > 1:
                complement = " ".join(TE.description.split(" ")[1:])
                TE.id += " " + complement
            TE.description = ""
            final_seqs.append(TE)
        SeqIO.write(final_seqs, "classification_prediction.fasta", "fasta")

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        end_all = time.time()
        print(f"[INFO] Inference process successfully complete. Total time={end_all - start_all} seconds. ")

    else:
        print(f"[ERROR] script_mode should be T or P, found {script_mode} instead.")
        print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta [script_mode]")
        sys.exit(1)
