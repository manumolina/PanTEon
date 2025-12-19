# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import backend as K
import itertools
from tqdm import tqdm
import seaborn as sn
import shutil
import subprocess
import pickle
from shutil import copy2
import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.base import *
from sklearn.model_selection import cross_val_predict,LeaveOneOut

from copy import deepcopy
import joblib

from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ====================
# CONFIGURACIÓN GPU
# ====================
"""config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)"""

# Superfamily dict
script_dir = os.path.dirname(os.path.abspath(__file__))
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}

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


def load_data(TE_lib, num_threads, mode="T"):
    curr_dir1 = script_dir
    labels = get_data(TE_lib, curr_dir1, f"features", mode, num_threads)
    if mode == "T":
        Y = [superf_dict[x] for x in labels]
    elif mode == "P":
        Y = labels
    else:
        return None, None

    X = feature_generation_parallel(curr_dir1, f"{curr_dir1}/data/feature_file.csv", f"features",
                                    batches=num_threads, max_workers=num_threads)
    return X, np.asarray(Y)


def _write_one(rec, idx, base, nbuckets):
    b = idx % nbuckets
    d = Path(base)/f"bucket_{b}"
    d.mkdir(parents=True, exist_ok=True)
    p = d/f"seq_{idx}.fasta"
    with open(p, "w") as f:
        SeqIO.write([rec], f, "fasta")
    # devuelve la ruta y la etiqueta en el **mismo** índice
    return str(p)


def get_data(fasta_file, curr_dir1, feature_dir, mode, threads):
    """
    Escribe un archivo por secuencia (seq{i}.fasta) en paralelo y genera list.txt
    conservando el orden de las etiquetas. Compatible con el pipeline actual.
    SIN usar os.chdir().
    """
    feature_destpath = os.path.join(curr_dir1, feature_dir)
    kanalyzer_destpath = os.path.join(feature_destpath, "kanalyze-2.0.0", "code")
    kanalyzer_input_destpath = os.path.join(feature_destpath, "kanalyze-2.0.0", "input_data")
    kanalyzer_output_destpath = os.path.join(feature_destpath, "kanalyze-2.0.0", "output_data")

    # Limpia y crea input/output como hacía tu versión original
    if os.path.isdir(kanalyzer_input_destpath):
        subprocess.run(["rm", "-R", kanalyzer_input_destpath], check=False)
    os.makedirs(kanalyzer_input_destpath, exist_ok=True)

    if os.path.isdir(kanalyzer_output_destpath):
        subprocess.run(["rm", "-R", kanalyzer_output_destpath], check=False)
    os.makedirs(kanalyzer_output_destpath, exist_ok=True)

    records = list(SeqIO.parse(fasta_file, "fasta"))
    n = len(records)

    # Etiquetas por secuencia en el mismo orden que records
    if mode == "T":
        labels = [rec.id.split("#")[1].split(" ")[0] for rec in records]
    elif mode == "P":
        labels = [rec.id for rec in records]
    else:
        labels = []

    # Función de escritura de un archivo por secuencia
    def _write_one(idx_rec):
        idx, rec = idx_rec
        out_path = os.path.join(kanalyzer_input_destpath, f"seq{idx}.fasta")
        with open(out_path, "w") as of:
            # Mantiene formato fasta estándar igual que antes
            SeqIO.write([rec], of, "fasta")
        return out_path

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(_write_one, ((i, r) for i, r in enumerate(records, start=1))))

    # Genera list.txt EN input_data, sin cambiar el cwd
    input_list_path = os.path.join(kanalyzer_input_destpath, "list.txt")

    # Orden determinista: seq1.fasta ... seqN.fasta
    files = [f"seq{i}.fasta" for i in range(1, n + 1)]
    with open(input_list_path, "w") as ff:
        for f in files:
            ff.write(f + "\n")

    # Copia list.txt a las dos ubicaciones que espera el pipeline
    shutil.copy2(input_list_path, os.path.join(feature_destpath, "list.txt"))
    shutil.copy2(input_list_path, os.path.join(kanalyzer_destpath, "list.txt"))

    # Limpia el list.txt temporal en input_data para dejar el estado como antes
    try:
        os.remove(input_list_path)
    except FileNotFoundError:
        pass

    return labels


def feature_generation_parallel(curr_dir1, output_file, feature_dir, batches, max_workers):
    """
    Paraleliza runKanalyzer_generate_all_features por lotes, fusiona outputs y
    ejecuta una única vez KmersFeaturesCollector para generar feature_file.csv.
    Mantiene la forma X: [n_secuencias, n_features].
    SIN usar os.chdir().
    """
    import concurrent.futures
    from pathlib import Path
    import shutil
    import subprocess
    import os
    import pandas as pd

    # --- Rutas base (idénticas a tu feature_generation) ---
    feature_destpath = os.path.join(curr_dir1, feature_dir)
    base_kanalyzer_dir = Path(feature_destpath) / "kanalyze-2.0.0"
    base_code_dir = base_kanalyzer_dir / "code"
    base_input_dir = base_kanalyzer_dir / "input_data"
    base_output_dir = base_kanalyzer_dir / "output_data"

    # Limpia mer dirs base
    mer_dirs = [base_output_dir / "2mer", base_output_dir / "3mer", base_output_dir / "4mer"]
    for md in mer_dirs:
        if md.exists():
            subprocess.run(["rm", "-R", str(md)], check=False)
        md.mkdir(parents=True, exist_ok=True)

    # --- Leemos la lista completa de entradas (en el mismo orden que etiquetas) ---
    list_txt_path = Path(feature_destpath) / "list.txt"
    with open(list_txt_path, "r") as f:
        all_files = [ln.strip() for ln in f if ln.strip()]

    if len(all_files) == 0:
        raise RuntimeError("list.txt vacío: no hay entradas para generar features")

    # Ajuste de lotes: no más lotes que archivos
    batches = max(1, min(batches, len(all_files)))
    if max_workers is None:
        max_workers = batches

    # Particionamos en lotes balanceados (conservando orden)
    def chunks(seq, n):
        k = (len(seq) + n - 1) // n
        for i in range(0, len(seq), k):
            yield seq[i:i + k]

    batches_list = list(chunks(all_files, batches))

    # --- Preparar replicas aisladas de kanalyze para correr en paralelo ---
    clones = []
    clones_root = Path(feature_destpath) / "kanalyze_parallel_work"
    if clones_root.exists():
        shutil.rmtree(clones_root)
    clones_root.mkdir(parents=True, exist_ok=True)

    for i, files_batch in enumerate(batches_list):
        clone_dir = clones_root / f"kanalyze-2.0.0_batch_{i}"
        shutil.copytree(base_kanalyzer_dir, clone_dir)

        clone_code = clone_dir / "code"
        clone_input = clone_dir / "input_data"
        clone_output = clone_dir / "output_data"

        # Limpiamos output del clone
        for sub in ["2mer", "3mer", "4mer"]:
            subdir = clone_output / sub
            subdir.mkdir(parents=True, exist_ok=True)
            for child in subdir.glob("*"):
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)

        # Preparar input: symlinks a los FASTA reales en el batch (fallback: copia)
        for p in clone_input.glob("*"):
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

        for fname in files_batch:
            src = base_input_dir / fname
            dst = clone_input / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

        # list.txt del lote (en code del clone)
        with open(clone_code / "list.txt", "w") as lf:
            for fname in files_batch:
                lf.write(fname + "\n")

        clones.append((i, clone_dir, files_batch))

    # --- Ejecutar runKanalyzer_generate_all_features en paralelo ---
    def _run_clone(args):
        idx, clone_dir, _files = args
        code_dir = clone_dir / "code"
        # Importante: sin chdir, usamos cwd=...
        subprocess.run(["chmod", "775", "runKanalyzer_generate_all_features"],
                       cwd=str(code_dir), check=False)
        subprocess.run(["./runKanalyzer_generate_all_features"],
                       cwd=str(code_dir),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        return idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_run_clone, clones))

    # --- Fusionar salidas de todos los clones en el output base ---
    for sub in ["2mer", "3mer", "4mer"]:
        dst_dir = base_output_dir / sub
        for i, clone_dir, _ in clones:
            src_dir = clone_dir / "output_data" / sub
            if not src_dir.exists():
                continue

            for item in sorted(src_dir.iterdir()):
                target = dst_dir / item.name
                if target.exists():
                    target = dst_dir / f"b{i}_{item.name}"

                if item.is_file():
                    shutil.copy2(item, target)
                else:
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)

    # --- Ejecutar KmersFeaturesCollector una sola vez (igual que antes) ---
    # Tu versión compilaba/ejecutaba "en feature_destpath". Lo hacemos con cwd=feature_destpath.
    subprocess.run(["javac", "KmersFeaturesCollector.java"], cwd=feature_destpath, check=False)
    subprocess.run(["javac", "BufferReaderAndWriter.java"], cwd=feature_destpath, check=False)
    subprocess.run(["java", "KmersFeaturesCollector"], cwd=feature_destpath, check=False)

    # El java genera feature_file.csv en feature_destpath (como antes cuando hacías chdir allí)
    generated_csv = os.path.join(feature_destpath, "feature_file.csv")
    shutil.copy2(generated_csv, output_file)

    # Limpia el CSV temporal (igual que tu rm)
    try:
        os.remove(generated_csv)
    except FileNotFoundError:
        pass

    # --- Limpieza opcional de los clones ---
    shutil.rmtree(clones_root, ignore_errors=True)

    # --- Leer el CSV final y devolver numpy como antes ---
    df = pd.read_csv(output_file)
    return df.to_numpy()


def get_model():
    estimators = [
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, algorithm='auto'))
        ])),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('svm_rbf', SVC(
                C=512,
                gamma=0.0078125,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])),
        ('et', Pipeline([
            ('scaler', StandardScaler()),
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=1000,
                max_depth=8,
                class_weight='balanced',
                random_state=42
            ))
        ]))
    ]
    meta_classifier = Pipeline([('scaler', preprocessing.StandardScaler()), ('Log_Reg',
                                                                             LogisticRegression(solver='lbfgs',
                                                                                                multi_class='multinomial',
                                                                                                class_weight="balanced",
                                                                                                max_iter=12000,
                                                                                                n_jobs=-1))])

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_classifier,
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1
    )
    return clf


def run_experiment(model, X_train, Y_train):
    model.fit(X_train, Y_train)


def metrics(Y_validation,predictions, num_classes):

    classes = len(np.unique(Y_validation))
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
    plt.savefig('Train_Curve_ClassifyTE.png', bbox_inches='tight', dpi=500)

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
    plt.savefig('Train_Curve_los_ClassifyTE.png', bbox_inches='tight', dpi=500)


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

        os.makedirs("data_for_training/", exist_ok=True)
        X, Y = load_data(TE_library, 1, script_mode)

        ##########################
        # 0. Save the data
        np.save("data_for_training/X.npy", X)
        np.save("data_for_training/Y.npy", Y)

        """X = np.load("data_for_training/X.npy")
        Y = np.load("data_for_training/Y.npy")"""

        num_classes = int(np.max(Y) + 1)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. data split: 80% train, 10% dev and 10% test
        print("### Step 1: Starting the dataset spliting ......")
        start = time.time()

        validation_size = 0.2
        seed = 7
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y)
        X_dev, X_test, Y_dev, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed, stratify=Y_temp)

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

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ##########################
        # 2. Preprocess input data

        ###########################
        # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices

        ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        batch_size = 512
        num_epochs = 100
        model = get_model()
        run_experiment(model, X_train, Y_train)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        joblib.dump(model, "trained_models/ClassifyTE_retrained_model.pkl")

        end = time.time()
        print(f"### Step 5 Done !! [{end - start}]......")

        ###########################
        # 6. Training report
        # plot_training_metrics(history)

        ###########################
        # 7. Testing report
        print("### Step 7: Creating the testing reports ......")
        start = time.time()

        predicted_classes = model.predict(X_test)
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
        os.makedirs("data_for_training/", exist_ok=True)
        X, labels = load_data(TE_library, 1, script_mode)
        labels = labels.tolist()

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data


        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        model = joblib.load("trained_models/ClassifyTE_retrained_model.pkl")

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Predict the labels
        print("### Step 3: Starting to predict the TE classification......")
        start = time.time()

        y_preds_probs = model.predict_proba(X)

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
