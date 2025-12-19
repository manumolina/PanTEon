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

DL_FRAMEWORK = "tensorflow"
superf_dict = {'CLASSI/LINE/L1': 0, 'CLASSI/DIRS/DIRS': 1, 'CLASSI/LTR/LTR': 2, 'CLASSI/LTR/GYPSY': 3,
'CLASSI/LTR/LARD': 4, 'CLASSII/TIR/TIR': 5, 'CLASSI/LINE/CR1': 6, 'CLASSI/LINE/RTE': 7, 'CLASSII/HELITRON/HELITRON': 8,
'CLASSII/TIR/P': 9, 'CLASSII/TIR/PIFHARBINGER': 10, 'CLASSII/TIR/MULE': 11, 'CLASSI/LTR/COPIA': 12, 'CLASSII/TIR/HAT': 13,
'CLASSII/TIR/TC1MARINER': 14, 'CLASSI/LINE/I': 15, 'CLASSII/TIR/CACTA': 16, 'CLASSI/LINE/LINE': 17}
batch_size = 256
num_epochs = 200

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
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(X_dev, Y_dev), callbacks=[lr_scheduler, early_stopping], verbose=1)
    return history



