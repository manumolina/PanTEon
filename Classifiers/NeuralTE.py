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
from tensorflow.keras.callbacks import ModelCheckpoint

import random
import re
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from openpyxl.utils import get_column_letter
from pandas import ExcelWriter
from tensorflow.keras.utils import to_categorical

import atexit
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D
import time


##############################################
current_folder = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_folder, "..")

# 1. Data preprocessing parameters
## Whether to use corresponding features for classification, all of which have been proven helpful for classification.
use_kmers = 1   # Whether to use k-mer feature
use_terminal = 1    # Whether to use LTR and TIR features
use_TSD = 0     # Whether to use TSD feature
use_domain = 1  # Whether to use TE domain feature
use_ends = 1    # Whether to use 5-bp ends feature


use_minority = 0 # Whether to use minority samples to correct results
is_train = 0  # Whether it is in the model training stage
keep_raw = 0  # Whether to retain the raw input sequence, 1 yes, 0 no, only save species having TSDs
only_preprocess = 0 # Whether to only perform data preprocessing
is_predict = 1  # Enable prediction mode. Setting to 0 requires the input FASTA file to be in Repbase format (seq_name\tLabel\tspecies).
is_wicker = 1   # Use Wicker classification labels. Setting to 0 will output RepeatMasker classification labels.
is_plant = 0 # Is the input genome of a plant? 0 represents non-plant, while 1 represents plant.
is_debug = 0 # Is debug mode


# 2. Program and model parameters
internal_kmer_sizes = [1, 3]   # Size of k-mer used for converting internal sequences to k-mer frequency features
terminal_kmer_sizes = [1, 2, 3] # Size of k-mer used for converting terminal sequences to k-mer frequency features
## CNN model parameters
cnn_num_convs = 3 # Number of CNN convolutional layers
cnn_filters_array = [16, 16, 16] # Number of filters per convolutional layer in CNN
cnn_kernel_sizes_array = [7, 7, 7] # Kernel size for each convolutional layer in CNN; for 2D convolutional layers, set as [(3, 3), ...]
cnn_dropout = 0.5 # CNN dropout threshold
## Training parameters
batch_size = 32 # Batch size for training
epochs = 50 #  Number of epochs for training
use_checkpoint = 0  # Whether to use checkpoint training; set to 1 to resume training from the parameters of the previous failed training session, avoiding training from scratch


################################################### The following parameters do not need modification ######################################################################
version_num = '1.0.1'
work_dir = project_dir + '/work' # temp work directory

non_temp_files = ['classified\.info', 'classified_TE\.fa', '.*\.domain']

# minority sample labels
#minority_labels_class = {'Crypton': 0, '5S': 1, '7SL': 2, 'Merlin': 3, 'P': 4, 'R2': 5, 'Unknown': 6}
minority_labels_class = {'Crypton': 0, '5S': 1, 'Merlin': 2, 'P': 3, 'R2': 4, 'Unknown': 5}

## Superfamily labels based on Wicker classification original NeuralTE
all_wicker_class_original = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                    'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                    'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                    'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26, '5S': 27, 'Unknown': 28,
                    # New added superfamilies:
                    'LINE': 29, 'LTR': 30, 'SINE': 31, 'TIR': 32}

all_wicker_class = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
                   'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
                   'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
                   'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
                   'KOLOBOK': 28, 'ACADEM-1': 29, 'Unknown': 30}


## Augmentation for each Repbase data
expandClassNum = {'Merlin': 20, 'Transib': 10, 'P': 10, 'Crypton': 10, 'Penelope': 5, 'R2': 20, 'RTE': 8, 'Jockey': 10, 'I': 10}

class_num = len(all_wicker_class)
inverted_all_wicker_class = {value: key for key, value in all_wicker_class.items()}
# Maximum length of TSD (Target Site Duplication)
max_tsd_length = 15
# Obtain CNN input dimensions
X_feature_len = 0
# Dimensions of TE terminal and internal sequences
if use_kmers != 0:
    for kmer_size in internal_kmer_sizes:
        X_feature_len += pow(4, kmer_size)
    if use_terminal != 0:
        for i in range(2):
            for kmer_size in terminal_kmer_sizes:
                X_feature_len += pow(4, kmer_size)
if use_TSD != 0:
    X_feature_len += max_tsd_length * 4 + 1
# if use_minority != 0:
#     X_feature_len += len(minority_labels_class)
if use_domain != 0:
    X_feature_len += len(all_wicker_class_original)
if use_ends != 0:
    X_feature_len += 10 * 4
###########################################################################

# ====================
# CONFIGURACIÓN GPU
# ====================
"""config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)"""

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


def get_model(work_dir, num_features, class_num):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = work_dir + "/ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # construct model
    os.system('cd ' + checkpoint_dir + ' && rm -rf ckpt*')
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    # Open a strategy scope and create/restore the model
    with strategy.scope():
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return load_model(latest_checkpoint)
        print("Creating a new model")

        # CNN model
        # input layer
        input_layer = Input(shape=(num_features, 1))
        conv_input_layer = input_layer
        # Create multiple convolutional layers
        for i in range(cnn_num_convs):
            # Add convolutional layers
            conv = Conv1D(cnn_filters_array[i], cnn_kernel_sizes_array[i], activation='relu')(conv_input_layer)
            conv_input_layer = conv
        dropout1 = Dropout(0.5)(conv_input_layer)
        # Add flattening and fully connected layers
        flatten = Flatten()(dropout1)
        dense1 = Dense(128, activation='relu')(flatten)
        # dropout2 = Dropout(config.cnn_dropout)(dense1)
        # Output layer
        output_layer = Dense(int(class_num), activation='softmax')(dense1)
        # Build the model
        model = Model(inputs=input_layer, outputs=output_layer)
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])
        #model.compile(loss=[categorical_focal_loss(alpha=0.25, gamma=2)], optimizer='adam', metrics=['accuracy'])
        # Print model summary
        #model.summary()

    return model


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(X_dev, Y_dev), callbacks=[lr_scheduler, early_stopping], verbose=1)
    return history, None


def metrics(Y_validation,predictions, num_classes):

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
    plt.savefig('confusionMatrix.png', bbox_inches='tight', dpi=500)


def load_data(internal_kmer_sizes, terminal_kmer_sizes, data_path, work_dir, project_dir, threads):
    # Copy input files to the working directory
    if not os.path.exists(data_path):
        print('Input file not exist: ' + data_path)
        exit(-1)
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy2(data_path, work_dir)
    genome_info_path = work_dir + '/genome.info'
    data_path = work_dir + '/' + os.path.basename(data_path)
    domain_train_path = data_path + '.domain'

    minority_temp = work_dir + '/minority'
    if not os.path.exists(minority_temp):
        os.makedirs(minority_temp)
    minority_train_path = minority_temp + '/train.minority.ref'
    minority_out = minority_temp + '/train.minority.out'

    data_path = preprocess_data(data_path, domain_train_path, minority_train_path, minority_out, work_dir,
                                     project_dir, project_dir+"/tools/", threads, 0)

    X, Y, seq_names, labels = load_repbase_with_TSD(data_path, domain_train_path, minority_train_path, minority_out,
                                            all_wicker_class_original, project_dir + '/data/TEClasses.tsv')

    X, Y = generate_feature_mats(X, Y, seq_names, minority_labels_class, all_wicker_class,
                                 internal_kmer_sizes, terminal_kmer_sizes, threads, all_wicker_class_original)

    # Reshape data into the format accepted by the model
    X = X.reshape(X.shape[0], X_feature_len, 1).astype('float32', copy=False)
    return X, Y, seq_names, data_path, labels

def preprocess_data(data, domain_train_path, minority_train_path, minority_out, work_dir, project_dir,
                    tool_dir, threads, is_train):
    # Delete previous run's retained results
    SegLTR2intactLTRMap = work_dir + '/segLTR2intactLTR.map'
    os.system('rm -f ' + SegLTR2intactLTRMap)

    generate_domain_info(data, project_dir + '/data/RepeatPeps.lib', work_dir, threads)
    generate_minority_info(data, minority_train_path, minority_out, threads, is_train)
    data = generate_terminal_info(data, work_dir, tool_dir, threads)
    return data


##################### utils.data_util.py
##word_seq generates eg. ['AA', 'AT', 'TC', 'CG', 'GT']
def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)

def generate_kmer_dic(repeat_num):
    ##initiate a dic to store the kmer dic
    ##kmer_dic = {'ATC':0,'TTC':1,...}
    kmer_dic = {}
    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        #print(eachitem)
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0

    return (kmer_dic)

def generate_mat(words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1
    num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])
    return (num_list)

# fuera de la función, una sola vez:
BASE_TO_INT = np.frombuffer(bytearray(b"AGCT"), dtype=np.uint8)  # 'A','G','C','T'
MAP = {ord('A'):0, ord('G'):1, ord('C'):2, ord('T'):3}

def encode_seq_np(seq):
    a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    out = np.full(a.shape, 255, dtype=np.uint8)  # 255 = N/otros
    for ch, val in MAP.items():
        out[a == ch] = val
    return out

def kmer_counts_fast(enc, k):
    # devuelve conteo en orden lexicográfico A,G,C,T conforme a tu generate_kmer_dic
    if enc.size < k:
        return np.zeros(4**k, dtype=np.int32)
    # rolling hash base-4
    power = 4**(k-1)
    idx = enc[:k].copy()
    if (idx==255).any():       # si hay N en la primera ventana
        cur = -1
    else:
        cur = 0
        for v in idx: cur = cur*4 + v
    counts = np.zeros(4**k, dtype=np.int32)
    bad = 0
    for i in range(k, enc.size+1):
        if cur >= 0:
            counts[cur] += 1
        if i == enc.size: break
        left = enc[i-k]
        right = enc[i]
        if right == 255:
            bad = k
        elif bad > 0:
            bad -= 1
        if bad > 0 or left == 255:
            cur = -1
        else:
            if cur < 0:
                # recomputa ventana
                window = enc[i-k+1:i+1]
                if (window==255).any():
                    cur = -1
                else:
                    cur = 0
                    for v in window: cur = cur*4 + v
            else:
                cur = (cur - left*power)*4 + right
    return counts

def get_batch_kmer_freq_v1(grouped_x, internal_kmer_sizes, terminal_kmer_sizes, minority_labels_class, all_wicker_class_original):
    """
        Versión optimizada: cuenta k-mers vectorizada (orden A,G,C,T), preasigna el vector final
        y mantiene idénticos TSD / ends / domains. Devuelve {seq_name: np.ndarray}.
        """

    # ==== Helpers ====
    BASE_ORDER = 'AGCT'  # ¡igual que generate_kmer_dic!
    BASE_MAP = {ord(b): i for i, b in enumerate(BASE_ORDER)}  # 'A':0,'G':1,'C':2,'T':3
    BAD = 255

    def encode_seq_np(seq: str) -> np.ndarray:
        a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        out = np.full(a.shape, BAD, dtype=np.uint8)
        for ch, val in BASE_MAP.items():
            out[a == ch] = val
        return out

    def kmer_counts_fast(enc: np.ndarray, k: int) -> np.ndarray:
        """Cuenta k-mers (stride=1) ignorando ventanas con N/otros.
		El orden de salida coincide con itertools.product(['A','G','C','T'], repeat=k).
		"""
        L = enc.size
        if L < k:
            return np.zeros(4 ** k, dtype=np.int32)
        power = 4 ** (k - 1)
        counts = np.zeros(4 ** k, dtype=np.int32)
        w = enc[:k]
        if (w == BAD).any():
            cur = -1
            bad = k
        else:
            cur = 0
            for v in w: cur = cur * 4 + int(v)
            bad = 0
        for i in range(k, L + 1):
            if cur >= 0:
                counts[cur] += 1
            if i == L:
                break
            left = int(enc[i - k]);
            right = int(enc[i])
            if right == BAD:
                bad = k
            elif bad > 0:
                bad -= 1
            if bad > 0 or left == BAD:
                cur = -1
                if bad == 0:
                    w = enc[i - k + 1:i + 1]
                    if (w == BAD).any():
                        cur = -1
                        bad = k
                    else:
                        cur = 0
                        for v in w: cur = cur * 4 + int(v)
            else:
                cur = (cur - left * power) * 4 + right
        return counts

    # ==== Longitudes para preasignar ====
    global use_kmers, use_terminal, use_TSD, use_ends, use_domain, max_tsd_length
    internal_len = sum(4 ** k for k in internal_kmer_sizes) if use_kmers else 0
    term_len = (2 * sum(4 ** k for k in terminal_kmer_sizes)) if (use_kmers and use_terminal) else 0
    tsd_len = (1 + max_tsd_length * 4) if use_TSD else 0
    ends_len = (10 * 4) if use_ends else 0
    domain_len = len(all_wicker_class_original) if use_domain else 0
    total_len = internal_len + term_len + tsd_len + ends_len + domain_len

    group_dict = {}
    for x in grouped_x:
        seq_name = x[0]
        seq = x[1]
        TSD_seq = x[2]
        TSD_len = x[3]
        LTR_pos = x[4]
        TIR_pos = x[5]
        domain_label_set = x[6]

        # ----- Partición en internal/LTR/TIR (mismo criterio que el original) -----
        internal_seq = ''
        LTR_seq = ''
        TIR_seq = ''
        LTR_pos_str = str(LTR_pos.split(':')[1]).strip()
        TIR_pos_str = str(TIR_pos.split(':')[1]).strip()
        if LTR_pos_str == '' and TIR_pos_str == '':
            internal_seq = seq
        if TIR_pos_str != '':
            T = TIR_pos_str.split(',')
            left_TIR_start = int(T[0].split('-')[0])
            left_TIR_end = int(T[0].split('-')[1])
            right_TIR_start = int(T[1].split('-')[0])
            right_TIR_end = int(T[1].split('-')[1])
            TIR_seq = seq[left_TIR_start - 1: left_TIR_end] + seq[right_TIR_start - 1: right_TIR_end]
            internal_seq = seq[left_TIR_end: right_TIR_start - 1]
        if LTR_pos_str != '':
            L = LTR_pos_str.split(',')
            left_LTR_start = int(L[0].split('-')[0])
            left_LTR_end = int(L[0].split('-')[1])
            right_LTR_start = int(L[1].split('-')[0])
            right_LTR_end = int(L[1].split('-')[1])
            LTR_seq = seq[left_LTR_start - 1: left_LTR_end]
            internal_seq = seq[left_LTR_end: right_LTR_start - 1]

        # ----- Vector de características (preasignado) -----
        connected = np.zeros(total_len, dtype=np.int32)
        offset = 0

        if use_kmers:
            target_seq = internal_seq if use_terminal else seq
            enc = encode_seq_np(target_seq) if target_seq else np.array([], dtype=np.uint8)
            for k in internal_kmer_sizes:
                cnt = kmer_counts_fast(enc, k) if enc.size else np.zeros(4 ** k, dtype=np.int32)
                n = cnt.size
                connected[offset:offset + n] = cnt;
                offset += n

            if use_terminal:
                enc_LTR = encode_seq_np(LTR_seq) if LTR_seq else np.array([], dtype=np.uint8)
                enc_TIR = encode_seq_np(TIR_seq) if TIR_seq else np.array([], dtype=np.uint8)
                for k in terminal_kmer_sizes:
                    cnt = kmer_counts_fast(enc_LTR, k) if enc_LTR.size else np.zeros(4 ** k, dtype=np.int32)
                    n = cnt.size
                    connected[offset:offset + n] = cnt;
                    offset += n
                    cnt = kmer_counts_fast(enc_TIR, k) if enc_TIR.size else np.zeros(4 ** k, dtype=np.int32)
                    n = cnt.size
                    connected[offset:offset + n] = cnt;
                    offset += n

        if use_TSD:
            max_length = max_tsd_length
            if (TSD_seq == 'Unknown') or ('N' in TSD_seq):
                connected[offset] = max_length + 1;
                offset += 1
                encoded_TSD = np.ones((max_length, 4), dtype=np.int8)
            else:
                connected[offset] = int(TSD_len);
                offset += 1
                encoded_TSD = np.zeros((max_length, 4), dtype=np.int8)
                L = min(len(TSD_seq), max_length)
                eye = np.eye(4, dtype=np.int8)
                for i in range(L):
                    b = TSD_seq[i]
                    if b == 'A':
                        encoded_TSD[i] = eye[0]
                    elif b == 'T':
                        encoded_TSD[i] = eye[1]
                    elif b == 'C':
                        encoded_TSD[i] = eye[2]
                    elif b == 'G':
                        encoded_TSD[i] = eye[3]
            n = max_length * 4
            connected[offset:offset + n] = encoded_TSD.reshape(-1);
            offset += n

        if use_ends:
            end_seq = (seq[:5] + seq[-5:]) if len(seq) >= 10 else (seq + 'N' * (10 - len(seq)))
            eye = np.eye(4, dtype=np.int8)
            encoded_end_seq = np.zeros((10, 4), dtype=np.int8)
            for i, base in enumerate(end_seq[:10]):
                if base == 'A':
                    encoded_end_seq[i] = eye[0]
                elif base == 'T':
                    encoded_end_seq[i] = eye[1]
                elif base == 'C':
                    encoded_end_seq[i] = eye[2]
                elif base == 'G':
                    encoded_end_seq[i] = eye[3]
            n = 10 * 4
            connected[offset:offset + n] = encoded_end_seq.reshape(-1);
            offset += n

        if use_domain:
            encoder = np.zeros(domain_len, dtype=np.int32)
            for domain_label in domain_label_set:
                domain_label_num = all_wicker_class_original[domain_label]
                encoder[domain_label_num] = 1
            n = domain_len
            connected[offset:offset + n] = encoder
            offset += n

        group_dict[seq_name] = connected

    return group_dict

def split_list_into_groups(lst, group_size):
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]

def generate_feature_mats(X, Y, seq_names, minority_labels_class, all_wicker_class, internal_kmer_sizes, terminal_kmer_sizes, threads, all_wicker_class_original):
    seq_mats = {}
    jobs = []
    grouped_X = split_list_into_groups(X, 5000)

    ex = ProcessPoolExecutor(threads)

    for grouped_x in grouped_X:
        job = ex.submit(get_batch_kmer_freq_v1, grouped_x, internal_kmer_sizes, terminal_kmer_sizes, minority_labels_class, all_wicker_class_original)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_group_dict = job.result()
        seq_mats.update(cur_group_dict)

    #print(all_wicker_class)
    final_X = []
    final_Y = []
    for item in seq_names:
        seq_name = item[0]
        x = seq_mats[seq_name]
        final_X.append(x)
        label = Y[seq_name]
        #print(label)
        label_num = all_wicker_class[label]
        final_Y.append(label_num)
    return np.array(final_X), np.array(final_Y)

def replace_non_atcg(sequence):
    return re.sub("[^ATCG]", "", sequence)

def getRMToWicker(RM_Wicker_struct):
    # 3.2 Convert Dfam classification names into Wicker format.
    ## 3.2.1 This file contains the conversion between RepeatMasker category, Repbase, and Wicker category.
    rmToWicker = {}
    wicker_superfamily_set = set()
    with open(RM_Wicker_struct, 'r') as f_r:
        for i, line in enumerate(f_r):
            parts = line.split('\t')
            rm_type = parts[5]
            rm_subtype = parts[6]
            repbase_type = parts[7]
            wicker_type = parts[8]
            wicker_type_parts = wicker_type.split('/')
            # print(rm_type + ',' + rm_subtype + ',' + repbase_type + ',' + wicker_type)
            # if len(wicker_type_parts) != 3:
            #     continue
            wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
            if len(wicker_superfamily_parts) == 1:
                wicker_superfamily = wicker_superfamily_parts[0]
            elif len(wicker_superfamily_parts) > 1:
                wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
            rm_full_type = rm_type + '/' + rm_subtype
            if wicker_superfamily == 'ERV':
                wicker_superfamily = 'Retrovirus'
            rmToWicker[rm_full_type] = wicker_superfamily
            wicker_superfamily_set.add(wicker_superfamily)
    # Supplement some elements.
    rmToWicker['LINE/R2'] = 'R2'
    rmToWicker['LINE/RTE'] = 'RTE'
    rmToWicker['LTR/ERVL'] = 'Retrovirus'
    rmToWicker['LTR/Ngaro'] = 'DIRS'
    return rmToWicker

def load_repbase_with_TSD(path, domain_path, minority_train_path, minority_out, all_wicker_class_original, RM_Wicker_struct):
    rmToWicker = getRMToWicker(RM_Wicker_struct)
    domain_name_labels = {}
    if use_domain == 1 and os.path.exists(domain_path):
        # Load the domain file and read the TE-contained domain labels.
        with open(domain_path, 'r') as f_r:
            for i, line in enumerate(f_r):
                if i < 2:
                    continue
                parts = line.split('\t')
                TE_name = parts[0]
                label = parts[1].split('#')[1]
                if not rmToWicker.__contains__(label):
                    label = 'Unknown'
                else:
                    wicker_superfamily = rmToWicker[label]
                    label = wicker_superfamily
                    if not all_wicker_class_original.__contains__(label):
                        label = 'Unknown'
                if not domain_name_labels.__contains__(TE_name):
                    domain_name_labels[TE_name] = set()
                label_set = domain_name_labels[TE_name]
                label_set.add(label)

    names, contigs = read_fasta_v1(path)
    X = []
    Y = {}
    seq_names = []
    labels = []
    for name in names:
        feature_info = {}
        parts = name.split("\t")
        seq_name = parts[0].split(" ")[0]
        if "#" in parts[0].split(" ")[0]:
            label = parts[0].split(" ")[0].split("#")[1]
        else:
            label = "Unknown"
        labels.append(parts[0].split(" ")[0])
        for p_name in parts:
            if 'TSD:' in p_name:
                TSD_seq = p_name.split(':')[1]
                feature_info['TSD_seq'] = TSD_seq
            elif 'TSD_len:' in p_name:
                tsd_len_str = p_name.split(':')[1]
                if tsd_len_str == '':
                    TSD_len = 0
                else:
                    TSD_len = int(tsd_len_str)
                feature_info['TSD_len'] = TSD_len
            elif 'LTR:' in p_name:
                LTR_info = p_name
                feature_info['LTR_info'] = LTR_info
            elif 'TIR:' in p_name:
                TIR_info = p_name
                feature_info['TIR_info'] = TIR_info
        if use_TSD:
            TSD_seq = feature_info['TSD_seq']
            TSD_len = feature_info['TSD_len']
        else:
            TSD_seq = ''
            TSD_len = 0

        if use_terminal:
            LTR_info = feature_info['LTR_info']
            TIR_info = feature_info['TIR_info']
        else:
            LTR_info = 'LTR:'
            TIR_info = 'TIR:'

        if seq_name.endswith('-RC'):
            raw_seq_name = seq_name[:-3]
        else:
            raw_seq_name = seq_name
        if domain_name_labels.__contains__(raw_seq_name):
            domain_label_set = domain_name_labels[raw_seq_name]
        else:
            domain_label_set = {'Unknown'}

        seq = contigs[name]
        seq = replace_non_atcg(seq)  # undetermined nucleotides in splice
        x_feature = (seq_name, seq, TSD_seq, TSD_len, LTR_info, TIR_info, domain_label_set)
        X.append(x_feature)
        Y[seq_name] = label
        seq_names.append((seq_name, label))
    return X, Y, seq_names, labels

def split_fasta(cur_path, output_dir, num_chunks):
    split_files = []

    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    os.makedirs(output_dir)

    names, contigs = read_fasta_v1(cur_path)
    num_names = len(names)
    chunk_size = num_names // num_chunks

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else num_names
        chunk = names[chunk_start:chunk_end]
        output_path = output_dir + '/out_' + str(i) + '.fa'
        with open(output_path, 'w') as out_file:
            for name in chunk:
                seq = contigs[name]
                out_file.write('>'+name+'\n'+seq+'\n')
        split_files.append(output_path)
    return split_files

def run_command(command):
    subprocess.run(command, check=True, shell=True)

def identify_terminals(split_file, output_dir, tool_dir):
    base_file = os.path.basename(split_file)
    try:
        ltrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/ltrsearch -l 50 ' + base_file + ' > /dev/null 2>&1'
        itrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/itrsearch -i 0.7 -l 7 ' + base_file + ' > /dev/null 2>&1'
        run_command(ltrsearch_command)
        run_command(itrsearch_command)
        # os.system(ltrsearch_command)
        # os.system(itrsearch_command)
        ltr_file = split_file + '.ltr'
        tir_file = split_file + '.itr'

        # Read ltr and itr files to get the start and end positions of ltr and itr.
        ltr_names, ltr_contigs = read_fasta_v1(ltr_file)
        tir_names, tir_contigs = read_fasta_v1(tir_file)
        LTR_info = {}
        for ltr_name in ltr_names:
            parts = ltr_name.split(' ')
            orig_name = parts[0] + " " + parts[1]
            terminal_info = " ".join(parts[1:])
            LTR_info_parts = terminal_info.split('LTR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
            LTR_left_pos_parts = LTR_info_parts[0].split(',')
            LTR_right_pos_parts = LTR_info_parts[1].split(',')
            lLTR_start = int(LTR_left_pos_parts[0])
            lLTR_end = int(LTR_left_pos_parts[1])
            rLTR_start = int(LTR_right_pos_parts[0])
            rLTR_end = int(LTR_right_pos_parts[1])
            LTR_info[orig_name] = (lLTR_start, lLTR_end, rLTR_start, rLTR_end)
        TIR_info = {}
        for tir_name in tir_names:
            parts = tir_name.split(' ')
            orig_name = parts[0] + " " + parts[1]
            terminal_info = " ".join(parts[1:])
            TIR_info_parts = terminal_info.split('ITR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
            TIR_left_pos_parts = TIR_info_parts[0].split(',')
            TIR_right_pos_parts = TIR_info_parts[1].split(',')
            lTIR_start = int(TIR_left_pos_parts[0])
            lTIR_end = int(TIR_left_pos_parts[1])
            rTIR_start = int(TIR_right_pos_parts[1])
            rTIR_end = int(TIR_right_pos_parts[0])
            TIR_info[orig_name] = (lTIR_start, lTIR_end, rTIR_start, rTIR_end)

        # Update the header of the split_file, adding two columns LTR:1-206,4552-4757 TIR:1-33,3869-3836.
        update_split_file = split_file + '.updated'
        update_contigs = {}
        names, contigs = read_fasta_v1(split_file)
        for name in names:
            orig_name = name
            LTR_str = 'LTR:'
            if LTR_info.__contains__(orig_name):
                lLTR_start, lLTR_end, rLTR_start, rLTR_end = LTR_info[orig_name]
                LTR_str += str(lLTR_start) + '-' + str(lLTR_end) + ',' + str(rLTR_start) + '-' + str(rLTR_end)
            TIR_str = 'TIR:'
            if TIR_info.__contains__(orig_name):
                lTIR_start, lTIR_end, rTIR_start, rTIR_end = TIR_info[orig_name]
                TIR_str += str(lTIR_start) + '-' + str(lTIR_end) + ',' + str(rTIR_start) + '-' + str(rTIR_end)
            update_name = name + '\t' + LTR_str + '\t' + TIR_str
            update_contigs[update_name] = contigs[name]
        store_fasta(update_contigs, update_split_file)
        #print(f"everything all right! ;) {split_file}")
        return update_split_file
    except Exception as e:
        print(f"Error processing file {split_file} .....")
        return e

def generate_terminal_info(data_path, work_dir, tool_dir, threads):
    output_dir = work_dir + '/temp'
    # Split the file into threads blocks.
    split_files = split_fasta(data_path, output_dir, threads)

    # Parallelize the identification of LTR and TIR.
    cur_update_path = data_path + '.update'
    os.system('rm -f ' + cur_update_path)
    with ProcessPoolExecutor(threads) as executor:
        futures = []
        for split_file in split_files:
            future = executor.submit(identify_terminals, split_file, output_dir, tool_dir)
            futures.append(future)
        executor.shutdown(wait=True)

        is_exit = False
        for future in as_completed(futures):
            update_split_file = future.result()
            if isinstance(update_split_file, str):
                os.system('cat ' + update_split_file + ' >> ' + cur_update_path)
            else:
                print(f"An error occurred: {update_split_file}")
                is_exit = True
                break
        if is_exit:
            print('Error occur, exit...')
            exit(1)
        else:
            shutil.move(cur_update_path, data_path)

    return data_path

def generate_domain_info(input_path, domain_path, work_dir, threads):
    output_table = input_path + '.domain'
    temp_dir = work_dir + '/domain'
    get_domain_info(input_path, domain_path, output_table, threads, temp_dir)

def generate_minority_info(train_path, minority_train_path, minority_out, threads, is_train):
    if is_train:
        minority_contigs = {}
        train_contigNames, train_contigs = read_fasta_v1(train_path)
        # 1. extract minority dataset
        for name in train_contigNames:
            label = name.split('\t')[1]
            if minority_labels_class.__contains__(label):
                minority_contigs[name] = train_contigs[name]
        store_fasta(minority_contigs, minority_train_path)
    # elif not os.path.exists(minority_train_path):
    #     print('We are currently in the model prediction step, attempting to use the minority feature. '
    #           'However, the minority data from the training set at: ' + minority_train_path + ' cannot be found. '
    #                                                                                     'Please verify if this data exists or consider setting the parameter `--use_minority 0`.')
    #     sys.exit(-1)
    #
    # # 2. conduct blastn alignment
    # blastn2Results_path = minority_out
    # os.system('makeblastdb -in ' + minority_train_path + ' -dbtype nucl')
    # align_command = 'blastn -db ' + minority_train_path + ' -num_threads ' \
    #                 + str(threads) + ' -query ' + train_path + ' -evalue 1e-20 -outfmt 6 > ' + blastn2Results_path
    # os.system(align_command)
    # return blastn2Results_path

def store2file(data_partition, cur_consensus_path):
    if len(data_partition) > 0:
        with open(cur_consensus_path, 'w') as f_save:
            for item in data_partition:
                f_save.write('>'+item[0]+'\n'+item[1]+'\n')
        f_save.close()

def PET(seq_item, partitions):
    # sort contigs by length
    original = seq_item
    original = sorted(original, key=lambda x: len(x[1]), reverse=True)
    return divided_array(original, partitions)

def divided_array(original_array, partitions):
    final_partitions = [[] for _ in range(partitions)]
    node_index = 0

    read_from_start = True
    read_from_end = False
    i = 0
    j = len(original_array) - 1
    while i <= j:
        # read from file start
        if read_from_start:
            final_partitions[node_index % partitions].append(original_array[i])
            i += 1
        if read_from_end:
            final_partitions[node_index % partitions].append(original_array[j])
            j -= 1
        node_index += 1
        if node_index % partitions == 0:
            # reverse
            read_from_end = bool(1 - read_from_end)
            read_from_start = bool(1 - read_from_start)
    return final_partitions

def get_domain_info(cons, lib, output_table, threads, temp_dir):
    if os.path.exists(temp_dir):
        os.system('rm -rf ' + temp_dir)
    os.makedirs(temp_dir)

    consensus_contignames, consensus_contigs = read_fasta_v1(cons)
    # Copy the lib to the output directory. If the current process involves
    # evaluation, then it's necessary to filter out domains from lib
    # that contain test species
    temp_lib = temp_dir + '/RepeatPeps.lib'
    shutil.copy2(lib, temp_lib)
    if is_predict == 0:
        test_species_set = set()
        for name in consensus_contignames:
            parts = name.split('\t')
            species = parts[2]
            test_species_set.add(species)
        # filter out test species from the protein library of RepeatMasker.
        lib_contigNames, lib_contigs = read_fasta_v1(lib)
        rm_contigs = {}
        for name in lib_contigNames:
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, name)
            if match:
                species = match.group(1)
            else:
                species = 'Unknown'
            # filter content in '()'
            pattern = r'\([^)]*\)'
            species = re.sub(pattern, '', species)
            species = re.sub(r'\s+', ' ', species).strip()
            if species not in test_species_set:
                rm_contigs[name] = lib_contigs[name]
        store_fasta(rm_contigs, temp_lib)

    lib = temp_lib
    blast_db_command = 'makeblastdb -dbtype prot -in ' + lib
    os.system(blast_db_command + ' > /dev/null 2>&1')
    # 1. Divide the cons, and for each block, use blastx -num_threads 1 -evalue 1e-20 to compare cons with domain.
    partitions_num = int(threads)
    data_partitions = PET(consensus_contigs.items(), partitions_num)
    merge_distance = 100
    file_list = []
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for partition_index, data_partition in enumerate(data_partitions):
        if len(data_partition) <= 0:
            continue
        cur_consensus_path = temp_dir + '/'+str(partition_index)+'.fa'
        store2file(data_partition, cur_consensus_path)
        cur_output = temp_dir + '/'+str(partition_index)+'.out'
        cur_table = temp_dir + '/' + str(partition_index) + '.tbl'
        cur_file = (cur_consensus_path, lib, cur_output, cur_table)

        job = ex.submit(multiple_alignment_blastx_v1, cur_file, merge_distance)
        jobs.append(job)
    ex.shutdown(wait=True)

    # 2. Generate a table of the best matches between query and domain.
    os.system("echo 'TE_name\tdomain_name\tTE_start\tTE_end\tdomain_start\tdomain_end\n' > " + output_table)
    is_exit = False
    for job in as_completed(jobs):
        cur_table = job.result()
        if isinstance(cur_table, str):
            os.system('cat ' + cur_table + ' >> ' + output_table)
        else:
            print(f"An error occurred: {cur_table}")
            is_exit = True
            break
    if is_exit:
        print('Error occur, exit...')
        exit(1)

def multiple_alignment_blastx_v1(repeats_path, merge_distance):
    try:
        split_repeats_path = repeats_path[0]
        protein_db_path = repeats_path[1]
        blastx2Results_path = repeats_path[2]
        cur_table = repeats_path[3]
        align_command = 'blastx -db ' + protein_db_path + ' -num_threads ' \
                        + str(1) + ' -evalue 1e-20 -query ' + split_repeats_path + ' -outfmt 6 > ' + blastx2Results_path
        # os.system(align_command)
        run_command(align_command)

        fixed_extend_base_threshold = merge_distance

        # parse blastn output, determine the repeat boundary
        # query_records = {query_name: {subject_name: [(q_start, q_end, s_start, s_end), ...] }}
        query_records = {}
        with open(blastx2Results_path, 'r') as f_r:
            for idx, line in enumerate(f_r):
                # print('current line idx: %d' % (idx))
                parts = line.split('\t')
                query_name = parts[0]
                subject_name = parts[1]
                identity = float(parts[2])
                alignment_len = int(parts[3])
                q_start = int(parts[6])
                q_end = int(parts[7])
                s_start = int(parts[8])
                s_end = int(parts[9])
                if not query_records.__contains__(query_name):
                    query_records[query_name] = {}
                subject_dict = query_records[query_name]
                if not subject_dict.__contains__(subject_name):
                    subject_dict[subject_name] = []
                cur_records = subject_dict[subject_name]
                # q_start, q_end, s_start, s_end
                if q_start <= q_end:
                    cur_records.append((q_start, q_end, s_start, s_end))
                else:
                    cur_records.append((q_end, q_start, s_start, s_end))

        # print('len(query_records): %d' % (len(query_records)))

        # remove redundant records
        keep_longest_query = {}
        for query_name in query_records.keys():
            keep_longest_query[query_name] = []

            subject_dict = query_records[query_name]
            # print('len(subject_dict): %d' % (len(subject_dict)))

            # forward and reverse respectively, cluster
            # pos --> [q_start, q_end, s_start, s_end]
            # reverse --> [q_start, q_end, s_end, s_start]
            pos_array = []
            reverse_array = []
            for subject_name in subject_dict.keys():
                cur_records = subject_dict[subject_name]
                cur_pos = []
                cur_reverse = []
                for frag in cur_records:
                    q_start = frag[0]
                    q_end = frag[1]
                    s_start = frag[2]
                    s_end = frag[3]
                    if s_start <= s_end:
                        cur_pos.append([q_start, q_end, s_start, s_end, subject_name])
                    else:
                        cur_reverse.append([q_start, q_end, s_end, s_start, subject_name])

                # sort by q_start
                cur_pos.sort(key=lambda x: (x[0], x[1]))
                cur_reverse.sort(key=lambda x: (x[0], x[1]))

                # cluster
                pos_array.append(cur_pos)
                reverse_array.append(cur_reverse)

            # print('len(pos_array): %d' % (len(pos_array)))

            merge_domains = []
            # forward strand
            for pos in pos_array:
                clusters = {}
                cluster_index = 0
                if len(pos) > 0:
                    cur_cluster = []
                    cur_cluster.append(pos[0])
                    clusters[cluster_index] = cur_cluster
                    for i in range(1, len(pos)):
                        frag = pos[i]
                        cur_cluster = clusters[cluster_index]
                        is_closed = False
                        for exist_frag in reversed(cur_cluster):
                            if (frag[0] - exist_frag[1] < fixed_extend_base_threshold):
                                is_closed = True
                                break
                        if is_closed:
                            cur_cluster.append(frag)
                        else:
                            cluster_index += 1
                            if not clusters.__contains__(cluster_index):
                                clusters[cluster_index] = []
                            cur_cluster = clusters[cluster_index]
                            cur_cluster.append(frag)

                for cluster_index in clusters.keys():
                    cur_cluster = clusters[cluster_index]
                    cur_cluster.sort(key=lambda x: (x[2], x[3]))

                    cluster_longest_query_start = -1
                    cluster_longest_query_end = -1
                    cluster_longest_subject_start = -1
                    cluster_longest_subject_end = -1
                    subject_name = ''
                    if len(cur_cluster) > 0:
                        cluster_longest_query_start = cur_cluster[0][0]
                        cluster_longest_subject_start = cur_cluster[0][2]
                        subject_name = cur_cluster[0][4]
                        for frag in cur_cluster:
                            cluster_longest_query_end = max(cluster_longest_query_end, frag[1])
                            cluster_longest_subject_end = max(cluster_longest_subject_end, frag[3])

                    if cluster_longest_query_start >= 0:
                        domain_len = cluster_longest_query_end - cluster_longest_query_start + 1
                        subject_len = cluster_longest_subject_end - cluster_longest_subject_start + 1
                        merge_domains.append([cluster_longest_query_start, cluster_longest_query_end,
                                              domain_len, cluster_longest_subject_start, cluster_longest_subject_end,
                                              subject_len, subject_name])

            # reverse strand
            for reverse_pos in reverse_array:
                clusters = {}
                cluster_index = 0
                if len(reverse_pos) > 0:
                    cur_cluster = []
                    cur_cluster.append(reverse_pos[0])
                    clusters[cluster_index] = cur_cluster
                    for i in range(1, len(reverse_pos)):
                        frag = reverse_pos[i]
                        cur_cluster = clusters[cluster_index]
                        is_closed = False
                        for exist_frag in reversed(cur_cluster):
                            if (exist_frag[1] - frag[0] < fixed_extend_base_threshold):
                                is_closed = True
                                break
                        if is_closed:
                            cur_cluster.append(frag)
                        else:
                            cluster_index += 1
                            if not clusters.__contains__(cluster_index):
                                clusters[cluster_index] = []
                            cur_cluster = clusters[cluster_index]
                            cur_cluster.append(frag)

                for cluster_index in clusters.keys():
                    cur_cluster = clusters[cluster_index]
                    cur_cluster.sort(key=lambda x: (x[2], x[3]))

                    cluster_longest_query_start = -1
                    cluster_longest_query_end = -1
                    cluster_longest_subject_start = -1
                    cluster_longest_subject_end = -1
                    subject_name = ''
                    if len(cur_cluster) > 0:
                        cluster_longest_query_start = cur_cluster[0][0]
                        cluster_longest_subject_start = cur_cluster[0][2]
                        subject_name = cur_cluster[0][4]
                        for frag in cur_cluster:
                            cluster_longest_query_end = max(cluster_longest_query_end, frag[1])
                            cluster_longest_subject_end = max(cluster_longest_subject_end, frag[3])

                    if cluster_longest_query_start >= 0:
                        domain_len = cluster_longest_query_end - cluster_longest_query_start + 1
                        subject_len = cluster_longest_subject_end - cluster_longest_subject_start + 1
                        merge_domains.append([cluster_longest_query_start, cluster_longest_query_end,
                                              domain_len, cluster_longest_subject_start, cluster_longest_subject_end,
                                              subject_len, subject_name])

            # remove redundant domains
            # keep_longest_query --> [[left, right, q_len, s_left, s_right, s_len, subject], ...]
            keep_domains = []
            merge_domains.sort(key=lambda x: (x[0], -x[1]))
            for i in range(len(merge_domains)):
                domain_i = merge_domains[i]
                is_new_domain = True
                for j in range(i):
                    domain_j = merge_domains[j]
                    left = max(domain_i[0], domain_j[0])
                    right = min(domain_i[1], domain_j[1])
                    if right >= left:
                        # if more than 50% overlapped with the previous longer one, drop it
                        overlap = right - left + 1
                        len_i = domain_i[1] - domain_i[0] + 1
                        if (overlap / len_i) > 0.5:
                            is_new_domain = False
                            break
                if is_new_domain:
                    keep_domains.append(domain_i)

            keep_longest_query[query_name] = keep_domains

        # Save table
        with open(cur_table, 'w') as f_save:
            for query_name in keep_longest_query.keys():
                domain_array = keep_longest_query[query_name]
                merge_domains = []
                for domain_info in domain_array:
                    # quitar duplicados por solape (>50%) dentro de la misma query
                    is_new_domain = True
                    for k in range(len(merge_domains)):
                        exist_domain = merge_domains[k]
                        left = max(exist_domain[0], domain_info[0])
                        right = min(exist_domain[1], domain_info[1])
                        if right >= left:
                            overlap = right - left + 1
                            len_i = domain_info[1] - domain_info[0] + 1
                            if (overlap / len_i) > 0.5:
                                is_new_domain = False
                                break
                    if is_new_domain:
                        merge_domains.append(domain_info)

                for domain_info in merge_domains:
                    domain_name = str(domain_info[6]).replace(',', '')
                    f_save.write(query_name + '\t' + domain_name + '\t' +
                                 str(domain_info[0]) + '\t' + str(domain_info[1]) + '\t' +
                                 str(domain_info[3]) + '\t' + str(domain_info[4]) + '\n')

        return cur_table
    except Exception as e:
        return e

def read_fasta(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:].split(" ")[0].split('\t')[0]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs

def read_fasta_v1(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs

def store_fasta(contigs, file_path):
    with open(file_path, 'w') as f_save:
        for name in contigs.keys():
            seq = contigs[name]
            f_save.write('>'+name+'\n'+seq+'\n')
    f_save.close()


def plot_training_metrics(history):
    # plot metrics
    plt.figure()
    plt.plot(history.history['val_f1_m'])
    plt.plot(history.history['f1_m'])
    plt.legend(['val_f1_m', 'train_f1_m'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('f1_m')
    plt.title('Epoch vs f1_m')
    plt.savefig('Train_Curve_NeuralTE.png', bbox_inches='tight', dpi=500)

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
    plt.savefig('Train_Curve_los_NeuralTE.png', bbox_inches='tight', dpi=500)

# ====================
# MAIN
# ====================
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"[ERROR] Parameter TE_library.fasta is required.")
        print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta work_dir num_threads [script_mode]")
        sys.exit(1)
    else:
        TE_library = sys.argv[1]
        work_dir = sys.argv[2]
        threads = int(sys.argv[3])
        if len(sys.argv) > 4:
            script_mode = sys.argv[4].upper()
            if script_mode not in ['T', 'P']:
                print(f"[ERROR] script_mode should be T or P, found {script_mode} instead.")
                print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta work_dir num_threads  [script_mode]")
                sys.exit(1)
        else:
            print("[INFO] Using training script mode by default")
            script_mode = "T"

    if script_mode == "T":
        start_all = time.time()
        print("### Step 0: Starting to load and transform the dataset......")
        start = time.time()

        os.makedirs("trained_models/", exist_ok=True)

        project_dir = current_folder = os.path.dirname(os.path.abspath(__file__))


        X, y, seq_names, data_path, labels = load_data(internal_kmer_sizes, terminal_kmer_sizes,
                                                              TE_library, work_dir, project_dir, threads)

        ##########################
        # 0. Save the data
        os.makedirs("data_for_training/", exist_ok=True)
        np.save("data_for_training/X.npy", X)
        np.save("data_for_training/Y.npy", y)

        """X = np.load("data_for_training/X.npy")
        y = np.load("data_for_training/Y.npy")"""
        num_classes = int(np.max(y) + 1)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. data split: 80% train, 10% dev and 10% test
        print("### Step 1: Starting the dataset spliting ......")
        start = time.time()

        validation_size = 0.2
        seed = 7
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=validation_size, random_state=seed, stratify=y)
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
        print("### Step 3: Starting the labels preprocessing steps ......")
        start = time.time()

        Y_train_one_hot =  np.array(to_categorical(Y_train, num_classes))
        Y_dev_one_hot =  np.array(to_categorical(Y_dev, num_classes))
        Y_test_one_hot = np.array(to_categorical(Y_test, num_classes))

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        batch_size = 512
        num_epochs = 100
        model = get_model(work_dir, X_feature_len, num_classes)
        tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        history, _ = run_experiment(model, X_train, Y_train_one_hot, X_dev, Y_dev_one_hot, batch_size=batch_size, num_epochs=num_epochs)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        model.save('trained_models/NeuralTE_retrained_model.h5')

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

        predicted_classes = model.predict(X_test)
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
        project_dir = current_folder = os.path.dirname(os.path.abspath(__file__))
        X, Y, seq_names, data_path, labels = load_data(internal_kmer_sizes, terminal_kmer_sizes,
                                               TE_library, work_dir, project_dir, threads)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data


        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        model = load_model("trained_models/NeuralTE_retrained_model.h5")

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Predict the labels
        print("### Step 3: Starting to predict the TE classification......")
        start = time.time()

        y_preds_probs = model.predict(X)

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
        print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta work_dir num_threads  [script_mode]")
        sys.exit(1)
