# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from random import seed
from random import randint
import random
import re
import numpy as np
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
import itertools
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import seaborn as sn
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import time


# Superfamily dict
superf_dict = {
    'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
    'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
    'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
    'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
    'KOLOBOK': 28, 'ACADEM-1': 29
    }

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

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


def load_data(TE_lib, mode="T"):
    seqs = []
    classifications = []

    for te in SeqIO.parse(TE_lib, "fasta"):
        seqs.append(re.sub(r'[^ACGT]', '', str(te.seq).upper()))
        if mode == "T":
            superfamily = te.id.split(" ")[0].split("#")[1]
            classifications.append(superf_dict[superfamily])
        elif mode == "P":
            classifications.append(te.id)
        else:
            return None, None

    X = generate_mats(seqs)
    return np.asarray(X), np.asarray(classifications)


##word_seq generates eg. ['AA', 'AT', 'TC', 'CG', 'GT']
def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)


##generate all the combinations of ATCG, we will input the k-mer number
def generate_kmer_dic (repeat_num):

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


##add the number into the kmer in the kmer dic
##generate the vector from the kmer_dic
##this will generat_mat for only one sample
def generate_mat (words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1

    num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])

    return (num_list)


##generate matrix for all samples
def generate_mats (seqs):

    seq_mats = []
    for eachseq in seqs:
        words_list = word_seq(eachseq, 7, stride=1)  ##change the k to 3
        kmer_dic = generate_kmer_dic(7)  ##this number should be the same as the window slide number
        num_list = generate_mat(words_list,kmer_dic)

        ##store the all the samples into seq_mats
        ##seq_mats = [[0,1,3,4],[3,4,5,6],...]
        seq_mats.append(num_list)

    return seq_mats


def get_model(num_classes):
    model = Sequential()

    model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, 16384, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(150, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(225, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    ##You can add a dropout layer to overcome the problem of overfitting to some extent. Dropout randomly turns off
    # a fraction of neurons during the training process, reducing the dependency on the training set by some amount.
    # How many fractions of neurons you want to turn off is decided by a hyperparameter, which can be tuned accordingly.
    # This way, turning off some neurons will not allow the network to memorize the training data since not all the neurons
    # will be active at the same time and the inactive neurons will not be able to learn anything.
    # This way, turning off some neurons will not allow the network to memorize the training data
    # since not all the neurons will be active at the same time and the inactive neurons will not be able to learn anything.

    model.add(Dense(int(num_classes), activation='softmax'))
    # since 4 classes ##the output have four unit
    ##Your output's are integers for class labels. Sigmoid logistic function outputs values in range (0,1).
    # The output of the softmax is also in range (0,1), but the softmax function adds another constraint on outputs:-
    # the sum of outputs must be 1. Therefore the output of softmax can be interpreted as probability of the input
    # for each class.

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])
    return model


def run_experiment(model, X_train, Y_train, labels, X_dev, Y_dev, batch_size, num_epochs):
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=15, restore_best_weights=True)

    X_train = X_train.astype("float32")
    Y_train = Y_train.astype("float32")
    X_dev = X_dev.astype("float32")
    Y_dev = Y_dev.astype("float32")

    train_steps = max(X_train.shape[0] // batch_size, 1)
    val_steps = max(X_dev.shape[0] // batch_size, 1)
    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                .shuffle(min(len(X_train), 10000), reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_dev, Y_dev))
              .batch(batch_size, drop_remainder=True)
              .repeat()
              .prefetch(tf.data.AUTOTUNE))

    history = model.fit(
        train_ds,
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    del train_ds
    del val_ds

    return history


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
    plt.savefig('Train_Curve_DeepTE.png', bbox_inches='tight', dpi=500)

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
    plt.savefig('Train_Curve_los_DeepTE.png', bbox_inches='tight', dpi=500)