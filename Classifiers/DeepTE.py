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
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}

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
    # 4. Define model architecture
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
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(X_dev, Y_dev), callbacks=[lr_scheduler, early_stopping], verbose=1)
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
        X, Y = load_data(TE_library, mode=script_mode)

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
        print("### Step 1: Starting the dataset spliting ......")
        start = time.time()

        validation_size = 0.2
        seed = 7
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        X_train, X_test_dev, Y_train, Y_test_dev = train_test_split(X, Y, test_size=validation_size, random_state=seed, stratify=Y)

        X_dev, X_test, Y_dev, Y_test = train_test_split(X_test_dev, Y_test_dev, test_size=0.5, random_state=seed, stratify=Y_test_dev)

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

        # Space optimization
        X = None
        Y = None

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ##########################
        # 2. Preprocess input data
        print("### Step 2: Starting the features preprocessing steps ......")
        start = time.time()

        X_train = X_train.reshape(X_train.shape[0], 1, 16384, 1)  ##shape[0] indicates sample number
        X_dev = X_dev.reshape(X_dev.shape[0], 1, 16384, 1)
        X_test = X_test.reshape(X_test.shape[0], 1, 16384, 1)  ##kmer == 3 so it would be 64
        X_train = X_train.astype('float64')
        X_dev = X_dev.astype('float64')
        X_test = X_test.astype('float64')

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
        print("### Step 3: Starting the labels preprocessing steps ......")
        start = time.time()

        Y_train_one_hot = to_categorical(Y_train, int(num_classes))  # four labels
        Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))  # four labels
        Y_test_one_hot = to_categorical(Y_test, int(num_classes))  # four labels

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        batch_size = 512
        num_epochs = 100
        model = get_model(num_classes)
        history = run_experiment(model, X_train, Y_train_one_hot, Y_train, X_dev, Y_dev_one_hot, batch_size, num_epochs)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        model.save('trained_models/DeepTE_retrained_model.h5')

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
        X, labels = load_data(TE_library, mode=script_mode)
        labels = labels.tolist()
        X = X.reshape(X.shape[0], 1, 16384, 1)
        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data


        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        model = load_model("trained_models/DeepTE_retrained_model.h5")

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
        print(f"[USAGE] python3 {sys.argv[0]} TE_library.fasta [script_mode]")
        sys.exit(1)
