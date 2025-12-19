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
from tensorflow.keras.models import load_model
import itertools
from tqdm import tqdm
import seaborn as sn

from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GRU, Lambda
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

import time
import joblib

# ====================
# CONFIGURACIÓN GPU
# ====================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# Superfamily dict
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}

# Add attention layer to the deep learning network
class attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="random_normal",
                                 trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x, **kwargs):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W))
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        return context, alpha


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
    plt.savefig('Train_Curve_CREATE.png', bbox_inches='tight', dpi=500)

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
    plt.savefig('Train_Curve_los_CREATE.png', bbox_inches='tight', dpi=500)


def create_gru_model(len_thre):
    # Define GRU model architecture
    rnn_input = Input(shape=(len_thre, 4), name="rnn_input")
    x = GRU(128, dropout=0.2, return_sequences=True, name="rnn1")(rnn_input)
    x = GRU(64, dropout=0.2, return_sequences=True, name="rnn2")(x)
    x = Flatten(name="rnn_flatten")(x)
    x = Dropout(0.5, name="rnn_dropout1")(x)
    x = Dense(128, activation="relu", name="rnn_fc1")(x)
    model = Model(inputs=rnn_input, outputs=x, name="GRU_Model")
    return model


def create_cnn_model(k):
    cnn_input = Input(shape=(1, pow(4, k), 1), name="cnn_input")
    x = Conv2D(filters=64, kernel_size=(1, 3), activation="relu", name="cnn_conv1")(cnn_input)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool1")(x)
    x = Conv2D(filters=128, kernel_size=(1, 3), activation="relu", name="cnn_conv2")(x)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool2")(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), activation="relu", name="cnn_conv3")(x)
    x = MaxPool2D(pool_size=(1, 2), name="cnn_pool3")(x)
    x = Flatten(name="cnn_flatten")(x)
    x = Dropout(0.5, name="cnn_dropout1")(x)
    x = Dense(128, activation="relu", name="cnn_fc1")(x)
    model = Model(inputs=cnn_input, outputs=x, name="CNN_Model")
    return model


# Add an attention layer to the CNN and RNN output joint layer
def create_attn_model(kmer, len_thre, class_num):
    cnn_model = create_cnn_model(kmer)
    rnn_model = create_gru_model(len_thre)
    merge_layer = Lambda(lambda x: tf.stack(x, axis=1), name="stack_layer")(
        [cnn_model.output, rnn_model.output]
    )
    attn_layer, attention_weight = attention(name="attention")(merge_layer)
    attn_layer = Lambda(lambda x: K.sum(x, axis=1), name="attn_sum")(attn_layer)
    dense_layer = Dense(128, activation="relu", name="fc")(attn_layer)
    dropout_layer = Dropout(0.5, name="dropout")(dense_layer)
    output_layer = Dense(int(class_num), activation="softmax", name="output")(dropout_layer)
    model = Model(inputs=[cnn_model.input, rnn_model.input], outputs=output_layer)
    attention_model = Model([cnn_model.input, rnn_model.input], attention_weight)
    return model, attention_model


def run_experiment(X_oh_train, X_oh_test, X_kmer_train, X_kmer_test, y_train, y_test, class_num, k, l):
    model, attention_model = create_attn_model(k, l, class_num)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=[f1_m])

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)

    X_oh_train, X_kmer_train, y_train = shuffle(X_oh_train, X_kmer_train, y_train)
    history = model.fit([X_kmer_train, X_oh_train], y_train, batch_size=32, epochs=10, verbose=1,
              validation_data=([X_kmer_test, X_oh_test], y_test), callbacks=[lr_scheduler, early_stopping])

    return model, history


def convert_undetermined_base(seq):
    seq = seq.upper()
    seq = seq.replace("Y", "C")
    seq = seq.replace("D", "G")
    seq = seq.replace("S", "C")
    seq = seq.replace("R", "G")
    seq = seq.replace("V", "A")
    seq = seq.replace("K", "G")
    seq = seq.replace("N", "T")
    seq = seq.replace("H", "A")
    seq = seq.replace("W", "A")
    seq = seq.replace("M", "C")
    seq = seq.replace("X", "G")
    seq = seq.replace("B", "C")
    # new_seq = Seq(seq)
    return seq


# Sequence one-hot encoding
def seq2oh(seq, len_thre):
    if len(seq) >= len_thre:
        seq_1 = list(seq)[0:len_thre//2]
        seq_2 = list(seq)[-len_thre//2:]
        seq = seq_1 + seq_2
    else:
        seq_1 = list(seq)[0:len(seq)//2]
        seq_2 = list(seq)[-len(seq)//2:]
        seq = seq_1 + [0] * (len_thre - len(seq)) + seq_2
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array(["A", "C", "G", "T"]).reshape(-1, 1))
    seq_encode = enc.transform(np.array(seq).reshape(-1, 1)).toarray()
    return seq_encode


def nucle2num(nucleotide):
    nucleotide = nucleotide.upper()
    if nucleotide == "A":
        return 0
    elif nucleotide == "G":
        return 1
    elif nucleotide == "C":
        return 2
    elif nucleotide == "T":
        return 3
    else:
        raise ValueError(f"Invalid nucleotide: {nucleotide}")


# Sequence kmer encoding
def seq2kmer(seq, k):
    b = 4 # base
    h = 0 # hash value
    hash_dict = {key: 0 for key in range(4 ** k)}
    if len(seq) >= k:
        # Initialize hash value
        for i in range(k):
            h = h * b + nucle2num(seq[i])
        hash_dict[h] = 1
        # Calculate frequency of remaining k-mers
        for i in range(k, len(seq)):
            h = (h - nucle2num(seq[i-k]) * b**(k-1)) * b + nucle2num(seq[i])
            hash_dict[h] += 1
    else:
        raise ValueError(f"Invalid k-mer size: {k}")
    return hash_dict


def get_kmer_data(data_file, k):
    kmer = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        kmer.append(list(seq2kmer(seq, k).values()))
    X_kmer = np.array(kmer)
    return X_kmer


def get_oh_data(data_file, len_thre):
    one_hot = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        one_hot.append(seq2oh(seq, len_thre))
    X_oh = np.array(one_hot)
    return X_oh


def get_label_data(data_file, mode="T"):
    labels = []
    for record in SeqIO.parse(data_file, "fasta"):
        if mode == "T":
            classification = record.id.split(" ")[0].split("#")[1]
            labels.append(superf_dict[classification])
        elif mode == "P":
            labels.append(record.id)
        else:
            return None
    return np.asarray(labels)


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

    k = 7
    l = 600

    if script_mode == "T":
        start_all = time.time()
        print("### Step 0: Starting to load and transform the dataset......")
        start = time.time()
        os.makedirs("trained_models/", exist_ok=True)

        X_kmer = get_kmer_data(TE_library, k)
        X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, pow(4, k), 1)
        X_kmer = X_kmer.astype("float64")

        X_oh = get_oh_data(TE_library, l)
        y = get_label_data(TE_library)

        num_classes = int(np.max(y) + 1)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. data split: 80% train, 10% dev and 10% test
        print("### Step 1: Starting the dataset spliting ......")
        start = time.time()

        validation_size = 0.2
        seed = 7
        X_oh_train, X_oh_test_dev, X_kmer_train, X_kmer_test_dev, Y_train, y_test_dev = train_test_split(X_oh, X_kmer, y, stratify=y,
                                                                            test_size=validation_size, random_state=seed)

        X_oh_test, X_oh_dev, X_kmer_test, X_kmer_dev, Y_test, Y_dev = train_test_split(X_oh_test_dev, X_kmer_test_dev, y_test_dev,
                                    stratify=y_test_dev, test_size=0.5, random_state=seed)

        print("\nDataset shapes:")
        print(f"X_oh_train shape: {X_oh_train.shape}")
        print(f"X_oh_dev shape: {X_oh_dev.shape}")
        print(f"X_oh_test shape: {X_oh_test.shape}")
        print(f"X_kmer_train shape: {X_kmer_train.shape}")
        print(f"X_kmer_dev shape: {X_kmer_dev.shape}")
        print(f"X_kmer_test shape: {X_kmer_test.shape}")

        print("\nLabel information:")
        print(f"Shape of Y_train: {Y_train.shape}")
        print(f"Shape of Y_train: {Y_dev.shape}")
        print(f"Shape of Y_test: {Y_test.shape}")

        print(f"\nNumber of unique classes in Y_train: {len(np.unique(Y_train))}")
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

        Y_train_one_hot = to_categorical(Y_train, int(num_classes))  # four labels
        Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))  # four labels
        Y_test_one_hot = to_categorical(Y_test, int(num_classes))  # four labels

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        model, history = run_experiment(X_oh_train, X_oh_dev, X_kmer_train, X_kmer_dev, Y_train_one_hot, Y_dev_one_hot, num_classes, k, l)
        tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        model.save('trained_models/CREATE_retrained_model.h5')

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

        predicted_classes = model.predict([X_kmer_test, X_oh_test])
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

        X_kmer = get_kmer_data(TE_library, k)
        X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, pow(4, k), 1)
        X_kmer = X_kmer.astype("float64")

        X_oh = get_oh_data(TE_library, l)

        labels = get_label_data(TE_library, mode=script_mode).tolist()

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data


        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        k = 7
        l = 600
        class_num = 30

        model, attention_model = create_attn_model(k, l, class_num)
        model.load_weights("trained_models/CREATE_retrained_model.h5")

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Predict the labels
        print("### Step 3: Starting to predict the TE classification......")
        start = time.time()

        y_preds_probs = model.predict([X_kmer, X_oh])

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
