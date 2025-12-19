# -*- coding: utf-8 -*-

import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
import sys
import seaborn as sn
import random
import re
import subprocess
from collections import Counter
import shutil
import argparse
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import time
import joblib
import torch
import torch.nn.functional as F
import pickle
import json
from hierarchicalsoftmax import greedy_predictions
from transformers import TrainingArguments
import difflib
from typing import Dict, Tuple, List
import importlib.util

# to load the ML based models
from Classifiers import NeuralTE
from Classifiers import CREATE
from Classifiers import ClassifyTE
from Classifiers import DeepTE
from Classifiers import TERL
from Classifiers import Inpactor2_Class
from Classifiers import Terrier
from Classifiers import BERTE
from Classifiers import TEClass2

superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
                   'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
                   'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
                   'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
                   'KOLOBOK': 28, 'ACADEM-1': 29}

class TrainingHistory:
    """Keras like history object for training"""
    def __init__(self, log_history):
        self.history = {
            'loss': [],
            'val_loss': [],
            'f1_m': [],
            'val_f1_m': []
        }
        for entry in log_history:
            if "loss" in entry and "epoch" in entry:
                self.history['loss'].append(entry['loss'])
            if "eval_loss" in entry:
                self.history['val_loss'].append(entry['eval_loss'])
            if "f1" in entry:
                self.history['f1_m'].append(entry['f1'])
            if "eval_f1" in entry:
                self.history['val_f1_m'].append(entry['eval_f1'])


def info(message):
    print(f"[INFO] {message}")


def error(message):
    print(f"[ERROR] {message}")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PanTEon: Deep Learning Framework for Transposable Element Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="module", required=True)

    # -----------------------
    # training
    # -----------------------
    p_train = subparsers.add_parser("training", help="Train models")
    p_train.add_argument("-f", "--fasta", required=True, help="Path to the TE fasta file")
    p_train.add_argument("-w", "--work-dir", required=False, help="Path to the working directory", default="work_dir")
    p_train.add_argument("-t", "--threads", required=True, type=int, help="Number of threads to be used")
    p_train.add_argument(
        "-n", "--models", help=("Models to be used (comma-separated). Options=All, NeuralTE, Terrier, CREATE, "
              "ClassifyTE, DeepTE, Inpactor2_Class, TERL, BERTE, TEClass2")
    )
    p_train.add_argument(
        "-d", "--models_directory", required=True,
        help="Directory where models will be stored during training"
    )
    p_train.add_argument("-z", "--min_prob", required=False, default=0.6, type=float,
                       help="Minimum probability to classify a TE")

    # -----------------------
    # inference
    # -----------------------
    p_inf = subparsers.add_parser("inference", help="Run inference (prediction)")
    p_inf.add_argument("-f", "--fasta", required=True, help="Path to the TE fasta file")
    p_inf.add_argument("-t", "--threads", required=True, type=int, help="Number of threads to be used")
    p_inf.add_argument("-w", "--work-dir", required=True, help="Path to the working directory", default="work_dir")
    p_inf.add_argument(
        "-n", "--models",
        help=("Models to be used (comma-separated). Options=All, NeuralTE, Terrier, CREATE, "
              "ClassifyTE, DeepTE, Inpactor2_Class, TERL, BERTE, TEClass2")
    )
    p_inf.add_argument(
        "-d", "--models_directory", required=True,
        help="Directory containing trained models"
    )
    p_inf.add_argument("-p", "--prefix", required=True,
                       help="Prefix for the output results")
    p_inf.add_argument("-z", "--min_prob", required=False, default=0.6, type=float,
                       help="Minimum probability to classify a TE")

    # -----------------------
    # library
    # -----------------------
    p_lib = subparsers.add_parser("library", help="Create TE library from PanTEon Database")
    p_lib.add_argument("--taxon", required=False,
                       help="Taxon name (e.g., Plantae, Chordata, etc.)")
    p_lib.add_argument("--req_class", required=False,
                       help="Classification name (e.g., ClassI, LTR, Helitron, etc.)")
    p_lib.add_argument("--view_only", action="store_true", default=False,
                       help="Only print report (do not write FASTA)")

    # -----------------------
    # evaluation
    # -----------------------
    p_eval = subparsers.add_parser("evaluation", help="Evaluate models from fasta files")
    p_eval.add_argument("--true_fasta", required=True, help="FASTA with ground truth classifications.")
    p_eval.add_argument("--pred_fasta", required=True, help="FASTA with predicted classifications.")
    p_eval.add_argument("--level", type=int, default=-1,
                    help="Level (1=A, 2=B, 3=C) to extract the class from ID#A/B/C. "
                         "By default -1 (last level).")
    p_eval.add_argument("--out_confusion", default="confusion_matrix_fasta.csv",
                    help="Output CSV for the confusion matrix.")
    p_eval.add_argument("--out_report", default="classification_report_fasta.csv",
                    help="Output CSV for the classification report.")

    args = parser.parse_args()

    # validate shared constraints
    if hasattr(args, "threads") and args.threads is not None and args.threads < 1:
        parser.error("-t/--threads should be >= 1")

    return args


def plot_training_metrics(history, model, path="."):
    # plot metrics
    plt.figure()
    plt.plot(history.history['val_f1_m'])
    plt.plot(history.history['f1_m'])
    plt.legend(['val_f1_m', 'train_f1_m'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('f1_m')
    plt.title('Epoch vs f1_m')
    plt.savefig(f'{path}/Train_Curve_{model}.png', bbox_inches='tight', dpi=500)

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
    plt.savefig(f'{path}/Train_Curve_los_{model}.png', bbox_inches='tight', dpi=500)


def metrics(Y_validation,predictions, num_classes, model, reportDir):

    acc = accuracy_score(Y_validation, predictions)
    f1 = f1_score(Y_validation, predictions,average='weighted')
    rec = recall_score(Y_validation, predictions,average='weighted')
    pre = precision_score(Y_validation, predictions, average='weighted')
    info(f"Performance metrics for model {model}:")
    print(f'    -> Accuracy: {acc}')
    print(f'    -> F1 score: {f1}')
    print(f'    -> Recall: {rec}')
    print(f'    -> Precision: {pre}')
    print(f'\n     -> clasification report:\n', classification_report(Y_validation, predictions))
    print(f'\n     -> confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #Creamos la matriz de confusión
    snn_cm = confusion_matrix(Y_validation, predictions)
    num_classes = min(num_classes, len(snn_cm))

    # Visualizamos la matriz de confusión
    snn_df_cm = pd.DataFrame(snn_cm, range(num_classes), range(num_classes))
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.savefig(f'{reportDir}/confusionMatrix_{model}.png', bbox_inches='tight', dpi=500)
    return acc, f1, rec, pre


def training(TE_library, work_dir, threads, models, num_classes, output_directory, superf_dict, custom_registry, PanTEon_dir):
    dataTraining_dir = f"{output_directory}/data_for_training/"
    report_dir = f"{output_directory}/reports/"
    os.makedirs(dataTraining_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Paths to training, validation, and test fasta files
    training_fasta = f"{dataTraining_dir}/TE_training.fasta"
    val_fasta = f"{dataTraining_dir}/TE_val.fasta"
    test_fasta = f"{dataTraining_dir}/TE_test.fasta"


    if os.path.exists(training_fasta) and os.path.exists(val_fasta) and os.path.exists(test_fasta):
        info("using the split fasta file found at folder data_for_training. Skipping....")

    else:
        TE_dataset = [te for te in SeqIO.parse(TE_library, "fasta")]
        classifications = [te.id.split(" ")[0].split("#")[1] for te in TE_dataset]
        validation_size = 0.2
        seed = 7
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        TE_train, TE_temp, classification_train, classification_temp = train_test_split(TE_dataset, classifications,
                                                                                        test_size=validation_size,
                                                                                        random_state=seed,
                                                                                        stratify=classifications)
        TE_test, TE_val, classification_test, classification_val = train_test_split(TE_temp, classification_temp,
                                                                                    test_size=0.5,
                                                                                    random_state=seed,
                                                                                    stratify=classification_temp)

        # Save the fasta files for training validation and test
        SeqIO.write(TE_train, training_fasta, "fasta")
        SeqIO.write(TE_val, val_fasta, "fasta")
        SeqIO.write(TE_test, test_fasta, "fasta")

    os.makedirs(output_directory, exist_ok=True)

    model_metrics = {}

    # Training PanTEon in-built models
    for model_name in models:
        start = time.time()
        info(f"Starting {model_name} training....")

        if model_name == "NeuralTE":
            if os.path.exists(f"{output_directory}/NeuralTE_retrained_model.keras"):
                info(
                    f"Using the found model at {output_directory}/NeuralTE_retrained_model.keras. Skipping retraining....")
            else:
                internal_kmer_sizes = [1, 3]
                terminal_kmer_sizes = [1, 2, 3]
                X_feature_len = 309
                project_dir = os.path.dirname(os.path.abspath(__file__))
                NeuralTE.all_wicker_class = superf_dict
                NeuralTE.class_num = num_classes
                NeuralTE.inverted_all_wicker_class = {value: key for key, value in superf_dict.items()}

                X_train, Y_train, _, _, _ = NeuralTE.load_data(internal_kmer_sizes,
                                                               terminal_kmer_sizes,
                                                               training_fasta,
                                                               work_dir, project_dir, threads)

                X_dev, Y_dev, _, _, _ = NeuralTE.load_data(internal_kmer_sizes,
                                                           terminal_kmer_sizes,
                                                           val_fasta,
                                                           work_dir, project_dir, threads)
                X_test, Y_test, _, _, _ = NeuralTE.load_data(internal_kmer_sizes,
                                                             terminal_kmer_sizes,
                                                             test_fasta,
                                                             work_dir, project_dir, threads)

                Y_train_one_hot = np.array(to_categorical(Y_train, num_classes))
                Y_dev_one_hot = np.array(to_categorical(Y_dev, num_classes))

                batch_size = 512
                num_epochs = 100
                model = NeuralTE.get_model(work_dir, X_feature_len, num_classes)

                history, _ = NeuralTE.run_experiment(model, X_train, Y_train_one_hot, X_dev, Y_dev_one_hot,
                                                     batch_size=batch_size, num_epochs=num_epochs)

                model.save(output_directory+'/NeuralTE_retrained_model.keras')
                plot_training_metrics(history, "NeuralTE", report_dir)

                predicted_classes = model.predict(X_test)
                predicted_classes = np.argmax(predicted_classes, axis=1)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "NeuralTE", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "CREATE":
            if os.path.exists(f'{output_directory}/CREATE_retrained_model.keras'):
                info(
                    f"Using the found model at {output_directory}/CREATE_retrained_model.keras. Skipping retraining....")
            else:
                k = 7
                l = 600
                CREATE.superf_dict = superf_dict

                X_kmer_train = CREATE.get_kmer_data(training_fasta, k)
                X_kmer_train = X_kmer_train.reshape(X_kmer_train.shape[0], 1, pow(4, k), 1)
                X_kmer_train = X_kmer_train.astype("float64")

                X_oh_train = CREATE.get_oh_data(training_fasta, l)
                Y_train = CREATE.get_label_data(training_fasta)

                X_kmer_dev = CREATE.get_kmer_data(val_fasta, k)
                X_kmer_dev = X_kmer_dev.reshape(X_kmer_dev.shape[0], 1, pow(4, k), 1)
                X_kmer_dev = X_kmer_dev.astype("float64")

                X_oh_dev = CREATE.get_oh_data(val_fasta, l)
                Y_dev = CREATE.get_label_data(val_fasta)

                X_kmer_test = CREATE.get_kmer_data(test_fasta, k)
                X_kmer_test = X_kmer_test.reshape(X_kmer_test.shape[0], 1, pow(4, k), 1)
                X_kmer_test = X_kmer_test.astype("float64")

                X_oh_test = CREATE.get_oh_data(test_fasta, l)
                Y_test = CREATE.get_label_data(test_fasta)

                Y_train_one_hot = to_categorical(Y_train, int(num_classes))
                Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))

                model, history = CREATE.run_experiment(X_oh_train, X_oh_dev, X_kmer_train, X_kmer_dev, Y_train_one_hot,
                                                       Y_dev_one_hot, num_classes, k, l)

                model.save(f'{output_directory}/CREATE_retrained_model.keras')
                plot_training_metrics(history, "CREATE", report_dir)

                predicted_classes = model.predict([X_kmer_test, X_oh_test])
                predicted_classes = np.argmax(predicted_classes, axis=1)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "CREATE", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "ClassifyTE":
            if os.path.exists(f"{output_directory}/ClassifyTE_retrained_model.pkl"):
                info(f"Using the found model at {output_directory}/ClassifyTE_retrained_model.pkl. Skipping retraining....")
            else:
                ClassifyTE.superf_dict = superf_dict
                ClassifyTE.script_dir = PanTEon_dir

                X_train, Y_train = ClassifyTE.load_data(training_fasta, threads, "T")
                X_test, Y_test = ClassifyTE.load_data(test_fasta, threads, "T")
                model = ClassifyTE.get_model()
                ClassifyTE.run_experiment(model, X_train, Y_train)

                joblib.dump(model, f"{output_directory}/ClassifyTE_retrained_model.pkl")

                predicted_classes = model.predict(X_test)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "ClassifyTE", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "DeepTE":
            if os.path.exists(f"{output_directory}/DeepTE_retrained_model.keras"):
                info(f"Using the found model at {output_directory}/DeepTE_retrained_model.keras. Skipping retraining....")
            else:
                DeepTE.superf_dict = superf_dict
                X_train, Y_train = DeepTE.load_data(training_fasta)
                X_dev, Y_dev = DeepTE.load_data(val_fasta)
                X_test, Y_test = DeepTE.load_data(test_fasta)

                X_train = X_train.reshape(X_train.shape[0], 1, 16384, 1)  ##shape[0] indicates sample number
                X_dev = X_dev.reshape(X_dev.shape[0], 1, 16384, 1)
                X_test = X_test.reshape(X_test.shape[0], 1, 16384, 1)  ##kmer == 3 so it would be 64
                X_train = X_train.astype('float64')
                X_dev = X_dev.astype('float64')
                X_test = X_test.astype('float64')

                Y_train_one_hot = to_categorical(Y_train, int(num_classes))
                Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))

                batch_size = 512
                num_epochs = 100
                model = DeepTE.get_model(num_classes)
                history = DeepTE.run_experiment(model, X_train, Y_train_one_hot, Y_train, X_dev, Y_dev_one_hot, batch_size,
                                                num_epochs)

                model.save(output_directory+'/DeepTE_retrained_model.keras')
                plot_training_metrics(history, "DeepTE", report_dir)

                predicted_classes = model.predict(X_test)
                predicted_classes = np.argmax(predicted_classes, axis=1)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "DeepTE", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "TERL":
            if os.path.exists(f"{output_directory}/TERL_Classify_retrained_model"):
                info(f"Using the found model at {output_directory}/TERL_Classify_retrained_model. Skipping retraining....")
            else:
                TERL.superf_dict = superf_dict
                max_len = 19926
                X_train, _ = TERL.data_handler(training_fasta, max_len=max_len, mode="P")
                Y_train = TERL.get_label_data(training_fasta)
                X_dev, _ = TERL.data_handler(val_fasta, max_len=max_len, mode="P")
                Y_dev = TERL.get_label_data(val_fasta)
                X_test, _ = TERL.data_handler(test_fasta, max_len=max_len, mode="P")
                Y_test = TERL.get_label_data(test_fasta)

                classes = np.unique(Y_train).tolist()
                vocab_size = len(['A', 'C', 'G', 'T', 'N', 5])  # 5 is the background signal for padding
                shuffled = np.random.permutation(range(Y_train.shape[0]))
                X_train = X_train[shuffled]
                Y_train = Y_train[shuffled]

                # Default parameters
                architecture = ["conv", "pool", "conv", "pool", "conv", "pool", "fc", "fc"]
                activation_functions = ["relu", "avg", "relu", "avg", "relu", "avg", "relu", "relu"]
                widths = [20, 10, 20, 15, 35, 15, 1000, 500]
                strides = [1, 10, 1, 15, 1, 15]
                dilations = [1, 1, 1]
                feature_maps = [64, 32, 32]
                optimizer = 'ADAM'
                learning_rate = 0.001
                l2 = 0.001
                train_batch_size = 32
                test_batch_size = 32
                epochs = 50
                dropout = 0.5

                labels, predictions, accuracies, best_result, training_time, training_out = TERL.train_evaluate(
                    X_train,
                    Y_train,
                    X_dev,
                    Y_dev,
                    vocab_size,
                    max_len,
                    classes,
                    num_classes,
                    architecture, activation_functions, widths,
                    strides, dilations, feature_maps,
                    TERL.get_optimizer(optimizer, learning_rate),
                    l2, train_batch_size, test_batch_size,
                    epochs, dropout,
                    output_file=f"{output_directory}/TERL_Classify_retrained_model"
                )

                if os.path.exists(f"{output_directory}/TERL_Classify_retrained_model"):
                    shutil.rmtree(f"{output_directory}/TERL_Classify_retrained_model")
                shutil.move(f"{output_directory}/TERL_Classify_retrained_model_" + str(epochs), f"{output_directory}/TERL_Classify_retrained_model")

                with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                    tf.compat.v1.saved_model.loader.load(
                        sess,
                        ['serve'],
                        f"{output_directory}/TERL_Classify_retrained_model"
                    )
                    test_size = len(X_test)
                    predicted_classes = np.array([], dtype=np.uint8)
                    for batch in range(0, test_size, test_batch_size):
                        x_batch = X_test[batch: batch + test_batch_size]

                        pre_xo = sess.run('one_hot_x:0', feed_dict={'pre_x:0': x_batch})
                        x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)

                        predicted_classes = np.concatenate([
                            predicted_classes,
                            sess.run(
                                'prediction:0',
                                feed_dict={
                                    'x_input:0': x_batch,
                                    'is_training:0': False
                                }
                            )
                        ])
                    acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "TERL", report_dir)
                    model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "Inpactor2_Class":
            if os.path.exists(f'{output_directory}/Inpactor2_Class_retrained_model.keras'):
                info(f"Using the found model at {output_directory}/Inpactor2_Class_retrained_model.keras. Skipping retraining....")
            else:
                Inpactor2_Class.superf_dict = superf_dict
                X_train, Y_train = Inpactor2_Class.load_data(training_fasta)
                X_dev, Y_dev = Inpactor2_Class.load_data(val_fasta)
                X_test, Y_test = Inpactor2_Class.load_data(test_fasta)
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_dev_scaled = scaler.transform(X_dev)
                X_test_scaled = scaler.transform(X_test)
                joblib.dump(scaler, f"{output_directory}/scaler.pkl")

                pca = decomposition.PCA(n_components=0.96, svd_solver='full')
                pca.fit(X_train_scaled)
                X_train_pca = pca.transform(X_train_scaled)
                X_dev_pca = pca.transform(X_dev_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                joblib.dump(pca, f"{output_directory}/pca.pkl")

                Y_train_one_hot = to_categorical(Y_train, int(num_classes))
                Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))
                Y_test_one_hot = to_categorical(Y_test, int(num_classes))

                batch_size = 512
                num_epochs = 200
                model = Inpactor2_Class.get_model(X_train_pca.shape[1], num_classes)
                history = Inpactor2_Class.run_experiment(model, X_train_pca, Y_train_one_hot, Y_train, X_dev_pca,
                                                            Y_dev_one_hot, batch_size=batch_size, num_epochs=num_epochs)

                model.save(f"{output_directory}/Inpactor2_Class_retrained_model.keras")
                plot_training_metrics(history, "Inpactor2_Class", report_dir)

                predicted_classes = model.predict(X_test_pca)
                predicted_classes = np.argmax(predicted_classes, axis=1)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "Inpactor2_Class", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "Terrier":
            if os.path.exists(f"{output_directory}/Terrier_retrained_model.pt"):
                info(f"Using the found model at {output_directory}/Terrier_retrained_model.pt. Skipping retraining....")
            else:
                # quick check
                for fasta_f in [training_fasta, val_fasta, test_fasta]:
                    bad_seqs = [x for x in SeqIO.parse(fasta_f, "fasta") if "/" not in x.id]
                    if len(bad_seqs) > 0:
                        error(f"there are some ID sequences without required character '/' in classification for Terrier:")

                Terrier.superf_dict = {key.split("/")[-1]: value for key, value in superf_dict.items()}
                orders = list(set([k.split("/")[-2] for k in superf_dict.keys()]))

                Terrier.order_dict = {orders[i]: i for i in range(len(orders))}
                max_len = 15000
                X_train, Y_train_order, Y_train_superf = Terrier.load_data(training_fasta, max_len)
                X_dev, Y_dev_order, Y_dev_superf = Terrier.load_data(val_fasta, max_len)
                X_test, Y_test_order, Y_test_superf = Terrier.load_data(test_fasta, max_len)

                Y_order = np.concatenate([Y_train_order, Y_dev_order, Y_test_order], axis=0)
                Y_superf = np.concatenate([Y_train_superf, Y_dev_superf, Y_test_superf], axis=0)

                root, order_node_map, superf_node_map = Terrier.build_hierarchy(Y_order, Y_superf, phi=1.02)
                Y_train = Terrier.targets_to_node_ids(root, Y_train_order, Y_train_superf, order_node_map,
                                                      superf_node_map)
                Y_dev = Terrier.targets_to_node_ids(root, Y_dev_order, Y_dev_superf, order_node_map, superf_node_map)
                Y_test = Terrier.targets_to_node_ids(root, Y_test_order, Y_test_superf, order_node_map, superf_node_map)

                leaf_nodes = [node for node in root.node_list if node and not node.children]
                leaf_id_to_label = {root.node_to_id[node]: node.name.replace("SUPERF::", "") for node in leaf_nodes}
                order_nodes = [node for node in root.children]
                order_id_to_label = {root.node_to_id[node]: node.name.replace("ORDER::", "") for node in order_nodes}

                train_dataset = Terrier.TransposonDataset(X_train, Y_train)
                val_dataset = Terrier.TransposonDataset(X_dev, Y_dev)
                test_dataset = Terrier.TransposonDataset(X_test, Y_test)

                batch_size = 32
                vocab_size = len("ACGTN") + 1
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                           num_workers=0)
                dev_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

                model = Terrier.TerrierNet(root=root, vocab_size=vocab_size)
                model, history = Terrier.run_experiment(model, root, train_loader, dev_loader, dev, 100, 1e-3, patience=10,
                                                weight_decay=0.0)

                torch.save(model.state_dict(), f"{output_directory}/Terrier_retrained_model.pt")
                with open(output_directory+"/root.pkl", "wb") as f:
                    pickle.dump(root, f)
                json.dump(leaf_id_to_label, open(output_directory+"/leaf_id_to_label.json", "w"))
                json.dump(order_id_to_label, open(output_directory+"/order_id_to_label.json", "w"))

                model.eval()
                test_logits_list, test_targets_list = [], []
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb = xb.to(dev)
                        logits = model(xb)
                        test_logits_list.append(logits.cpu())
                        test_targets_list.append(yb)
                test_logits = torch.cat(test_logits_list, dim=0)
                test_targets = torch.cat(test_targets_list, dim=0)

                # Reportes con etiquetas legibles
                # Predicciones greedy al nivel de hojas (superfamilia) y al nivel 1 (orden)

                pred_nodes = greedy_predictions(test_logits, root)  # lista de nodos (hojas)
                y_pred_superf_labels = [n.name.replace("SUPERF::", "") for n in pred_nodes]
                y_true_superf_labels = [leaf_id_to_label[int(i)] for i in test_targets.numpy().tolist()]
                inv = sorted(set(list(y_true_superf_labels) + list(y_pred_superf_labels)))
                # Mapear a índices compactos para matriz de confusión
                label_to_idx = {lab: i for i, lab in enumerate(inv)}
                y_true_idx = np.array([label_to_idx[l] for l in y_true_superf_labels], dtype=int)
                y_pred_idx = np.array([label_to_idx[l] for l in y_pred_superf_labels], dtype=int)

                acc, f1, rec, pre = metrics(y_true_idx, y_pred_idx, num_classes, "Terrier", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "BERTE":
            if os.path.exists(f"{output_directory}/BERTE_retrained_model.keras"):
                info(f"Using the found model at {output_directory}/BERTE_retrained_model.keras. Skipping retraining....")
            else:
                BERTE.superf_dict = superf_dict
                input_4mer_shape = (1, 4 ** 4 + 256, 1)
                input_5mer_shape = (1, 4 ** 5 + 256, 1)
                input_6mer_shape = (1, 4 ** 6 + 256, 1)
                X_train_4mer, X_train_5mer, X_train_6mer, Y_train = BERTE.load_data(training_fasta, bert_model_name=f"{PanTEon_dir}/tools/bert-mini", vocab_file=f"{PanTEon_dir}/data/kmer_vocab.txt")
                X_val_4mer, X_val_5mer, X_val_6mer, Y_dev = BERTE.load_data(val_fasta, bert_model_name=f"{PanTEon_dir}/tools/bert-mini", vocab_file=f"{PanTEon_dir}/data/kmer_vocab.txt")
                X_test_4mer, X_test_5mer, X_test_6mer, Y_test = BERTE.load_data(test_fasta, bert_model_name=f"{PanTEon_dir}/tools/bert-mini", vocab_file=f"{PanTEon_dir}/data/kmer_vocab.txt")

                X_train_4mer = X_train_4mer.reshape(X_train_4mer.shape[0], 1, 4 ** 4 + 256, 1)
                X_train_5mer = X_train_5mer.reshape(X_train_5mer.shape[0], 1, 4 ** 5 + 256, 1)
                X_train_6mer = X_train_6mer.reshape(X_train_6mer.shape[0], 1, 4 ** 6 + 256, 1)

                X_val_4mer = X_val_4mer.reshape(X_val_4mer.shape[0], 1, 4 ** 4 + 256, 1)
                X_val_5mer = X_val_5mer.reshape(X_val_5mer.shape[0], 1, 4 ** 5 + 256, 1)
                X_val_6mer = X_val_6mer.reshape(X_val_6mer.shape[0], 1, 4 ** 6 + 256, 1)

                X_test_4mer = X_test_4mer.reshape(X_test_4mer.shape[0], 1, 4 ** 4 + 256, 1)
                X_test_5mer = X_test_5mer.reshape(X_test_5mer.shape[0], 1, 4 ** 5 + 256, 1)
                X_test_6mer = X_test_6mer.reshape(X_test_6mer.shape[0], 1, 4 ** 6 + 256, 1)

                X_train = [X_train_4mer, X_train_5mer, X_train_6mer]
                X_dev = [X_val_4mer, X_val_5mer, X_val_6mer]
                X_test = [X_test_4mer, X_test_5mer, X_test_6mer]

                y_train_onehot = tf.one_hot(Y_train, depth=num_classes)
                y_val_onehot = tf.one_hot(Y_dev, depth=num_classes)

                batch_size = 64
                num_epochs = 50
                model = BERTE.get_model(input_4mer_shape, input_5mer_shape, input_6mer_shape, num_classes)

                history = BERTE.run_experiment(model, X_train, y_train_onehot, X_dev, y_val_onehot, batch_size=batch_size,
                                         num_epochs=num_epochs)

                model.save(f"{output_directory}/BERTE_retrained_model.keras")
                plot_training_metrics(history, "BERTE", report_dir)

                predicted_classes = model.predict(X_test)
                predicted_classes = np.argmax(predicted_classes, axis=1)
                acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, "BERTE", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        elif model_name == "TEClass2":
            if os.path.exists(f"{output_directory}/TEClass2_retrained_model"):
                info(f"Using the found model at {output_directory}/TEClass2_retrained_model. Skipping retraining....")
            else:
                TEClass2.superf_dict = superf_dict
                TEClass2.setup_device_and_log()
                # define the training arguments
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                training_args = TrainingArguments(
                    bf16=True,
                    fp16=False,
                    output_dir=work_dir,
                    num_train_epochs=100,  # 100
                    per_device_train_batch_size=210,  # 210
                    gradient_accumulation_steps=4,
                    per_device_eval_batch_size=210,  # 210
                    eval_strategy="steps",
                    eval_steps=2000,
                    eval_accumulation_steps=10,
                    save_strategy="steps",
                    save_steps=2000,
                    save_total_limit=2,
                    disable_tqdm=False,
                    dataloader_num_workers=threads,
                    dataloader_pin_memory=True,
                    dataloader_persistent_workers=True,
                    dataloader_drop_last=False,
                    group_by_length=False,
                    metric_for_best_model='eval_f1',
                    load_best_model_at_end=True,
                    warmup_steps=200,
                    weight_decay=0.01,
                    logging_steps=200,
                    logging_dir=f"{work_dir}/logs",
                    report_to="tensorboard",
                    optim="adamw_torch_fused",
                    ddp_backend=None,
                    ddp_find_unused_parameters=False,
                )

                joined_seqs = [x for x in SeqIO.parse(training_fasta, "fasta")]
                joined_seqs.extend([x for x in SeqIO.parse(val_fasta, "fasta")])
                joined_seqs.extend([x for x in SeqIO.parse(test_fasta, "fasta")])
                SeqIO.write(joined_seqs, f"{output_directory}/data_for_training/TE_all.fasta", "fasta")

                dataset_train, dataset_valid, dataset_test, datadict_ = TEClass2.load_data(f"{output_directory}/data_for_training/TE_all.fasta",
                                                                         mode="T", training=0.8, valid=0.1, test=0.1)

                tokenizer = TEClass2.tokenizer_fun(PanTEon_dir)
                dataset_train = TEClass2.TransposonDataset(dataset_train, datadict_, tokenizer, train=True)  # apply with augmentation
                dataset_valid = TEClass2.TransposonDataset(dataset_valid, datadict_, tokenizer)
                dataset_test = TEClass2.TransposonDataset(dataset_test, datadict_, tokenizer)

                vocab_file = TEClass2.load_vocab(f"{PanTEon_dir}/data/5mer_vocab")
                model = TEClass2.get_model(vocab_file, num_classes)
                total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                info(f"Total trainable parameters:  {total_trainable:,}")

                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.config.use_cache = False

                sample_weight = dataset_train.sample_weight
                # instantiate the trainer class
                trainer = TEClass2.DNAFormer_Trainer(
                    sample_weight=sample_weight,
                    model=model,
                    args=training_args,
                    compute_metrics=TEClass2.compute_metrics,
                    train_dataset=dataset_train,
                    eval_dataset=dataset_valid
                )

                train_result = trainer.train()
                info(f"{train_result}")

                trainer.save_model(f"{output_directory}/TEClass2_retrained_model")

                my_metrics = trainer.evaluate()
                trainer.save_metrics('eval', my_metrics)

                history = trainer.state.log_history
                training_history = TEClass2.TrainingHistory(history)

                plot_training_metrics(training_history, "TEClass2", report_dir)

                pred_out = trainer.predict(dataset_test)
                logits = pred_out.predictions
                if isinstance(logits, tuple):
                    logits = logits[0]

                predicted_classes = np.argmax(logits, axis=1)
                acc, f1, rec, pre = metrics(pred_out.label_ids, predicted_classes, num_classes, "TEClass2", report_dir)
                model_metrics[model_name] = [acc, f1, rec, pre]

        end = time.time()
        info(f"{model_name} training done!! [{end - start}]......")

    # Training custom (user-made) models
    for model_name in custom_registry:
        custom_model = custom_registry[model_name]
        start = time.time()
        info(f"Starting {model_name} training....")

        if os.path.exists(output_directory + f'/{model_name}_retrained_model.keras'):
            info(f"Using the found model at {output_directory}/{model_name}_model.keras. Skipping retraining....")
        elif os.path.exists(output_directory + f'/{model_name}_retrained_model.pt'):
            info(f"Using the found model at {output_directory}/{model_name}_model.pt. Skipping retraining....")
        else:
            custom_model.superf_dict = superf_dict
            X_train, Y_train = custom_model.load_data(training_fasta)
            X_dev, Y_dev = custom_model.load_data(val_fasta)
            X_test, Y_test = custom_model.load_data(test_fasta)

            Y_train_one_hot = to_categorical(Y_train, int(num_classes))
            Y_dev_one_hot = to_categorical(Y_dev, int(num_classes))

            if getattr(custom_model, "batch_size"):
                batch_size = custom_model.batch_size
            else:
                batch_size = 512
            if getattr(custom_model, "num_epochs"):
                num_epochs = custom_model.num_epochs
            else:
                num_epochs = 100

            info(f"Training the custom model {model_name} with epochs={num_epochs} and batch size={batch_size}")

            model = custom_model.get_model(X_train.shape[1], num_classes)
            history = custom_model.run_experiment(model, X_train, Y_train_one_hot, Y_train, X_dev, Y_dev_one_hot, batch_size,
                                            num_epochs)

            if custom_model.DL_FRAMEWORK.lower() == "tensorflow":
                model.save(output_directory+f'/{model_name}_retrained_model.keras')

                predicted_classes = model.predict(X_test)

            elif custom_model.DL_FRAMEWORK.lower()  == "pytorch":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch.save(model.state_dict(), output_directory + f'/{model_name}_retrained_model.pt')
                history = TrainingHistory(history)

                X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32)
                X_test = X_test.to(device)

                with torch.no_grad():
                    logits = model(X_test)
                    predicted_classes = F.softmax(logits, dim=1)
                predicted_classes = predicted_classes.cpu().numpy()


            plot_training_metrics(history, model_name, report_dir)
            predicted_classes = np.argmax(predicted_classes, axis=1)
            acc, f1, rec, pre = metrics(Y_test, predicted_classes, num_classes, model_name, report_dir)
            model_metrics[model_name] = [acc, f1, rec, pre]

        end = time.time()
        info(f"{model_name} training done!! [{end - start}]......")

    if len(model_metrics) > 0:
        df = pd.DataFrame(model_metrics)
        order = [0, 3, 2, 1]
        df = df.iloc[order].copy()
        df.index = ["Accuracy", "Precision", "Recall", "F1-score"]
        df.to_csv('PanTEon_training_reports.csv', index=True)

        info(f"Training complete. Consolidated results in the test dataset:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        print(df)


def inference(fasta_file, work_dir, threads, class_num, models, output_directory, inv_superf_dict, prefix, min_prob, custom_registry, PanTEon_dir):

    dict_predictions = {TE.id.split("#")[0]: [] for TE in SeqIO.parse(TE_library, "fasta")}
    used_models = ["SeqID"]

    # Inference with PanTEon in-built models
    for model_name in models:
        start = time.time()
        info(f"Starting {model_name} Prediction....")
        used_models.append(model_name)

        if model_name == "NeuralTE":
            if os.path.exists(f"{output_directory}/NeuralTE_retrained_model.keras"):
                NeuralTE.all_wicker_class = {value: key for key, value in inv_superf_dict.items()}
                NeuralTE.class_num = class_num
                NeuralTE.inverted_all_wicker_class = inv_superf_dict
                internal_kmer_sizes = [1, 3]
                terminal_kmer_sizes = [1, 2, 3]
                project_dir = os.path.dirname(os.path.abspath(__file__))
                X, Y, _, _, labels = NeuralTE.load_data(internal_kmer_sizes,
                                                               terminal_kmer_sizes,
                                                               fasta_file,
                                                               work_dir, project_dir, threads)

                model = load_model(f"{output_directory}/NeuralTE_retrained_model.keras", compile=False)
                y_preds_probs = model.predict(X)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/NeuralTE_retrained_model.keras). Have you trained this model before (using the training module)? ")

        elif model_name == "CREATE":
            if os.path.exists(f"{output_directory}/CREATE_retrained_model.keras"):
                k = 7
                l = 600

                X_kmer = CREATE.get_kmer_data(fasta_file, k)
                X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, pow(4, k), 1)
                X_kmer = X_kmer.astype("float64")
                X_oh = CREATE.get_oh_data(fasta_file, l)
                labels = CREATE.get_label_data(fasta_file, mode="P").tolist()

                model, attention_model = CREATE.create_attn_model(k, l, class_num)
                model.load_weights(f"{output_directory}/CREATE_retrained_model.keras")
                y_preds_probs = model.predict([X_kmer, X_oh])
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/CREATE_retrained_model.keras). Have you trained this model before (using the training module)? ")

        elif model_name == "ClassifyTE":
            if os.path.exists(f"{output_directory}/ClassifyTE_retrained_model.pkl"):
                ClassifyTE.script_dir = PanTEon_dir
                X, labels = ClassifyTE.load_data(fasta_file, threads, mode="P")
                labels = labels.tolist()

                model = joblib.load(f"{output_directory}/ClassifyTE_retrained_model.pkl")
                y_preds_probs = model.predict_proba(X)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/ClassifyTE_retrained_model.pkl). Have you trained this model before (using the training module)? ")
                sys.exit(0)

        elif model_name == "DeepTE":
            if os.path.exists(f"{output_directory}/DeepTE_retrained_model.keras"):
                X, labels = DeepTE.load_data(fasta_file, mode="P")
                labels = labels.tolist()
                X = X.reshape(X.shape[0], 1, 16384, 1)
                model = load_model(f"{output_directory}/DeepTE_retrained_model.keras", compile=False)
                y_preds_probs =  model.predict(X)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/DeepTE_retrained_model.keras). Have you trained this model before (using the training module)? ")

        elif model_name == "TERL":
            if os.path.exists(f"{output_directory}/TERL_Classify_retrained_model"):
                max_len = 19926
                X, _ = TERL.data_handler(fasta_file, max_len=max_len, mode="P")
                labels = [te.id for te in SeqIO.parse(fasta_file, "fasta")]

                with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                    tf.compat.v1.saved_model.loader.load(
                        sess,
                        ['serve'],
                        f"{output_directory}/TERL_Classify_retrained_model"
                    )

                    num_classes = sess.run('num_classes:0')
                    vocab_size = sess.run('vocab_size:0')
                    max_len = sess.run('max_len:0')
                    test_size = len(X)
                    test_batch_size = 32

                    # ****** CLASSIFICATION *******
                    y_preds_probs = np.empty((0, num_classes), dtype=np.float32)
                    for batch in range(0, test_size, test_batch_size):
                        x_batch = X[batch: batch + test_batch_size]

                        pre_xo = sess.run('one_hot_x:0', feed_dict={'pre_x:0': x_batch})
                        x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)

                        y_preds_probs = np.concatenate([
                            y_preds_probs,
                            sess.run(
                                'probabilities:0',
                                feed_dict={
                                    'x_input:0': x_batch,
                                    'is_training:0': False
                                }
                            )
                        ])
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/TERL_Classify_retrained_model). Have you trained this model before (using the training module)? ")

        elif model_name == "Inpactor2_Class":
            if os.path.exists(f"{output_directory}/Inpactor2_Class_retrained_model.keras"):
                X, labels = Inpactor2_Class.load_data(fasta_file, mode="P")
                scaler = joblib.load(f"{output_directory}/scaler.pkl")
                X_scaled = scaler.transform(X)

                pca = joblib.load(f"{output_directory}/pca.pkl")
                X_scaled_pca = pca.transform(X_scaled)
                model = load_model(f"{output_directory}/Inpactor2_Class_retrained_model.keras", compile=False)
                y_preds_probs = model.predict(X_scaled_pca)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/Inpactor2_Class_retrained_model.keras). Have you trained this model before (using the training module)? ")

        elif model_name == "Terrier":
            if os.path.exists(f"{output_directory}/Terrier_retrained_model.pt"):
                max_len = 15000
                X, labels, _ = Terrier.load_data(fasta_file, max_len, mode="P")
                labels = labels.tolist()
                batch_size = 32

                X_dataset = Terrier.InferenceDataset(X)
                X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, num_workers=0)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                with open(f"{output_directory}/root.pkl", "rb") as f:
                    root = pickle.load(f)
                vocab_size = len("ACGTN") + 1

                model = Terrier.TerrierNet(root=root, vocab_size=vocab_size).to(device)
                state = torch.load(f"{output_directory}/Terrier_retrained_model.pt", map_location=device)
                model.load_state_dict(state)
                model.eval()

                with torch.no_grad():
                    dummy = torch.zeros(1, X.shape[1], dtype=torch.long, device=device)
                    _ = model(dummy)

                test_logits_list, all_probs = [], []
                with torch.no_grad():
                    for xb in X_loader:
                        xb = xb.to(device)
                        logits = model(xb)
                        test_logits_list.append(logits.cpu())
                        probs = torch.softmax(logits, dim=-1)  # (B, C)
                        all_probs.append(probs.cpu().numpy())
                test_logits = torch.cat(test_logits_list, dim=0)
                pred_nodes = greedy_predictions(test_logits, root)
                y_preds_probs = np.concatenate(all_probs, axis=0)  # (N, C)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/Terrier_retrained_model.pt). Have you trained this model before (using the training module)? ")

        elif model_name == "BERTE":
            if os.path.exists(f"{output_directory}/BERTE_retrained_model.keras"):
                X4, X5, X6, labels = BERTE.load_data(fasta_file, bert_model_name=f"{PanTEon_dir}/tools/bert-mini", vocab_file=f"{PanTEon_dir}/data/kmer_vocab.txt", mode="P")
                labels = labels.tolist()

                X4 = X4.reshape(X4.shape[0], 1, 4 ** 4 + 256, 1)
                X5 = X5.reshape(X5.shape[0], 1, 4 ** 5 + 256, 1)
                X6 = X6.reshape(X6.shape[0], 1, 4 ** 6 + 256, 1)

                X_dataset = [X4, X5, X6]

                model = load_model(f"{output_directory}/BERTE_retrained_model.keras", compile=False)
                y_preds_probs = model.predict(X_dataset)
            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/Inpactor2_Class_retrained_model.keras). Have you trained this model before (using the training module)? ")

        elif model_name == "TEClass2":
            if os.path.exists(f"{output_directory}/TEClass2_retrained_model"):
                dataset_predict, datadict_, labels = TEClass2.load_data(TE_library, 'P')
                tokenizer = TEClass2.tokenizer_fun(PanTEon_dir)
                new_dataset = TEClass2.TransposonDataset(dataset_predict, datadict_, tokenizer, train=False)

                model = TEClass2.AutoModelForSequenceClassification.from_pretrained(
                     f"{output_directory}/TEClass2_retrained_model",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()

                pred_args = TEClass2.TrainingArguments(
                    output_dir=f"{work_dir}/infer",
                    per_device_eval_batch_size=128,
                    dataloader_num_workers=20,
                    dataloader_pin_memory=True,
                    report_to="none",
                    fp16=False,
                    bf16=torch.cuda.is_available(),  # usa bf16 en GPU si está disponible
                    disable_tqdm=True,
                )

                dummy_sample_weight = new_dataset.sample_weight

                infer_trainer = TEClass2.DNAFormer_Trainer(
                    sample_weight=dummy_sample_weight,
                    model=model,
                    args=pred_args,
                    compute_metrics=None,  # o deja tu compute_metrics si quieres métricas (si hay labels)
                    train_dataset=None,  # no entrenamos
                    eval_dataset=new_dataset  # el dataset nuevo va aquí
                )

                pred_out = infer_trainer.predict(new_dataset)
                logits = pred_out.predictions
                if isinstance(logits, tuple):
                    logits = logits[0]
                y_preds_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

            else:
                error(f"{model_name}'s trained model was not found (at path {output_directory}/Inpactor2_Class_retrained_model.keras). Have you trained this model before (using the training module)? ")

        if model_name == "Terrier":
            y_pred_idx = [int(n.name.replace("SUPERF::", "").split("::")[1]) for n in pred_nodes]
        else:
            y_pred_idx = y_preds_probs.argmax(axis=1)

        y_pred_label = [inv_superf_dict[int(i)] for i in y_pred_idx]
        prob_of_pred = y_preds_probs[np.arange(len(y_pred_idx)), y_pred_idx]

        df = pd.DataFrame(
            {
                "id": labels,
                "predicted_class": y_pred_label,
                "probability": prob_of_pred,
            }
        )
        df.to_csv(f"{prefix}_PanTEon_{model_name}.csv", index=False)

        final_seqs = []
        for TE in SeqIO.parse(TE_library, "fasta"):
            # remove previous classification if any
            original_name = TE.id.split("#")[0]
            position = labels.index(TE.id)
            new_classification = y_pred_label[position] if prob_of_pred[position] >= min_prob else "Unknown"

            # To save the merged report of classification predictions across all models
            dict_predictions[original_name].append(new_classification)

            TE.id = original_name + "#" + new_classification
            if len(TE.description.split(" ")) > 1:
                complement = " ".join(TE.description.split(" ")[1:])
                TE.id += " " + complement
            TE.description = ""
            final_seqs.append(TE)
        SeqIO.write(final_seqs, f"{prefix}_PanTEon_{model_name}.fasta", "fasta")
        end = time.time()
        info(f"{model_name}'s Prediction done!! [{end - start}]......")

    # Training custom (user-made) models
    for model_name in custom_registry:
        custom_model = custom_registry[model_name]
        start = time.time()
        info(f" Starting {model_name} Prediction....")
        used_models.append(model_name)

        if os.path.exists(output_directory + f'/{model_name}_retrained_model.keras') or os.path.exists(output_directory + f'/{model_name}_retrained_model.pt'):
            X, labels = custom_model.load_data(fasta_file, mode="P")

            if custom_model.DL_FRAMEWORK.lower() == "tensorflow":
                model = load_model(output_directory + f"/{model_name}_retrained_model.keras", compile=False)
                y_preds_probs = model.predict(X)
            elif custom_model.DL_FRAMEWORK.lower()  == "pytorch":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = custom_model.get_model(X.shape[1], class_num).to(device)
                state = torch.load(output_directory + f'/{model_name}_retrained_model.pt', map_location=device)
                model.load_state_dict(state)
                model.eval()

                X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits = model(X)
                    y_preds_probs = F.softmax(logits, dim=1)
                y_preds_probs = y_preds_probs.cpu().numpy()
            else:
                error(f"No DL_FRAMEWORK found: {custom_model.DL_FRAMEWORK.lower()}")

        else:
            error(
                f"{model_name}'s trained model was not found (at path {output_directory}/{model_name}_retrained_model.keras). Have you trained this model before (using the training module)? ")

        y_pred_idx = y_preds_probs.argmax(axis=1)
        y_pred_label = [inv_superf_dict[int(i)] for i in y_pred_idx]
        prob_of_pred = y_preds_probs[np.arange(len(y_pred_idx)), y_pred_idx]

        df = pd.DataFrame(
            {
                "id": labels,
                "predicted_class": y_pred_label,
                "probability": prob_of_pred,
            }
        )
        df.to_csv(f"{prefix}_PanTEon_{model_name}.csv", index=False)

        final_seqs = []
        for TE in SeqIO.parse(TE_library, "fasta"):
            # remove previous classification if any
            original_name = TE.id.split("#")[0]
            position = labels.index(TE.id)
            new_classification = y_pred_label[position] if prob_of_pred[position] >= min_prob else "Unknown"

            # To save the merged report of classification predictions across all models
            dict_predictions[original_name].append(new_classification)

            TE.id = original_name + "#" + new_classification
            if len(TE.description.split(" ")) > 1:
                complement = " ".join(TE.description.split(" ")[1:])
                TE.id += " " + complement
            TE.description = ""
            final_seqs.append(TE)
        SeqIO.write(final_seqs, f"{prefix}_PanTEon_{model_name}.fasta", "fasta")
        end = time.time()
        info(f"{model_name}'s Prediction done!! [{end - start}]......")

    rows = []
    for k, v in dict_predictions.items():
        rows.append([k, *v])
    dict_predictions_df = pd.DataFrame(rows, columns=used_models)
    dict_predictions_df.to_csv(f"{prefix}_PanTEon_consolidated_report.csv", index=False)


def load_config(json_path):
    """
    Carga un archivo JSON de configuración y retorna sus variables.

    Parámetros:
        json_path (str): Ruta al archivo JSON (por ejemplo, 'config.json').

    Retorna:
        tuple: (superf_dict, inv_superf_dict, num_classes, min_prob, species_group)
    """
    if not os.path.exists(json_path):
        error(f"The configuration JSON file {json_path} was not found.")
    with open(json_path, 'r') as f:
        data = json.load(f)

    superf_dict = data.get('superf_dict', {})
    inv_superf_dict = {int(k): v for k, v in data.get('inv_superf_dict', {}).items()}  # convierte claves a int
    num_classes = data.get('num_classes', 0)
    min_prob = data.get('min_prob', 0.0)
    species_group = data.get('species_group', 'unknown')

    return superf_dict, inv_superf_dict, num_classes, min_prob, species_group


def generate_dict_classification(TE_library):
    superf_dict, inv_superf_dict = {}, {}

    # quick verification:
    bad_seqs = [str(te.id) for te in SeqIO.parse(TE_library, "fasta") if "#" not in str(te.id)]
    if len(bad_seqs) > 0:
        print(f"[ERROR] there are some ID sequences without required character '#':")
        for bad_seq in bad_seqs:
            print(f"    -> {bad_seq}")
        sys.exit(0)

    classification = set([str(te.id).split(" ")[0].split("#")[1] for te in SeqIO.parse(TE_library, "fasta")])
    i = 0
    for cls in classification:
        superf_dict[cls] = i
        inv_superf_dict[i] = cls
        i += 1

    return superf_dict, inv_superf_dict, len(classification)


def create_config_json(output_path, superf_dict, inv_superf_dict, num_classes, min_prob):
    inv_superf_dict_str = {str(k): v for k, v in inv_superf_dict.items()}
    config = {
        "superf_dict": superf_dict,
        "inv_superf_dict": inv_superf_dict_str,
        "num_classes": num_classes,
        "min_prob": min_prob
    }

    output_path = Path(output_path).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def check_num_samples(TE_library, output_dir):
    kept_superf = []
    dict_superf = {}
    TEs_in_lib = [te for te in SeqIO.parse(TE_library, "fasta")]
    for te in TEs_in_lib:
        classification = te.id.split(" ")[0].split("#")[1]
        if classification in dict_superf.keys():
            dict_superf[classification] += 1
        else:
            dict_superf[classification] = 1

        if dict_superf[classification] == 10:
            kept_superf.append(classification)

    final_seqs = [te for te in SeqIO.parse(TE_library, "fasta") if te.id.split(" ")[0].split("#")[1] in kept_superf]
    if len(TEs_in_lib) == len(final_seqs):
        # No changes, so we return the same TE_lib
        return TE_library
    else:
        # some TEs will be eliminated:
        info("Some Orders/superfamilies have less than 10 sequences, so they will be ignored.")
        info("The ignored orders/superfamilies are:")
        for superf in dict_superf.keys():
            if dict_superf[superf] < 10:
                print(f"    -> {superf}: {dict_superf[superf]} seqs.")

        SeqIO.write(final_seqs, f"{output_dir}/TE_library_clean.fasta", "fasta")
        return f"{output_dir}/TE_library_clean.fasta"


def check_taxon_in_db(metadata_df, taxon, tax_cols):
    if taxon is None:
        return None

    taxon_norm = taxon.strip().lower()

    universe = set()
    for c in tax_cols:
        if c in metadata_df.columns:
            vals = (
                metadata_df[c]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            universe.update(v for v in vals if v and v != "nan")

    # Check exact match
    if taxon_norm in universe:
        return taxon_norm

    # Si no está, busca matches cercanos
    close = difflib.get_close_matches(taxon_norm, sorted(universe), n=10, cutoff=0.6)

    # También sugiere coincidencias por substring (útil para términos largos)
    contains = [x for x in universe if taxon_norm in x]
    contains = sorted(contains)[:10]

    msg = [f"Taxon '{taxon}' was not found in PanTEon Database."]
    if close:
        msg.append("Most similar matches: " + ", ".join(close))
    if contains and (not close or contains != close):
        msg.append("Other similar matches: " + ", ".join(contains))

    error("\n".join(msg))


def check_req_class_in_db(fasta_file, req_class):
    if req_class is None:
        return None

    req_class_norm = req_class.strip().lower()

    # Construye universo de tokens a partir del FASTA
    universe = set()
    with open(fasta_file, "r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if "#" not in header:
                continue
            after_hash = header.split("#", 1)[1].strip()
            class_part = after_hash.split(None, 1)[0] if after_hash else ""
            tokens = [t.strip().lower() for t in class_part.split("/") if t.strip()]
            universe.update(tokens)

    if req_class_norm in universe:
        return req_class_norm

    close = difflib.get_close_matches(req_class_norm, sorted(universe), n=10, cutoff=0.6)
    contains = sorted([x for x in universe if req_class_norm in x])[:10]

    msg = [f"Required TE classification '{req_class}' was not found at PanTEon Database."]
    if close:
        msg.append("Most similar matches: " + ", ".join(close))
    if contains and (not close or contains != close):
        msg.append("Other similar matches: " + ", ".join(contains))

    error("\n".join(msg))


def library(base_path, taxon, req_class, view_only):
    data_path = base_path / "data"

    try:
        metadata_file = next(data_path.glob("PanTEon_Database_metadata_v*.csv"))
        fasta_file = next(data_path.glob("PanTEon_Database_v*.fasta"))
    except StopIteration:
        error("PanTEon database files not found in data directory")

        # Extract version (v.X.Y.Z)
    version_pattern = r"_v([0-9]+(?:\.[0-9]+)*)"
    version_match = re.search(version_pattern, metadata_file.name)
    version = version_match.group(1) if version_match else "unknown"

    info(f"Using database version: v{version}")

    # ---- Checks ----
    if not data_path.exists():
        error(f"Data directory not found: {data_path}")

    for file_path in [metadata_file, fasta_file]:
        if not file_path.exists():
            error(f"Required file not found: {file_path}")

        if file_path.stat().st_size == 0:
            error(f"Required file is empty: {file_path}")

    # ---- Load metadata ----
    try:
        metadata_df = pd.read_csv(
            metadata_file,
            header=0  # first row contains column names
        )
    except Exception as e:
        error(f"Failed to load metadata file {metadata_file}: {e}")

    if metadata_df.empty:
        error(f"Metadata DataFrame is empty: {metadata_file}")

    # 3) species allowed by taxon (match in any taxonomy column)
    tax_cols = ["Kingdom", "Phylum", "Class", "Order", "Family", "Species"]

    # Checking that the taxon and req_class are present in the database
    taxon = check_taxon_in_db(metadata_df, taxon, tax_cols)
    req_class = check_req_class_in_db(fasta_file, req_class)

    if taxon is None:
        # ALL species allowed
        allowed_species = set(
            metadata_df["Species"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
    else:
        taxon = taxon.strip().lower()
        mask = False
        for c in tax_cols:
            if c in metadata_df.columns:
                mask = mask | (
                        metadata_df[c].astype(str).str.strip().str.lower() == taxon
                )

        allowed_species = set(
            metadata_df.loc[mask, "Species"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # --------------------------------------------------
        # 2) TE class filtering
        # --------------------------------------------------
    if req_class is not None:
        req_class = req_class.strip().lower()

    kept = []
    superfamily_counter = Counter()

    for rec in SeqIO.parse(str(fasta_file), "fasta"):
        desc = rec.description
        if "#" not in desc:
            continue

        after_hash = desc.split("#", 1)[1].strip()
        parts = after_hash.split(None, 1)
        if len(parts) < 2:
            continue

        class_part, species = parts
        species = species.strip().lower()

        class_tokens = [x.strip().lower() for x in class_part.split("/") if x.strip()]

        # ---- apply filters ----
        species_ok = species in allowed_species
        class_ok = True if req_class is None else (req_class in class_tokens)

        if species_ok and class_ok:
            kept.append(rec)

            superfamily = class_tokens[-1].upper()
            superfamily_counter[superfamily] += 1

    if view_only:
        info("\n" + "=" * 60)
        info(" PanTEon TE Library Report")
        info("=" * 60)
        info(f"Taxon filter      : {taxon}")
        info(f"TE class filter   : {req_class}")
        info(f"Total sequences   : {len(kept)}")
        info("-" * 60)

        if superfamily_counter:
            info(f"{'Superfamily':25s} | {'Count':>10s}")
            info("-" * 60)
            for sf, count in superfamily_counter.most_common():
                info(f"{sf:25s} | {count:10d}")
        else:
            info("No sequences selected.")

        info("=" * 60 + "\n")
    else:
        SeqIO.write(kept, f"PanTEonDB_{taxon}_{req_class}.fasta", "fasta")
    return metadata_df


def parse_header_get_id_and_label(rec_id: str, level: int = -1) -> Tuple[str, str]:
    """
    Para headers tipo: ID#A/B/C
    - ID base: lo de antes de '#'
    - label: por defecto el último componente (C). Con --level N (1-based) eliges A=1, B=2, C=3.
    """
    if "#" not in rec_id:
        # Si no hay '#', devolvemos todo como id y label vacío
        return rec_id, ""

    base_id, trail = rec_id.split("#", 1)
    # rec.id en BioPython no contiene espacios (solo el primer token de la descripción),
    # así que 'trail' ya debería ser "A/B/C" sin ' Folsomia'
    parts = trail.split("/")
    if not parts:
        return base_id, ""

    if level == -1:
        label = parts[-1]
    else:
        # level es 1-based; si se sale de rango, tomamos el último
        idx = level - 1
        label = parts[idx] if 0 <= idx < len(parts) else parts[-1]

    return base_id, label


def read_labels_from_fasta(path: str, level: int = -1) -> Dict[str, str]:
    """
    Lee un FASTA y devuelve un dict: base_id -> label extraído.
    """
    labels = {}
    for rec in SeqIO.parse(path, "fasta"):
        bid, lab = parse_header_get_id_and_label(rec.id, level=level)
        if lab:  # solo guardamos si encontramos etiqueta
            # Si hay duplicados de ID base, conservamos la primera aparición
            labels.setdefault(bid, lab)
    return labels


def eval_from_fasta(true_fasta, pred_fasta, level, out_confusion, out_report):
    true_map = read_labels_from_fasta(true_fasta, level=level)
    pred_map = read_labels_from_fasta(pred_fasta, level=level)

    ids_true = set(true_map.keys())
    ids_pred = set(pred_map.keys())
    common = sorted(ids_true & ids_pred)

    if not common:
        raise SystemExit("No hay IDs comunes entre ambos FASTA (comparando la parte antes de '#').")

    y_true: List[str] = [true_map[i] for i in common]
    y_pred: List[str] = [pred_map[i] for i in common]

    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=pd.Index(labels, name="True\\Pred"), columns=labels)
    cm_df.to_csv(out_confusion, index=True)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_report, index=True)

    info(f"IDs in TRUE: {len(ids_true)} | IDs in PRED: {len(ids_pred)} | Used common IDs: {len(common)}")
    info(f"Confusion Matrix -> {out_confusion}")
    info(f"Classification report -> {out_report}")
    info("\nQuick look (by class):")
    info(report_df.loc[labels][["precision", "recall", "f1-score", "support"]].round(4))


def load_custom_classifiers(custom_dir):
    registry = {}
    custom_path = Path(custom_dir)
    if not custom_path.exists() or not custom_path.is_dir():
        return registry  # 0 custom models, ok

    for pyf in sorted(custom_path.glob("*.py")):
        if pyf.name.startswith("_"):
            continue

        spec = importlib.util.spec_from_file_location(pyf.stem, str(pyf))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        name = pyf.stem
        if (hasattr(mod, "load_data") and hasattr(mod, "get_model") and hasattr(mod, "run_experiment")
                and getattr(mod, "superf_dict") and getattr(mod, "DL_FRAMEWORK")):
            if getattr(mod, "DL_FRAMEWORK").lower() in ["tensorflow", "pytorch"]:
                registry[str(name)] = mod
            else:
                error(f"The custom model {name} has an unknown DL_FRAMEWORK value {getattr(mod, 'DL_FRAMEWORK').lower()}. Allowed values are tensorflow and pytorch")
        else:
            error(f"The custom model {name} does not have the required functions: load_data, get_model, run_experiment and/or the required attributes: superf_dict and DL_FRAMEWORK")

    return registry


if __name__ == '__main__':
    args = parse_args()
    module = args.module.lower()
    PanTEon_dir = os.path.dirname(os.path.abspath(__file__))

    if module == "training":
        TE_library = args.fasta
        work_dir = args.work_dir
        threads = args.threads
        model_list = args.models
        output_directory = args.models_directory
        min_prob = args.min_prob

        info("Executing PanTEon training module ... ")

        models = []
        if model_list is None:
            models = []
            info("None in-built model selected (using -n/--models parameter). Trying to get custom models ... ")
        elif model_list.lower() == "all":
            models = ["NeuralTE", "Terrier", "CREATE", "ClassifyTE", "DeepTE", "Inpactor2_Class", "TERL", "BERTE",
                      "TEClass2"]
        else:
            for m in model_list.split(","):
                if m in ["NeuralTE", "Terrier", "CREATE", "ClassifyTE", "DeepTE", "Inpactor2_Class", "TERL", "BERTE",
                         "TEClass2"]:
                    models.append(m)
                else:
                    info(
                        f"The model {m} isn't in the valid options. Remember that the compatible models are: NeuralTE, Terrier, CREATE, ClassifyTE, DeepTE, Inpactor2_Class, TERL, BERTE, TEClass2")

            if len(models) == 0:
                info(f"there is not any compatible model in the -n parameter. Value got={model_list}. Trying to get custom models ...")

        # To call the custom classifiers done by the user
        custom_registry = load_custom_classifiers(f"{PanTEon_dir}/Custom_classifiers")
        if model_list is not None:
            info("Using the following in-built ML/DL models: ")
            for m in models:
                print(f"    -> {m}")
        if len(custom_registry) > 0:
            info("Using the following customs ML/DL models: ")
            for m in custom_registry.keys():
                print(f"    -> {m}")

        if not os.path.exists(TE_library):
            error(f"The input fasta file {TE_library} was not found.")

        os.makedirs(work_dir, exist_ok=True)

        if output_directory is None:
            error(
                "For training mode you must indicate the directory where the trained models will be saved (-d parameter)")

        if os.path.exists(f"{PanTEon_dir}/data_for_training"):
            shutil.rmtree(f"{PanTEon_dir}/data_for_training")
        os.makedirs(output_directory, exist_ok=True)
        TE_library = check_num_samples(TE_library, output_directory)
        superf_dict, inv_superf_dict, num_classes = generate_dict_classification(TE_library)

        info(f"Using the following classes ({num_classes}) ... ")
        print(f"    -> {list(superf_dict.keys())}")
        training(TE_library, work_dir, threads, models, num_classes, output_directory, superf_dict, custom_registry, PanTEon_dir)
        create_config_json(f"{output_directory}/training_variables.json", superf_dict, inv_superf_dict, num_classes,
                           min_prob)

    elif module == "inference":
        TE_library = args.fasta
        work_dir = args.work_dir
        threads = args.threads
        model_list = args.models
        output_directory = args.models_directory
        prefix = args.prefix
        min_prob = args.min_prob

        info("Executing PanTEon inference module ... ")
        models = []
        if model_list is None:
            models = []
            info("None in-built model selected (using -n/--models parameter). Trying to get custom models ... ")
        elif model_list.lower() == "all":
            models = ["NeuralTE", "Terrier", "CREATE", "ClassifyTE", "DeepTE", "Inpactor2_Class", "TERL", "BERTE",
                      "TEClass2"]
        else:
            for m in model_list.split(","):
                if m in ["NeuralTE", "Terrier", "CREATE", "ClassifyTE", "DeepTE", "Inpactor2_Class", "TERL", "BERTE",
                         "TEClass2"]:
                    models.append(m)
                else:
                    info(
                        f"The model {m} isn't in the valid options. Remember that the compatible models are: NeuralTE, Terrier, CREATE, ClassifyTE, DeepTE, Inpactor2_Class, TERL, BERTE, TEClass2")

            if len(models) == 0:
                error(f"there is not any compatible model in the -n parameter. Value got={model_list}. Trying to get custom models ...")

        # To call the custom classifiers done by the user
        custom_registry = load_custom_classifiers(f"{PanTEon_dir}/Custom_classifiers")

        info("Using the following ML/DL models: ")
        for m in models:
            print(f"    -> {m}")
        info("Using the following customs ML/DL models: ")
        for m in custom_registry.keys():
            print(f"    -> {m}")

        if not os.path.exists(TE_library):
            error(f"The input fasta file {TE_library} was not found.")

        os.makedirs(work_dir, exist_ok=True)

        if output_directory is not None:
            info(f"PanTEon inference (prediction) module using trained models located at: {output_directory} ... ")
            if not os.path.exists(output_directory):
                error(f"The model's directory path {output_directory} was not found.")
        else:
            error(
                f"for inference (prediction) module, you must indicate the path to the directory containing the training models with the parameter: -d [trained_model_dir]")

        if prefix is None:
            error("for inference (prediction) mode, you must indicate the -p parameter.")

        superf_dict, inv_superf_dict, num_classes, min_prob, species_group = load_config(
            f"{output_directory}/training_variables.json")

        info("Obtained the following information:")
        print(f"    -> Available classes ({num_classes}):")
        print(f"    -> {list(superf_dict.keys())}")
        print(f"    -> probability threshold = {min_prob} ")

        inference(TE_library, work_dir, threads, num_classes, models, output_directory, inv_superf_dict, prefix,
                  min_prob, custom_registry, PanTEon_dir)

    elif module == "library":
        taxon = args.taxon
        req_class = args.req_class
        view_only = args.view_only

        base_path = Path(__file__).resolve().parent

        info("Executing PanTEon library module ... ")
        library(base_path, taxon, req_class, view_only)

    elif module == "evaluation":
        info("Executing PanTEon evaluation module ... ")
        true_fasta = args.true_fasta
        pred_fasta = args.pred_fasta
        level = args.level
        out_confusion = args.out_confusion
        out_report = args.out_report
        eval_from_fasta(true_fasta, pred_fasta, level, out_confusion, out_report)

    else:
        error(f"No module found: {module}")

