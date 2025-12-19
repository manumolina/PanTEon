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
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModel, BertModel
import re
import json
from Bio.SeqIO import parse
from collections import Counter
from itertools import product
import pickle
import time
import math

import torch
from torch.nn.utils.rnn import pad_sequence

ALPHABET = 'ACGT'
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
				   'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
				   'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
				   'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
				   'KOLOBOK': 28, 'ACADEM-1': 29}

# ====================
# CONFIGURACIÓN GPU
# ====================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

#### Funciones auxiliares para el load_dataset optimziado ################################
# map A,C,G,T -> 0,1,2,3 con tabla de traduccion
_trans = np.full(256, 255, dtype=np.uint8)
_trans[ord('A')] = 0; _trans[ord('C')] = 1; _trans[ord('G')] = 2; _trans[ord('T')] = 3

class CustomTokenizer:
	def __init__(self, vocab_file, k):
		self.k = k
		self.vocab = []
		self.token2id = {}
		with open(vocab_file, 'r') as f:
			for idx, line in enumerate(f):
				token = line.strip()
				self.vocab.append(token)
				self.token2id[token] = idx

		# Special tokens
		self.pad_token = '[PAD]'
		self.unk_token = '[UNK]'
		self.cls_token = '[CLS]'
		self.sep_token = '[SEP]'

		self.pad_token_id = self.token2id[self.pad_token]
		self.unk_token_id = self.token2id[self.unk_token]
		self.cls_token_id = self.token2id[self.cls_token]
		self.sep_token_id = self.token2id[self.sep_token]

	def tokenize(self, sequence):
		"""Tokeniza una secuencia en k-mers con sliding window"""
		tokens = []
		for i in range(len(sequence) - self.k + 1):
			kmer = sequence[i:i + self.k]
			if kmer in self.token2id:
				tokens.append(kmer)
			else:
				tokens.append(self.unk_token)
		return tokens

	def convert_tokens_to_ids(self, tokens):
		return [self.token2id.get(token, self.unk_token_id) for token in tokens]


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


def load_data(TE_lib, mode="T", stride=1, max_tokens=503, bert_model_name="tools/bert-mini", vocab_file="data/kmer_vocab.txt"):
	seqs_1k, seqs_1250, seqs_1500, labels = [], [], [], []

	for te in SeqIO.parse(TE_lib, "fasta"):
		seq = str(te.seq).upper()
		if mode == "T":
			superf = te.id.split(" ")[0].split("#")[1]
			labels.append(superf_dict[superf])
		elif mode == "P":
			labels.append(te.id)
		else:
			return None, None, None, None

		seqs_1500.append(truncate_seq_indv(seq, 1500))
		seqs_1250.append(truncate_seq_indv(seq, 1250))
		seqs_1k.append(truncate_seq_indv(seq, 1000))

	labels = np.asarray(labels)

	# --- k-mer counts (vectorizado + súper rápido) ---
	X4_counts = np.vstack([kmer_counts_numpy(s, 4, stride) for s in seqs_1k])
	X5_counts = np.vstack([kmer_counts_numpy(s, 5, stride) for s in seqs_1250])
	X6_counts = np.vstack([kmer_counts_numpy(s, 6, stride) for s in seqs_1500])

	X4_counts = normalize_rows(X4_counts.astype(np.float32))
	X5_counts = normalize_rows(X5_counts.astype(np.float32))
	X6_counts = normalize_rows(X6_counts.astype(np.float32))

	# --- BERT (batched, GPU si hay y sin crear el mega-tensor) ---
	model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)     # :contentReference[oaicite:7]{index=7}

	# Tu tokenizer personalizado con vocab de k-mers
	tokenizer = CustomTokenizer(vocab_file, 4)                                        # :contentReference[oaicite:8]{index=8}
	X4_cls = bert_cls_embeddings_batched(seqs_1k,    model, tokenizer, k=4,  batch_size=256, max_tokens=max_tokens)
	tokenizer = CustomTokenizer(vocab_file, 5)
	X5_cls = bert_cls_embeddings_batched(seqs_1250,  model, tokenizer, k=5,  batch_size=256, max_tokens=max_tokens)
	tokenizer = CustomTokenizer(vocab_file, 6)
	X6_cls = bert_cls_embeddings_batched(seqs_1500,  model, tokenizer, k=6,  batch_size=256, max_tokens=max_tokens)

	# --- concat embeddings + counts (como en tu código, pero sin listas gigantes) ---
	X4 = np.concatenate([X4_cls, X4_counts], axis=1)
	X5 = np.concatenate([X5_cls, X5_counts], axis=1)
	X6 = np.concatenate([X6_cls, X6_counts], axis=1)

	return X4.astype(np.float32), X5.astype(np.float32), X6.astype(np.float32), labels


def kmer_counts_numpy(seq: str, k: int, stride: int = 1) -> np.ndarray:
	"""Cuenta k-mers con NumPy (sin bucles Python); ignora ventanas con N/char no-ACGT."""
	a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
	d = _trans[a]                     # 0..3 o 255 (inválido)
	if len(d) < k:
		return np.zeros(4**k, dtype=np.uint32)

	# ventanas deslizantes (L-k+1, k) y máscara de validez
	w = np.lib.stride_tricks.sliding_window_view(d, k)[::stride]
	valid = np.all(w != 255, axis=1)
	w = w[valid]
	if w.size == 0:
		return np.zeros(4**k, dtype=np.uint32)

	# codifica cada ventana en base 4: sum(w * 4**pos)
	pow4 = (4 ** np.arange(k-1, -1, -1, dtype=np.uint32))
	codes = (w.astype(np.uint32) * pow4).sum(axis=1)
	return np.bincount(codes, minlength=4**k).astype(np.uint32)


def normalize_rows(X: np.ndarray) -> np.ndarray:
	s = X.sum(axis=1, keepdims=True)
	s[s == 0] = 1
	return X / s

# Batching para BERT (sin construir todo el tensor a la vez)
def prepare_input_batch(seqs, tokenizer, max_tokens=503):
	input_ids = []
	for seq in seqs:
		toks = tokenizer.tokenize(seq)
		toks = [tokenizer.cls_token] + toks[:max_tokens-3] + [tokenizer.sep_token, tokenizer.sep_token]
		ids = tokenizer.convert_tokens_to_ids(toks)
		input_ids.append(torch.tensor(ids, dtype=torch.long))
	pad_id = tokenizer.pad_token_id
	input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
	attn = (input_ids != pad_id).long()
	return input_ids, attn

@torch.no_grad()
def bert_cls_embeddings_batched(seqs, model, tokenizer, k, batch_size=256, max_tokens=503, device=None):
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device).eval()
	hid = model.config.hidden_size
	out = np.empty((len(seqs), hid), dtype=np.float32)
	for i in range(0, len(seqs), batch_size):
		chunk = seqs[i:i+batch_size]
		tokenizer.k = k
		ids, mask = prepare_input_batch(chunk, tokenizer, max_tokens)
		ids = ids.to(device); mask = mask.to(device)
		outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
		cls = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
		out[i:i+len(chunk)] = cls
	return out

############################################################################################

def get_model(input_4mer_shape, input_5mer_shape, input_6mer_shape, num_class):
	input_a = tf.keras.Input(shape=input_4mer_shape, name="input_4")
	input_b = tf.keras.Input(shape=input_5mer_shape, name="input_5")
	input_c = tf.keras.Input(shape=input_6mer_shape, name="input_6")

	model1 = tf.keras.layers.Conv2D(100, (1, 3), activation='relu', input_shape=input_4mer_shape)(input_a)
	model1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model1)
	model1 = tf.keras.layers.Dropout(0.5)(model1)
	model1 = tf.keras.layers.Flatten()(model1)

	model2 = tf.keras.layers.Conv2D(100, (1, 5), activation='relu', input_shape=input_5mer_shape)(input_b)
	model2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model2)
	model2 = tf.keras.layers.Dropout(0.5)(model2)
	model2 = tf.keras.layers.Flatten()(model2)

	model3 = tf.keras.layers.Conv2D(100, (1, 7), activation='relu', input_shape=input_6mer_shape)(input_c)
	model3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(model3)
	model3 = tf.keras.layers.Dropout(0.5)(model3)
	model3 = tf.keras.layers.Flatten()(model3)

	concat = tf.keras.layers.concatenate([model1, model2, model3], axis=1, name="concat_layer")
	output = tf.keras.layers.Dense(num_class, activation='softmax')(concat)

	model = tf.keras.Model(inputs=[input_a, input_b, input_c], outputs=[output])

	opt = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
	# loss function
	# loss_fn = BinaryFocalLoss(gamma=2)
	loss_fn = tf.keras.losses.CategoricalCrossentropy()
	# Compile model
	model.compile(loss=loss_fn, optimizer=opt, metrics=[f1_m])
	return model


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
	lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_m', mode="max", factor=0.01, patience=10, verbose=1)
	early_stopping = EarlyStopping(monitor='val_f1_m', mode="max", patience=50, restore_best_weights=True)
	history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
						validation_data=(X_dev, Y_dev), callbacks=[lr_scheduler, early_stopping], verbose=1)
	return history


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
	plt.plot(history.history['val_f1_m'])
	plt.plot(history.history['f1_m'])
	plt.legend(['val_f1_m', 'train_f1_m'], loc='upper right')
	plt.xlabel('Epoch')
	plt.ylabel('f1_m')
	plt.title('Epoch vs f1_m')
	plt.savefig('Train_Curve.png', bbox_inches='tight', dpi=500)

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
	plt.savefig('Train_Curve_los.png', bbox_inches='tight', dpi=500)


def truncate_seq_indv(seq: str, L: int) -> str:
	if len(seq) <= L:
		return seq
	left = math.ceil(L / 2)   # un carácter más a la izquierda
	right = L // 2
	return seq[:left] + seq[-right:]

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
		stride = 1
		max_tokens = 503
		batch_size = 512
		epochs = 100
		input_4mer_shape = (1, 4 ** 4 + 256, 1)
		input_5mer_shape = (1, 4 ** 5 + 256, 1)
		input_6mer_shape = (1, 4 ** 6 + 256, 1)

		print("### Step 0: Starting to load and transform the dataset......")
		start = time.time()

		X4, X5, X6, Y = load_data(TE_library)

		##########################
		# 0. Save the data
		os.makedirs("data_for_training/", exist_ok=True)
		np.save("data_for_training/X4.npy", X4)
		np.save("data_for_training/X5.npy", X5)
		np.save("data_for_training/X6.npy", X6)
		np.save("data_for_training/Y.npy", Y)

		"""X4 = np.load("data_for_training/X4.npy")
		X5 = np.load("data_for_training/X5.npy")
		X6 = np.load("data_for_training/X6.npy")
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
		# División del conjunto de datos
		X_train_4mer, X_rem_4mer, Y_train, y_rem = train_test_split(X4, Y, test_size=validation_size, random_state=seed, stratify=Y)
		X_test_4mer, X_val_4mer, Y_test, Y_dev = train_test_split(X_rem_4mer, y_rem, test_size=0.5, random_state=seed,
																  stratify=y_rem)

		X_train_5mer, X_rem_5mer, Y_train, y_rem = train_test_split(X5, Y, test_size=validation_size, random_state=seed, stratify=Y)
		X_test_5mer, X_val_5mer, Y_test, Y_dev = train_test_split(X_rem_5mer, y_rem, test_size=0.5, random_state=seed,
																  stratify=y_rem)

		X_train_6mer, X_rem_6mer, Y_train, y_rem = train_test_split(X6, Y, test_size=validation_size, random_state=seed, stratify=Y)
		X_test_6mer, X_val_6mer, Y_test, Y_dev = train_test_split(X_rem_6mer, y_rem, test_size=0.5, random_state=seed,
																  stratify=y_rem)


		print("\nDataset k=4 shapes:")
		print(f"X_train shape: {X_train_4mer.shape}")
		print(f"X_dev shape: {X_val_4mer.shape}")
		print(f"X_test shape: {X_test_4mer.shape}")

		print("\nDataset k=5 shapes:")
		print(f"X_train shape: {X_train_5mer.shape}")
		print(f"X_dev shape: {X_val_5mer.shape}")
		print(f"X_test shape: {X_test_5mer.shape}")

		print("\nDataset k=6 shapes:")
		print(f"X_train shape: {X_train_6mer.shape}")
		print(f"X_dev shape: {X_val_6mer.shape}")
		print(f"X_test shape: {X_test_6mer.shape}")

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
		X4 = None
		X5 = None
		X6 = None
		Y = None

		end = time.time()
		print(f"### Step 1 Done !! [{end - start}]......")

		##########################
		# 2. Preprocess input data
		print("### Step 2: Starting the features preprocessing steps ......")
		start = time.time()

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

		end = time.time()
		print(f"### Step 2 Done !! [{end - start}]......")

		###########################
		# 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
		print("### Step 3: Starting the labels preprocessing steps ......")
		start = time.time()

		y_train_onehot = tf.one_hot(Y_train, depth=num_classes)
		y_val_onehot = tf.one_hot(Y_dev, depth=num_classes)
		y_test_onehot = tf.one_hot(Y_test, depth=num_classes)

		end = time.time()
		print(f"### Step 3 Done !! [{end - start}]......")

		###########################
		# 4. Fit model on training data
		print("### Step 4: Starting the fitting ......")
		start = time.time()

		batch_size = 64
		num_epochs = 50
		model = get(input_4mer_shape, input_5mer_shape, input_6mer_shape, num_classes)
		print(model.summary())
		tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

		history = run_experiment(model, X_train, y_train_onehot, X_dev, y_val_onehot, batch_size=batch_size, num_epochs=num_epochs)

		end = time.time()
		print(f"### Step 4 Done !! [{end - start}]......")

		###########################
		# 5.  save the model
		print("### Step 5: Saving the trained model ......")
		start = time.time()

		model.save('trained_models/BERTE_retrained_model.h5')

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
		X4, X5, X6, labels = load_data(TE_library, mode=script_mode)
		labels = labels.tolist()
		end = time.time()
		print(f"### Step 0 Done !! [{end - start}]......")

		##########################
		# 1. Preprocess input data
		print("### Step 1: Starting the preprocessing step......")
		start = time.time()
		X4 = X4.reshape(X4.shape[0], 1, 4 ** 4 + 256, 1)
		X5 = X5.reshape(X5.shape[0], 1, 4 ** 5 + 256, 1)
		X6 = X6.reshape(X6.shape[0], 1, 4 ** 6 + 256, 1)

		X_dataset = [X4, X5, X6]

		end = time.time()
		print(f"### Step 1 Done !! [{end - start}]......")

		###########################
		# 2. Load the already trained model
		print("### Step 2: Starting to load the model......")
		start = time.time()

		model = load_model("trained_models/BERTE_retrained_model.h5")

		end = time.time()
		print(f"### Step 2 Done !! [{end - start}]......")

		###########################
		# 3. Predict the labels
		print("### Step 3: Starting to predict the TE classification......")
		start = time.time()

		y_preds_probs = model.predict(X_dataset)

		end = time.time()
		print(f"### Step 3 Done !! [{end - start}]......")

		###########################
		# 4. Save results in fasta and in csv
		print("### Step 4: Starting to save the results......")
		start = time.time()

		inv_superf_dict = {
			0: 'CLASSI/LTR/LTR', 1: 'CLASSI/LTR/COPIA', 2: 'CLASSI/LTR/GYPSY', 3: 'CLASSI/LTR/ERV',
			4: 'CLASSI/LTR/BELPAO',
			5: 'CLASSI/LINE/LINE', 6: 'CLASSI/LINE/I', 7: 'CLASSI/LINE/L1', 8: 'CLASSI/LINE/RTE', 9: 'CLASSI/DIRS/DIRS',
			10: 'CLASSI/PLE/PLE', 11: 'CLASSI/SINE/SINE', 12: 'CLASSI/SINE/TRNA', 13: 'CLASSII/HELITRON/HELITRON',
			14: 'CLASSII/CRYPTON/CRYPTON', 15: 'CLASSII/TIR/HAT', 16: 'CLASSII/TIR/MERLIN', 17: 'CLASSII/TIR/P',
			18: 'CLASSII/TIR/TIR', 19: 'CLASSII/TIR/TC1MARINER', 20: 'CLASSII/TIR/MULE', 21: 'CLASSII/TIR/PIFHARBINGER',
			22: 'CLASSII/TIR/CACTA', 23: 'CLASSII/TIR/PIGGYBAC', 24: 'CLASSI/LINE/CR1', 25: 'CLASSI/LINE/R1',
			26: 'CLASSI/LTR/LARD', 27: 'CLASSI/SINE/ALU', 28: 'CLASSII/TIR/KOLOBOK', 29: 'CLASSII/TIR/ACADEM-1'
		}
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
