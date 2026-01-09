import os
import time
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DL_FRAMEWORK = "pytorch"
superf_dict = {' ': 0}
batch_size = 1024
num_epochs = 50

def load_data(fasta_path, mode="T"):
	"""
	mode="T": retorna (X, Y) donde:
	  - X: np.ndarray (N, D) counts de k-mers
	  - Y: np.ndarray (N,) con class_id (int)
	mode="P": retorna (X, labels_TEs)
	"""
	sequences = list(SeqIO.parse(fasta_path, "fasta"))
	k_range = (1, 6)

	# Construir vocabulario k-mer (A,C,G,T) para k=1..6
	all_kmer_set = set()
	k_min, k_max = k_range
	a = ("A", "C", "G", "T")
	for k in range(k_min, k_max + 1):
		for tup in product(a, repeat=k):
			all_kmer_set.add("".join(tup))
	all_kmers = sorted(all_kmer_set)
	kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

	if mode == "T":
		X = np.zeros((len(sequences), len(all_kmers)), dtype=np.int32)
		Y = np.full((len(sequences),), -1, dtype=np.int64)

		for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting k-mers (train)")):
			classification = seq.id.split(" ")[0].split("#")[1]
			if classification not in superf_dict:
				print(f"[ERROR] {classification} not found in superf_dict")
				continue

			Y[seq_idx] = superf_dict[classification]
			seq_str = str(seq.seq).upper()

			for k in range(k_range[0], k_range[1] + 1):
				for i in range(len(seq_str) - k + 1):
					kmer = seq_str[i:i + k]
					idx = kmer_index.get(kmer)
					if idx is not None:
						X[seq_idx, idx] += 1

		# filtra los que quedaron inválidos (por si hubo clases fuera del dict)
		mask = (Y >= 0)
		return X[mask], Y[mask]

	elif mode == "P":
		X = np.zeros((len(sequences), len(all_kmers)), dtype=np.int32)
		labels_TEs = []

		for seq_idx, seq in enumerate(tqdm(sequences, desc="Counting k-mers (predict)")):
			labels_TEs.append(seq.id)
			seq_str = str(seq.seq).upper()

			for k in range(k_range[0], k_range[1] + 1):
				for i in range(len(seq_str) - k + 1):
					kmer = seq_str[i:i + k]
					idx = kmer_index.get(kmer)
					if idx is not None:
						X[seq_idx, idx] += 1

		return X, labels_TEs

	return None, None


class SimpleMLP(nn.Module):
	def __init__(self, in_dim, num_classes):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, 200),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.BatchNorm1d(200),

			nn.Linear(200, 200),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.BatchNorm1d(200),

			nn.Linear(200, 200),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.BatchNorm1d(200),

			nn.Linear(200, num_classes)
		)

	def forward(self, x):
		return self.net(x)


def get_model(shape, num_classes):
	"""
	shape: D (int) número de features (k-mers)
	Retorna un nn.Module (no compila nada aquí como Keras)
	"""
	torch.manual_seed(0)
	return SimpleMLP(shape, num_classes)


def _to_class_index(y):
	y = np.asarray(y)
	if y.ndim == 2:
		return np.argmax(y, axis=1).astype(np.int64)
	return y.astype(np.int64)


def run_experiment(model, X_train, Y_train, labels, X_dev, Y_dev, batch_size, num_epochs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	hist = []

	# Convertir data a tensores
	X_train_t = torch.tensor(np.asarray(X_train), dtype=torch.float32)
	y_train_t = torch.tensor(_to_class_index(Y_train), dtype=torch.long)

	X_dev_t = torch.tensor(np.asarray(X_dev), dtype=torch.float32)
	y_dev_t = torch.tensor(_to_class_index(Y_dev), dtype=torch.long)

	train_loader = DataLoader(
		TensorDataset(X_train_t, y_train_t),
		batch_size=batch_size,
		shuffle=True,
		drop_last=False
	)

	# Optim + loss
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	best_val_acc = -1.0
	best_state = None

	for epoch in range(1, num_epochs + 1):
		t0 = time.time()
		model.train()
		running_loss = 0.0

		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)

			optimizer.zero_grad()
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * xb.size(0)

		train_loss = running_loss / len(train_loader.dataset)

		# Validación
		model.eval()
		with torch.no_grad():
			logits = model(X_dev_t.to(device))
			val_loss = criterion(logits, y_dev_t.to(device)).item()
			preds = torch.argmax(logits, dim=1).cpu().numpy()
			y_true = y_dev_t.cpu().numpy()
			val_acc = float((preds == y_true).mean())

		row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_m": val_acc}
		hist.append(row)

		# “best checkpoint” en memoria (mínimo viable)
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

		dt = time.time() - t0
		print(f"[Epoch {epoch:03d}/{num_epochs}] "
			  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={dt:.1f}s")

	# Restaurar mejor estado (si existió)
	if best_state is not None:
		model.load_state_dict(best_state)

	return hist
