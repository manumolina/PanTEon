# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
from Bio import SeqIO
import random
import math
import json
import seaborn as sn
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from typing import List, Tuple, Dict
import time
import pickle

try:
    from hierarchicalsoftmax import (
        SoftmaxNode,
        HierarchicalSoftmaxLazyLinear,
        HierarchicalSoftmaxLoss,
        greedy_accuracy,
        greedy_f1_score,
        greedy_predictions
    )
except Exception as e:
    raise SystemExit("Package 'hierarchicalsoftmax' couldn't be loaded. Install it with:  pip install hierarchicalsoftmax\nDetails: %s" % e)


class TerrierNet(nn.Module):
    def __init__(self, root: SoftmaxNode,
                 vocab_size: int, emb_dim: int = 18, n_layers: int = 4, k: int = 7,
                 growth: float = 1.96, first_channels: int = 64,
                 dropout: float = 0.248, penult: int = 1953):
        super().__init__()
        self.root = root
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        chans = [int(round(first_channels * (growth ** i))) for i in range(n_layers)]
        blocks, in_c = [], emb_dim
        for out_c in chans:
            blocks.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(dropout))
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_c = out_c
        self.conv = nn.Sequential(*blocks)

        self.penultimate = nn.Linear(chans[-1], penult)
        # Capa final jerárquica; out_features se deduce de root.layer_size
        self.hsoftmax = HierarchicalSoftmaxLazyLinear(root=root)

    def forward(self, x):
        x = self.embedding(x)        # (B, L, E)
        h = x.transpose(1, 2)        # (B, E, L)
        h = self.conv(h)             # (B, C, L')
        h = h.mean(dim=2)            # GAP
        z = torch.relu(self.penultimate(h))
        logits = self.hsoftmax(z)    # (B, root.layer_size), SIN softmax (logits crudos)
        return logits


class TransposonDataset(Dataset):
    def __init__(self, X: np.ndarray, y_node_ids: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.int64))
        self.y = torch.from_numpy(y_node_ids.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

# Superfamily dict
superf_dict = {'LTR': 0, 'COPIA': 1, 'GYPSY': 2, 'ERV': 3, 'BELPAO': 4, 'LINE': 5, 'I': 6, 'L1': 7,
               'RTE': 8, 'DIRS': 9, 'PLE': 10, 'SINE': 11, 'TRNA': 12, 'HELITRON': 13, 'CRYPTON': 14,
               'HAT': 15, 'MERLIN': 16, 'P': 17, 'TIR': 18, 'TC1MARINER': 19, 'MULE': 20,
               'PIFHARBINGER': 21, 'CACTA': 22, 'PIGGYBAC': 23, 'CR1': 24, 'R1': 25, 'LARD': 26, 'ALU': 27,
               'KOLOBOK': 28, 'ACADEM-1': 29}
order_dict = {'LTR': 0, 'LINE': 1, 'DIRS': 2, 'PLE': 3, 'SINE': 4, 'HELITRON': 5, 'CRYPTON': 6, 'TIR': 7 }

# -----------------------------
# Utilidades
# -----------------------------

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Parser de FASTA y features
# -----------------------------

DNA_ALPHABET = "ACGTN"
BASE2IDX = {b: i+1 for i,b in enumerate(DNA_ALPHABET)}  # 0=PAD

def clean_seq(seq):
    s = str(seq).upper()
    return "".join(ch if ch in DNA_ALPHABET else "N" for ch in s)


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """Codifica nucleótidos a enteros (A=1,C=2,G=3,T=4), padding con 0; truncado si excede max_len."""
    arr = np.zeros(max_len, dtype=np.int64)
    L = min(len(seq), max_len)
    for i in range(L):
        arr[i] = BASE2IDX.get(seq[i], 0)
    return arr


# -----------------------------
# Etiquetas y Jerarquía
# -----------------------------
def build_hierarchy(order_labels: List[str], superf_labels: List[str], phi: float = 1.02):
    """Construye el árbol SoftmaxNode: root -> orders -> superfamilies.
    Asigna alpha=1.0 al root y alpha=phi a cada nodo 'order' (para ponderar Superfamily)."""
    root = SoftmaxNode("ROOT")
    root.alpha = 1.0

    # Crear nodos de orden
    order_names = sorted(set(order_labels))
    order_nodes: Dict[str, SoftmaxNode] = {name: SoftmaxNode(f"ORDER::{name}", parent=root) for name in order_names}
    # alpha en el padre de superfamily (orden) controla el peso del nivel inferior
    for node in order_nodes.values():
        node.alpha = phi  # pondera la pérdida de Superfamily

    # Crear nodos de superfamilia bajo cada orden según aparezcan
    # Permitimos que una superfamilia exista en múltiples órdenes si ocurre en datos (clave por par)
    sf_pairs = sorted(set(zip(order_labels, superf_labels)))
    superf_nodes: Dict[Tuple[str, str], SoftmaxNode] = {}
    for o, s in sf_pairs:
        o_node = order_nodes[o]
        s_node = SoftmaxNode(f"SUPERF::{o}::{s}", parent=o_node)
        superf_nodes[(o, s)] = s_node

    # Indexar (necesario para layer/loss/metrics)
    root.set_indexes()

    # Mapas útiles
    order_id_map = {name: order_nodes[name] for name in order_names}
    superf_id_map = {(o, s): superf_nodes[(o, s)] for (o, s) in sf_pairs}

    return root, order_id_map, superf_id_map

def targets_to_node_ids(root: SoftmaxNode, order_labels: List[str], superf_labels: List[str],
                        order_node_map: Dict[str, SoftmaxNode],
                        superf_node_map: Dict[Tuple[str, str], SoftmaxNode]) -> np.ndarray:
    """Devuelve un vector de índices de nodos (globales) para las superfamilias (hojas)."""
    nodes = []
    for o, s in zip(order_labels, superf_labels):
        nodes.append(superf_node_map[(o, s)])
    ids = root.get_node_ids(nodes)
    return np.asarray(ids, dtype=np.int64)


def build_superf_id_maps(root, superf_node_map):
    """Devuelve dos diccionarios complementarios:
       pair2id:  (order, superfamily) -> node_id (entero)
       id2pair:  node_id (entero)      -> (order, superfamily)
    """
    pair2id = {}
    id2pair = {}
    for (order, superf), node in superf_node_map.items():
        node_id = int(root.get_node_ids([node])[0])  # ID global de esa hoja
        pair2id[(order, superf)] = node_id
        id2pair[node_id] = (order, superf)
    return pair2id, id2pair


def load_data(TE_lib, max_len, mode="T"):
    seqs = []
    classifications_superf = []
    classifications_order = []

    for te in SeqIO.parse(TE_lib, "fasta"):
        seqs.append(encode_sequence(clean_seq(te.seq), max_len))
        if mode == "T":
            superfamily = te.id.split("#")[1].split(" ")[0].split("/")[-1]
            order = te.id.split("#")[1].split(" ")[0].split("/")[-2]
            classifications_superf.append(superf_dict[superfamily])
            classifications_order.append(order_dict[order])
        elif mode == "P":
            classifications_order.append(te.id)
        else:
            return None, None, None

    X = np.asarray(seqs)
    Y_order = np.asarray(classifications_order)
    Y_superf = np.asarray(classifications_superf)
    return X, Y_order, Y_superf


def run_experiment(model: nn.Module, root: SoftmaxNode,
          train_loader: DataLoader, val_loader: DataLoader, device,
          epochs=100, max_lr=1e-3, patience=10, weight_decay=0.0) -> Tuple[nn.Module, List[Dict]]:
    model = model.to(device)
    opt = Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(opt, max_lr=max_lr, epochs=epochs,
                           steps_per_epoch=len(train_loader), anneal_strategy="cos", pct_start=0.3)
    hs_loss = HierarchicalSoftmaxLoss(root)

    best_val = math.inf
    best_state = None
    hist = []
    no_imp = 0

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = hs_loss(logits, yb)
            loss.backward()
            opt.step()
            scheduler.step()
            run_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        tr_loss = run_loss / max(1, seen)

        model.eval()
        with torch.no_grad():
            val_logits_list, val_targets_list = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_logits_list.append(logits.cpu())
                val_targets_list.append(yb)
            val_logits = torch.cat(val_logits_list, dim=0)
            val_targets = torch.cat(val_targets_list, dim=0)
            val_metrics = evaluate(val_logits, val_targets, root)
            val_loss = float(HierarchicalSoftmaxLoss(root)(val_logits, val_targets).item())

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **val_metrics}
        hist.append(row)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} ",
              f"F1_superf={val_metrics['superf_f1_weighted']:.4f} F1_order={val_metrics['order_f1_weighted']:.4f}")

        if val_loss < best_val - 1e-9:
            best_val, best_state, no_imp = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping @ epoch {epoch}. best val_loss={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist


def evaluate(logits: torch.Tensor, targets: torch.Tensor, root: SoftmaxNode) -> Dict[str, float]:
    acc_superf = greedy_accuracy(logits, targets, root)
    f1_superf = greedy_f1_score(logits, targets, root, average="weighted")

    # Accuracy a nivel Order (depth=1)
    acc_order = greedy_accuracy(logits, targets, root, max_depth=1)
    f1_order = greedy_f1_score(logits, targets, root, average="weighted", max_depth=1)

    return {
        "superf_acc": float(acc_superf),
        "superf_f1_weighted": float(f1_superf),
        "order_acc": float(acc_order),
        "order_f1_weighted": float(f1_order),
    }


def plot_training_metrics(history):
    history = pd.DataFrame(history)
    # plot metrics

    plt.figure()
    plt.plot(history['superf_f1_weighted'])
    plt.plot(history['order_f1_weighted'])
    plt.legend(['superf_f1_weighted', 'order_f1_weighted'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('f1')
    plt.title('Epoch vs f1_m')
    plt.savefig('Train_Curve_DeepTE.png', bbox_inches='tight', dpi=500)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss evolution")
    plt.tight_layout()
    plt.savefig("train_curve_loss.png", dpi=300)
    plt.close('all')


def metrics(y_true_labels, y_pred_labels,num_classes, prefix: str):
    inv = sorted(set(list(y_true_labels) + list(y_pred_labels)))
    # Mapear a índices compactos para matriz de confusión
    label_to_idx = {lab: i for i, lab in enumerate(inv)}
    y_true_idx = np.array([label_to_idx[l] for l in y_true_labels], dtype=int)
    y_pred_idx = np.array([label_to_idx[l] for l in y_pred_labels], dtype=int)

    print(f"Metrics for {prefix}")
    print('Accuracy:', accuracy_score(y_true_idx, y_pred_idx))
    print('F1 score:', f1_score(y_true_idx, y_pred_idx, average='weighted'))
    print('Recall:', recall_score(y_true_idx, y_pred_idx, average='weighted'))
    print('Precision:', precision_score(y_true_idx, y_pred_idx, average='weighted'))
    print('\n clasification report:\n', classification_report(y_true_idx, y_pred_idx))
    print('\n confusion matrix:\n', confusion_matrix(y_true_idx, y_pred_idx))
    # Creamos la matriz de confusión
    snn_cm = confusion_matrix(y_true_idx, y_pred_idx)

    # Visualizamos la matriz de confusión
    snn_df_cm = pd.DataFrame(snn_cm, range(num_classes), range(num_classes))
    plt.figure(figsize=(20, 14))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12})  # font size
    plt.savefig(f'confusionMatrix_{prefix}.png', bbox_inches='tight', dpi=500)


# Para órdenes, convertir ambos a su nodo de orden
def to_order_label(node_name: str) -> str:
    # 'SUPERF::<ORDER>::<SUPERF>' -> <ORDER>
    parts = node_name.split("::")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2 and parts[0] == "ORDER":
        return parts[1]
    return node_name


if __name__ == '__main__':
    max_len = 15000
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

        X, Y_order, Y_superf = load_data(TE_library, max_len)

        ##########################
        # 0. Save the data
        os.makedirs("data_for_training/", exist_ok=True)
        np.save("data_for_training/X.npy", X)
        np.save("data_for_training/Y_order.npy", Y_order)
        np.save("data_for_training/Y_superf.npy", Y_superf)

        """X = np.load("data_for_training/X.npy")
        Y_order = np.load("data_for_training/Y_order.npy")
        Y_superf = np.load("data_for_training/Y_superf.npy")"""

        num_classes = int(np.max(Y_superf) + 1)
        vocab_size = len(DNA_ALPHABET) + 1  # +PAD

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. data split: 80% train, 10% dev and 10% test
        print("### Step 1: Starting the dataset splitting ......")
        start = time.time()

        root, order_node_map, superf_node_map = build_hierarchy(Y_order, Y_superf, phi=1.02)
        y_node_ids = targets_to_node_ids(root, Y_order, Y_superf, order_node_map, superf_node_map)

        pair2id, id2pair = build_superf_id_maps(root, superf_node_map)
        unique, counts = np.unique(np.asarray(y_node_ids), return_counts=True)
        for cls, count in zip(unique, counts):
            (order_label, superf_label) = id2pair[cls]
            print(f"Class [{cls}] {order_label}/{superf_label}: {count} samples")

        # Etiquetas legibles por cada id de hoja
        leaf_nodes = [node for node in root.node_list if node and not node.children]
        leaf_id_to_label = {root.node_to_id[node]: node.name.replace("SUPERF::", "") for node in leaf_nodes}
        order_nodes = [node for node in root.children]
        order_id_to_label = {root.node_to_id[node]: node.name.replace("ORDER::", "") for node in order_nodes}

        dev = device_auto()
        print(f"[INFO] Device: {dev}")

        validation_size = 0.2
        seed = 7
        np.random.seed(seed)
        random.seed(seed)
        X_train, X_test_dev, Y_train, Y_test_dev = train_test_split(X, y_node_ids, test_size=validation_size, random_state=seed, stratify=y_node_ids)

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

        # Optimizing memory usage
        X = None
        Y = None

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ##########################
        # 2. Preprocess input data
        print("### Step 2: Starting the features preprocessing steps ......")
        start = time.time()

        batch_size = 32
        train_dataset = TransposonDataset(X_train, Y_train)
        val_dataset = TransposonDataset(X_dev, Y_dev)
        test_dataset = TransposonDataset(X_test, Y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dev_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices

       ###########################
        # 4. Fit model on training data
        print("### Step 4: Starting the fitting ......")
        start = time.time()

        model = TerrierNet(root=root, vocab_size=vocab_size)
        model, history = run_experiment(model, root, train_loader, dev_loader, dev, 100, 1e-3, patience=10, weight_decay=0.0)

        end = time.time()
        print(f"### Step 4 Done !! [{end - start}]......")

        ###########################
        # 5.  save the model
        print("### Step 5: Saving the trained model ......")
        start = time.time()

        torch.save(model.state_dict(), "trained_models/Terrier_retrained_model.pt")
        with open("trained_models/root.pkl", "wb") as f:
            pickle.dump(root, f)
        json.dump(leaf_id_to_label, open("trained_models/leaf_id_to_label.json", "w"))
        json.dump(order_id_to_label, open("trained_models/order_id_to_label.json", "w"))

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
        test_metrics = evaluate(test_logits, test_targets, root)
        print("[TEST]", test_metrics)

        # Reportes con etiquetas legibles
        # Predicciones greedy al nivel de hojas (superfamilia) y al nivel 1 (orden)

        pred_nodes = greedy_predictions(test_logits, root)  # lista de nodos (hojas)
        y_pred_superf_labels = [n.name.replace("SUPERF::", "") for n in pred_nodes]
        y_true_superf_labels = [leaf_id_to_label[int(i)] for i in test_targets.numpy().tolist()]

        metrics(y_true_superf_labels, y_pred_superf_labels, int(np.max(Y_superf) + 1), prefix="superfamily")

        y_pred_order_labels = [to_order_label(n.name) for n in pred_nodes]
        y_true_order_labels = [name.split("::")[1] for name in
                               y_true_superf_labels]  # viene como '<ORDER>::<SUPERF>' tras replace?
        # Ajuste: leaf_id_to_label guardó '<ORDER>::<SUPERF>' porque hicimos replace('SUPERF::','')
        metrics(y_true_order_labels, y_pred_order_labels, int(np.max(Y_order) + 1), prefix="order")

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

        X, labels, _ = load_data(TE_library, max_len, mode=script_mode)
        batch_size = 32
        labels = labels.tolist()

        X_dataset = InferenceDataset(X)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, num_workers=0)

        end = time.time()
        print(f"### Step 0 Done !! [{end - start}]......")

        ##########################
        # 1. Preprocess input data
        print("### Step 1: Starting the preprocessing step......")
        start = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("trained_models/root.pkl", "rb") as f:
            root = pickle.load(f)

        leaf_id_to_label = json.load(open("trained_models/leaf_id_to_label.json"))
        order_id_to_label = json.load(open("trained_models/order_id_to_label.json"))
        vocab_size = len(DNA_ALPHABET) + 1

        end = time.time()
        print(f"### Step 1 Done !! [{end - start}]......")

        ###########################
        # 2. Load the already trained model
        print("### Step 2: Starting to load the model......")
        start = time.time()

        model = TerrierNet(root=root, vocab_size=vocab_size).to(device)
        state = torch.load("trained_models/Terrier_retrained_model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        end = time.time()
        print(f"### Step 2 Done !! [{end - start}]......")

        ###########################
        # 3. Predict the labels
        print("### Step 3: Starting to predict the TE classification......")
        start = time.time()

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
        pred_nodes = greedy_predictions(test_logits, root)  # lista de nodos (hojas)

        y_preds_probs = np.concatenate(all_probs, axis=0)  # (N, C)

        end = time.time()
        print(f"### Step 3 Done !! [{end - start}]......")

        ###########################
        # 4. Save results in fasta and in csv
        print("### Step 4: Starting to save the results......")
        start = time.time()

        inv_superf_dict = {value: key for key, value in superf_dict.items()}
        y_pred_idx = [int(n.name.replace("SUPERF::", "").split("::")[1]) for n in pred_nodes]
        y_pred_label = [inv_superf_dict[int(i)] for i in y_pred_idx]
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

