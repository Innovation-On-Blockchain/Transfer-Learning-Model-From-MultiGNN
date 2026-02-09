# Transfer Learning with Multi-GNN for Node-Level Sanctions Classification

A project for **Math 168: Introduction to Networks** applying graph neural network (GNN) techniques to classify sanctioned entities on blockchain transaction networks. We adapt IBM's open-source [Multi-GNN](https://github.com/IBM/Multi-GNN) framework — originally designed for edge-level anti-money laundering (AML) detection — into a node-level classifier that identifies sanctioned addresses in a cryptocurrency transaction graph.

This project demonstrates how classical network concepts from Newman's *Networks* (2nd ed., Oxford, 2018) underpin modern machine learning on graphs.

---

## Table of Contents

- [Motivation](#motivation)
- [Connection to Newman's *Networks*](#connection-to-newmans-networks)
  - [Network Representation](#1-network-representation-newman-ch-6-secs-62646361)
  - [Degree and Neighborhood Aggregation](#2-degree-and-neighborhood-aggregation-newman-chs-67)
  - [Homophily and Assortative Mixing](#3-homophily-and-assortative-mixing-newman-assortative-mixing)
  - [Community Structure and Node Classification](#4-community-structure-and-node-classification-newman-ch-11)
  - [Message Passing](#5-message-passing-newman-ch-17)
- [Architecture](#architecture)
  - [Original IBM Multi-GNN (Edge-Level)](#original-ibm-multi-gnn-edge-level)
  - [Our Adaptation (Node-Level)](#our-adaptation-node-level)
- [Data](#data)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Motivation

Blockchain transaction networks form large directed graphs in which addresses (nodes) exchange value through transactions (edges). A small fraction of these addresses are sanctioned by regulatory authorities. Identifying them automatically from the graph structure is a **node classification** problem — one of the central tasks in network science and graph machine learning.

IBM's Multi-GNN was built to detect illicit *transactions* (edges). Our goal is different: we want to classify *addresses* (nodes) as sanctioned or non-sanctioned, then test whether a model trained on one blockchain can transfer to another. This edge-to-node shift requires rethinking how the GNN produces its final output, while keeping the powerful message-passing backbone intact.

---

## Connection to Newman's *Networks*

Each stage of this project maps onto foundational concepts from the textbook. Below we trace the connections explicitly.

### 1. Network Representation (Newman Ch. 6, Secs. 6.2–6.4/6.3/6.6.1)

Newman Ch. 6 introduces the mathematical language for describing networks: the **adjacency matrix** *A*, where *A<sub>ij</sub>* = 1 if node *i* connects to node *j* (Secs. 6.2–6.3), with extensions to **directed networks** (Sec. 6.4) and **weighted networks** (Sec. 6.3). The **incidence matrix** (Sec. 6.6.1), which lists edges explicitly by their endpoint nodes, provides an edge-centric representation of the same structure.

Our transaction graph is a **directed, weighted, attributed network**:

| Newman concept | Our implementation |
|---|---|
| Adjacency matrix *A* (Sec. 6.2) | `edge_index` tensor of shape `[2, num_edges]` — a sparse edge list representation, equivalent to enumerating the nonzero entries of *A* |
| Incidence matrix (Sec. 6.6.1) | The `edge_index` format is also analogous to the incidence matrix, which explicitly pairs each edge with its two endpoint nodes |
| Directed edges (Sec. 6.4) | Each column in `edge_index` is an ordered pair *(from_id, to_id)*, capturing the direction of value flow |
| Weighted / attributed edges (Sec. 6.3) | `edge_attr` tensor stores real-valued features per edge (amounts, timestamps, currency types) — a generalization of scalar edge weights *w<sub>ij</sub>* |
| Node metadata | `x` tensor holds node features; `y` tensor holds node labels (sanctioned / non-sanctioned) |

In Newman's notation, we are working with a graph *G* = (*V*, *E*) where |*V*| &#8776; 19,000 and |*E*| is determined by the transaction data. The graph is stored as a PyTorch Geometric `Data` object, which internally uses the sparse edge list format — closely related to both the adjacency matrix and incidence matrix representations discussed in Newman Ch. 6.

### 2. Degree and Neighborhood Aggregation (Newman Chs. 6/7)

Newman defines the **degree** *k<sub>i</sub>* of a node as the number of edges attached to it (Ch. 6), and for directed networks distinguishes **in-degree** and **out-degree** (Sec. 6.4). Chapter 7 extends these ideas to broader **structural measures** — centrality, clustering coefficients, path lengths — that quantify a node's position and importance within the network.

Degree is the simplest structural property of a node, yet it already carries meaningful information: a node with unusually high in-degree in a transaction graph may be an exchange or a mixer.

GNNs generalize these local structural concepts. At each layer, a node *i* aggregates information from its **neighborhood** *N*(*i*) — the set of nodes adjacent to *i*. After *L* layers of aggregation, node *i*'s representation encodes information from its *L*-hop neighborhood. In our model (*L* = 2 GNN layers, following the IBM hyperparameters), each node's final embedding captures the structural and feature patterns within two hops — encompassing the node's direct transaction partners and their partners.

This repeated local aggregation is why the **degree distribution** matters for GNN performance: nodes with very high degree aggregate over many neighbors (potentially over-smoothing), while isolated nodes receive no messages at all.

### 3. Homophily and Assortative Mixing (Newman: Assortative Mixing)

Newman devotes a chapter to **assortative mixing** — the tendency of nodes to connect to others that are similar to themselves. When similarity is measured by a categorical attribute (e.g., sanctioned vs. non-sanctioned), this property is called **homophily**: sanctioned addresses may preferentially transact with other sanctioned addresses, forming tightly connected substructures.

The entire premise of GNN-based node classification rests on a form of homophily. The message-passing layers assume that a node's label can be inferred from the features and labels of its neighbors. If the network exhibited no assortative mixing with respect to the sanctioned/non-sanctioned label, then neighborhood aggregation would provide no signal, and the GNN would perform no better than a classifier ignoring graph structure.

Newman quantifies assortative mixing through the **assortativity coefficient** *r* and the **mixing matrix** *e<sub>ij</sub>*. Although we do not compute *r* explicitly in this project, the fact that the GNN achieves above-chance F1 on the sanctioned class implies that the network exhibits at least partial assortative mixing by sanction status — a testable prediction that could be verified directly using Newman's assortativity framework.

### 4. Community Structure and Node Classification (Newman Ch. 11)

Node classification is closely related to **community detection** (Newman Ch. 11). Both tasks assign categorical labels to nodes based on structural patterns. In community detection, the labels are latent (discovered from the network); in node classification, we have partial ground truth and aim to generalize.

Newman describes several community detection approaches, including **spectral partitioning** using the eigenvectors of the graph Laplacian or modularity matrix (Ch. 11). While GNNs are not covered in Newman's textbook, they can be understood as a learned, nonlinear generalization of these classical spectral methods: the GNN layers perform operations on the adjacency structure (via message passing), but with learnable filters rather than fixed eigenvector projections.

Our class imbalance (~82 sanctioned vs. ~19,000 non-sanctioned) is *analogous* to the observation in Newman that real networks often have **heterogeneous community sizes**: meaningful communities can be very small relative to the network. Newman's discussion of resolution limits in modularity optimization illustrates how small groups can be overlooked by algorithms that favor large partitions. The class-weighted loss function we use (weighting the sanctioned class ~230:1) addresses a related problem in the supervised setting — ensuring the model does not simply assign every node to the majority class. Note that class imbalance handling is a machine-learning technique, not a topic Newman covers directly; the connection to heterogeneous community sizes is conceptual rather than methodological.

### 5. Message Passing (Newman Ch. 17)

Newman Ch. 17 introduces **message passing** in the context of belief propagation on networks — a framework from statistical physics in which each node sends a "message" to its neighbors summarizing its current state, and each node updates its own state based on the messages it receives. Through repeated rounds of message exchange, local information propagates globally through the network. Newman's treatment focuses on probabilistic inference (computing marginal distributions, percolation thresholds, epidemic spreading), but the underlying computational pattern is general.

Graph neural networks adopt this same iterative local-update pattern. In our GINe model, each GNN layer performs the update:

> **x**<sub>*i*</sub><sup>(*l*+1)</sup> = ( **x**<sub>*i*</sub><sup>(*l*)</sup> + ReLU( BN( GINEConv( **x**<sup>(*l*)</sup>, edge_index, edge_attr )<sub>*i*</sub> )) ) / 2

This is a residual message-passing step: node *i* receives aggregated "messages" from its neighbors (via GINEConv), applies a nonlinearity, and combines the result with its previous state. The residual connection (averaging with the previous state) prevents the loss of local information across layers — addressing the **over-smoothing** problem where, after too many rounds of message passing, all node representations converge to the same value. Newman's analysis of convergence and fixed-point behavior in belief propagation provides an analogy for why deep message passing can reach oversmoothed states, and motivates mechanisms like residual connections — though Newman does not discuss over-smoothing or residual connections directly, as these are concepts from the GNN literature.

The optional **edge update MLPs** extend the message-passing framework further: after nodes update, the edge representations are also updated based on the current states of their endpoint nodes. This corresponds to a richer message-passing scheme where both node and edge "beliefs" evolve together — a concept that has no direct analogue in Newman's belief propagation formulation but follows the same iterative local-update philosophy.

---

## Architecture

### Original IBM Multi-GNN (Edge-Level)

The IBM Multi-GNN classifies **edges** (transactions) as illicit or licit. After the GNN layers update node embeddings, the model reads out edge predictions by concatenating the embeddings of each edge's source and destination nodes with the edge's own features:

```
                                     ┌──────────────┐
Node features ──► node_emb ──►┌─────►│  GNN Layer 1  │──►┌─────► GNN Layer 2 ──►...
                              │      └──────────────┘    │
Edge features ──► edge_emb ──►┘   (+ optional edge MLPs) ┘

After GNN layers (EDGE readout):
  For each edge (i,j):  concat[ x_i  ||  x_j  ||  edge_attr_ij ]  ──► MLP(3×hidden → 2)
                         src_emb  dst_emb  edge_emb
```

### Our Adaptation (Node-Level)

We keep the GNN message-passing layers identical — they still update node embeddings using neighbor aggregation with edge features. The only change is the **readout**: instead of concatenating endpoint pairs per edge, we feed each node embedding directly into the classification MLP:

```
                                     ┌──────────────┐
Node features ──► node_emb ──►┌─────►│  GNN Layer 1  │──►┌─────► GNN Layer 2 ──►...
                              │      └──────────────┘    │
Edge features ──► edge_emb ──►┘   (+ optional edge MLPs) ┘

After GNN layers (NODE readout):
  For each node i:  x_i  ──► MLP(1×hidden → 2)
                    node_emb
```

| Component | Original (edge-level) | Adapted (node-level) |
|---|---|---|
| GNN backbone | GINEConv / GATConv with residual + batch norm | **Identical** |
| Edge update MLPs | `MLP([x_src \|\| x_dst \|\| edge_attr])` | **Identical** |
| Readout | `concat(x_src, x_dst, edge_attr)` per edge | `x_i` per node |
| Final MLP input | `3 x n_hidden` | `1 x n_hidden` |
| Prediction target | One label per edge | One label per node |
| Loss function | Weighted CrossEntropy (w ~ 6.3:1) | Weighted CrossEntropy (w ~ 230:1) |

---

## Data

The model expects three files in the working directory:

| File | Description |
|---|---|
| `formatted_transactions.parquet` | Edge list with columns `from_id`, `to_id`, and edge feature columns (timestamps, amounts, currencies, etc.) |
| `node_labels.parquet` | Node labels — first column is the node ID, last column is the binary label (1 = sanctioned, 0 = non-sanctioned) |
| `data_splits.json` | Train/validation split defined by edge indices — keys containing `train` and `val` map to lists of edge IDs |

**Class distribution:** ~82 sanctioned nodes vs. ~19,000 non-sanctioned nodes. The model automatically computes class weights from the training set to handle this imbalance.

**Node masks:** Train/validation node masks are derived from the edge-based splits — a node belongs to the training set if it appears as an endpoint of any training edge, and similarly for validation.

---

## Usage

### Running on Google Colab

1. Open `Training.ipynb` in Google Colab
2. Uncomment the file upload cell and upload your three data files
3. Run all cells sequentially

### Running Locally

```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn tqdm pyarrow
jupyter notebook Training.ipynb
```

### Hyperparameters

Default values are taken from the IBM Multi-GNN model settings (optimized for the GIN architecture on AML data):

| Parameter | Value | Description |
|---|---|---|
| `MODEL_TYPE` | `"gin"` | `"gin"` for GINe or `"gat"` for GATe |
| `N_HIDDEN` | 66 | Hidden dimension for node/edge embeddings |
| `N_GNN_LAYERS` | 2 | Number of message-passing layers (2-hop neighborhood) |
| `EDGE_UPDATES` | `True` | Enable edge update MLPs after each GNN layer |
| `LR` | 0.006 | Adam optimizer learning rate |
| `N_EPOCHS` | 100 | Training epochs |
| `DROPOUT` | 0.01 | Dropout in GNN layers |
| `FINAL_DROPOUT` | 0.1 | Dropout in final classification MLP |

### Outputs

| Output | Description |
|---|---|
| `best_model.pt` | Best checkpoint (by validation F1), including model state and optimizer state |
| `trained_node_gnn_weights.pt` | Final model weights with all hyperparameters saved for reloading |
| `training_results.png` | Confusion matrix and training curves (F1, precision, recall over epochs) |

---

## Repository Structure

```
.
├── README.md                  # This file
├── Training.ipynb             # Complete training pipeline (data loading → evaluation)
├── formatted_transactions.parquet   # (user-provided) Transaction edge list
├── node_labels.parquet              # (user-provided) Node-level sanction labels
└── data_splits.json                 # (user-provided) Train/val edge splits
```

---

## References

- **Newman, M. E. J.** (2018). *Networks* (2nd ed.). Oxford University Press.
  - Ch. 6: Mathematics of networks — adjacency matrix (Sec. 6.2), weighted networks (Sec. 6.3), directed networks (Sec. 6.4), incidence matrix (Sec. 6.6.1)
  - Ch. 7: Measures and metrics — degree, centrality, clustering, structural measures
  - Assortative mixing chapter — homophily, assortativity coefficient, mixing matrices
  - Ch. 11: Graph partitioning and community detection — spectral methods, modularity, resolution limits
  - Ch. 17: Message passing — belief propagation, iterative local updates on networks
- **Cardoso, W., Weber, M., et al.** (2022). [Multi-GNN: A Graph Neural Network Framework for Anti-Money Laundering](https://github.com/IBM/Multi-GNN). IBM Research.
- **Xu, K., Hu, W., Leskovec, J., & Jegelka, S.** (2019). How Powerful are Graph Neural Networks? *ICLR 2019*. (Theoretical foundation of the GIN architecture)
- **Velickovic, P., et al.** (2018). Graph Attention Networks. *ICLR 2018*. (Foundation of the GAT architecture)
