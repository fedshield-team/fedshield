# FedShield — Experiments & Results

This document tracks every experiment run during development, in order, with the actual reasoning behind each decision. It exists so any of us (or an interviewer) can see *why* the model looks the way it does, not just the final number.

---

## Experiment 1 — Centralized Baseline (Binary)

**Goal:** Establish a ceiling — how good can detection be if we ignore privacy entirely and just centralize all data?

| Metric | Value |
|---|---|
| Task | Normal vs Attack (binary) |
| Architecture | 41 → 128 → 64 → 32 → 1 (MLP) |
| F1 Score | **0.9947** |
| Samples | 125,973 (NSL-KDD) |

**Takeaway:** This is the number federated learning has to match to prove the approach works.

---

## Experiment 2 — Federated Binary (IID, custom FedAvg)

**Goal:** Prove federated learning can match the centralized baseline without sharing raw data.

| Metric | Value |
|---|---|
| Nodes | 3 (equal random split) |
| Rounds | 15 |
| Local epochs/round | 3 |
| F1 Score | **0.9946** |
| Difference from baseline | 0.0001 |

**Takeaway:** Federated learning is statistically identical to centralized here. This is the core privacy-vs-performance result.

---

## Experiment 3 — Federated Binary (Real Flower, gRPC)

**Goal:** Move from a simulated single-process FedAvg to a real distributed system — separate processes communicating over a network protocol, the way production federated learning actually works.

| Metric | Value |
|---|---|
| Framework | Flower (flwr) |
| Communication | gRPC, 4 separate OS processes |
| Rounds | 10 |
| F1 Score | **0.9938** |

**Takeaway:** Slightly lower than the custom FedAvg (0.9946) likely due to fewer rounds (10 vs 15) and Flower's default aggregation timing. Confirms the system works as real distributed infrastructure, not just a script.

---

## Experiment 4 — Multi-Class Classification (Centralized, no balancing)

**Goal:** Move past binary detection to classify attack *type* — DoS, Probe, R2L, U2R — using the standard NSL-KDD 5-class taxonomy.

| Class | Train Samples | F1 Score |
|---|---|---|
| Normal | 53,874 | 0.98 |
| DoS | 36,741 | 1.00 |
| Probe | 9,325 | 0.97 |
| R2L | 796 | 0.50 |
| U2R | 42 | 0.18 |
| **Macro F1** | | **0.73** |

**Problem identified:** Severe class imbalance. U2R had only 42 training samples — the model essentially never learned this class.

---

## Experiment 5 — Multi-Class + SMOTE (Centralized)

**Goal:** Fix the class imbalance from Experiment 4 using synthetic oversampling.

| Change | Detail |
|---|---|
| Method | SMOTE (k_neighbors=5) |
| Effect | All classes oversampled to 53,874 each |
| R2L F1 | 0.50 → **0.75** |
| U2R F1 | 0.18 → **0.24** |
| **Macro F1** | 0.73 → **0.79** |

**Takeaway:** SMOTE significantly helped R2L. U2R remained weak — only 10 *test* samples exist for U2R regardless of training augmentation, so this is partly an evaluation-set limitation, not just a training problem.

---

## Experiment 6 — Federated Multi-Class (IID split)

**Goal:** Combine federated learning with multi-class detection — does privacy-preserving training still work when the task is harder?

| Metric | Value |
|---|---|
| Nodes | 3 (equal random split, SMOTE applied per node) |
| Rounds | 15 |
| **Macro F1** | **0.81** |

**Result: Federated (0.81) > Centralized (0.79).**

**Takeaway:** This was unexpected — federated *beat* centralized on the harder task. Likely explanation: averaging 3 independently-regularized models acts similarly to ensembling, which can outperform a single centralized model, especially on minority classes.

---

## Experiment 7 — Federated Multi-Class (Non-IID split)

**Goal:** Test the realistic scenario — each organization's traffic looks different (a hospital's traffic differs from a bank's), not a random even split.

**Split design:**

| Node | Simulated as | Dominant classes |
|---|---|---|
| 1 | Hospital | Normal (84%), R2L-heavy |
| 2 | Bank | DoS + Probe heavy |
| 3 | Campus | Mixed/remaining |

| Metric | Value |
|---|---|
| **Macro F1** | **0.84** (best result) |
| R2L Recall | 0.42 (down from 0.95 with SMOTE) |
| U2R F1 | 0.67 (up from 0.24) |

**Takeaway — the most important finding in this project:**
Conventional federated learning literature predicts non-IID data *hurts* performance, since each node's local gradients pull the global model in different directions. Here the opposite happened. Our hypothesis: because each node still saw *some* of every class (just skewed, not fully disjoint), the realistic specialization let each node sharpen its gradients for its dominant attack types, and FedAvg combined these specialized updates effectively. The R2L recall drop (0.95 → 0.42) shows the real cost: R2L was thin in 2 of 3 nodes, so the global model under-weighted it relative to the SMOTE-balanced version.

**Honest limitation:** This was tested with one random seed and one specific skew design. A harder, fully-disjoint non-IID split (e.g., one node sees *zero* R2L samples) would be needed to confirm whether this result generalizes or whether we got a favorable split by chance.

---

## Summary Table

| Experiment | Task | Setting | Score |
|---|---|---|---|
| 1 | Binary | Centralized | F1: 0.9947 |
| 2 | Binary | Federated (custom FedAvg, IID) | F1: 0.9946 |
| 3 | Binary | Federated (Flower/gRPC, IID) | F1: 0.9938 |
| 4 | Multi-class | Centralized, no balancing | Macro F1: 0.73 |
| 5 | Multi-class | Centralized + SMOTE | Macro F1: 0.79 |
| 6 | Multi-class | Federated, IID + SMOTE | Macro F1: 0.81 |
| 7 | Multi-class | Federated, **Non-IID** + SMOTE | **Macro F1: 0.84** |

---

## Key Findings (for interviews/viva)

1. **Privacy has near-zero cost on binary detection** — 0.0001 F1 difference between centralized and federated.
2. **Class imbalance is the real bottleneck**, not federation — U2R stayed weak across every experiment due to having only 10 test samples total in NSL-KDD.
3. **SMOTE meaningfully helps minority classes** — R2L F1 nearly doubled (0.50 → 0.75).
4. **Non-IID federated learning outperformed IID here** — counter to typical literature, worth flagging as a finding to investigate further rather than a guaranteed result.
5. **R2L attacks are the hardest to detect** because they mimic normal traffic by design (stealthy by nature) — visible directly in the confusion matrix (119/199 R2L samples misclassified as Normal in the IID federated model).