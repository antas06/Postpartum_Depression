# 🧠 Postpartum Depression Prediction  
Machine Learning & Deep Learning Frameworks for Classification

This repository presents a structured comparison between traditional machine learning models and advanced deep learning architectures for postpartum depression detection using tabular clinical and socio-demographic data.

--------------------------------------------------------------------

PROJECT OVERVIEW

The study follows a three-tier modeling strategy:

1. Baseline Machine Learning Model – Random Forest Classifier  
2. Baseline Deep Learning Model – Standard Artificial Neural Network (ANN)  
3. Proposed Model – High-Capacity TabTransformer with Multi-Head Attention  

The objective is to evaluate how attention-based tabular transformers outperform conventional models in capturing complex feature interactions.

--------------------------------------------------------------------

DATASET DESCRIPTION

- Records: 30,000 synthetic patient samples  
- Data Generation: CTGAN augmentation  
- Feature Types:
  - Categorical: Socio-demographic attributes  
  - Numerical: Clinical and obstetric history  
- Preprocessing Pipeline:
  - ABC Optimization for anomaly correction (Gravida feature)
  - Label Encoding for categorical variables
  - Standard Scaling for numerical variables

--------------------------------------------------------------------

BASELINE MODEL 1: RANDOM FOREST CLASSIFIER

Model Rationale:  
Random Forest serves as a strong non-parametric baseline capable of handling non-linear relationships without requiring feature scaling.

Architectural Flow (Pseudocode):

INPUT: Unscaled tabular dataset (X_train, y_train)

1. Bootstrapping  
   Generate 100 random subsets of the training data with replacement.

2. Tree Construction  
   For each estimator:
   - Train a decision tree
   - Random feature selection at each split
   - Optimize splits using Gini Impurity
   - Grow tree to maximum depth

OUTPUT:  
Predictions from all trees are aggregated using majority voting to produce the final binary classification (Depressed / Not Depressed).

Hyperparameters:
- n_estimators: 100  
- n_jobs: -1  
- random_state: 42  
- Feature Scaling: Not required  

--------------------------------------------------------------------

BASELINE MODEL 2: ARTIFICIAL NEURAL NETWORK (ANN)

Model Rationale:  
The ANN acts as a deep learning baseline and requires normalized inputs to ensure stable gradient descent.

Architectural Flow (Pseudocode):

INPUT: MinMax scaled features (range 0–1)

Hidden Layer 1:
- Dense layer with 64 neurons
- ReLU activation
- Dropout rate: 0.3

Hidden Layer 2:
- Dense layer with 32 neurons
- ReLU activation
- Dropout rate: 0.2

Output Layer:
- Dense layer with 1 neuron
- Sigmoid activation

OUTPUT:  
A probability value P ∈ [0,1] is produced. Final classification is determined using a threshold of P > 0.5.

Hyperparameters:
- Scaler: MinMaxScaler  
- Hidden Layers: [64, 32]  
- Activation: ReLU  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Batch Size: 32  
- Epochs: 100  
- Validation Split: 0.20  
- Early Stopping: Patience = 10  

--------------------------------------------------------------------

PROPOSED MODEL: HIGH-CAPACITY TABTRANSFORMER

Model Motivation:  
The TabTransformer introduces self-attention for categorical variables, enabling the model to learn inter-feature dependencies that traditional models fail to capture, especially in complex clinical datasets.

Architectural Flow (Pseudocode):

INPUT:
- Categorical Features: Label Encoded
- Numerical Features: Standard Scaled

1. Categorical Embedding  
   Each categorical feature is mapped to a dense vector of dimension 32.

2. Multi-Head Attention Block  
   - MultiHeadAttention with 8 heads and key dimension 32  
   - Residual connection with original embeddings  
   - Layer normalization  
   - Flattening of attention output

3. Feature Fusion  
   Concatenate attention-processed categorical representation with numerical features.

4. MLP Classification Head  
   - Dense(128) → ReLU → Dropout(0.2)  
   - Dense(64) → ReLU → Dropout(0.2)  
   - Dense(32) → ReLU  

OUTPUT:
- Dense(1) with Sigmoid activation  
- Binary probability indicating Depressed or Not Depressed

Hyperparameters:
- Embedding Dimension: 32  
- Attention Heads: 8  
- Key Dimension: 32  
- MLP Layers: [128, 64, 32]  
- Dropout Rate: 0.2  
- Optimizer: Adam  
- Initial Learning Rate: 0.001  
- Loss Function: Binary Crossentropy  
- Batch Size: 32  
- Epochs: 100  
- Validation Split: 0.20  
- Early Stopping: Patience = 20  
- Learning Rate Scheduler: Factor = 0.2, Patience = 5  

--------------------------------------------------------------------

KEY CONTRIBUTIONS

- Comparative evaluation of ML, DL, and Transformer-based tabular models  
- Use of CTGAN-generated synthetic healthcare data  
- Attention-based modeling of categorical–numerical interactions  
- Robust regularization using Dropout, Early Stopping, and LR Scheduling  

--------------------------------------------------------------------

CONCLUSION

The TabTransformer demonstrates superior capability in learning complex dependencies within tabular healthcare data, making it a strong candidate for clinical decision support systems aimed at postpartum depression screening.

--------------------------------------------------------------------

LICENSE

This project is intended strictly for academic and research use.