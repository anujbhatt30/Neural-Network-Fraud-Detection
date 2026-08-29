# Fraud Detection Model

A machine learning project to detect fraudulent transactions using [FILL IN: dataset name, e.g. "the Kaggle Credit Card Fraud Detection dataset"]. Three models were built and compared to handle the severe class imbalance typical of fraud detection problems.

---

## 📊 Problem Statement

Fraudulent transactions are rare compared to legitimate ones, making this a **highly imbalanced classification problem**. The goal is to maximize fraud detection (recall) while keeping false positives manageable enough for practical use — a trade-off explored across three model variants.

---

## 📁 Dataset

- **Source:** [FILL IN: e.g. Kaggle - Credit Card Fraud Detection]
- **Size:** [FILL IN: number of rows/transactions]
- **Features:** [FILL IN: e.g. anonymized PCA features V1-V28, Amount, Time]
- **Class distribution:** [FILL IN: e.g. 492 frauds out of 284,807 transactions (~0.17%)]

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- [FILL IN: any resampling library used, e.g. imbalanced-learn (SMOTE)]

---

## 🏗️ Approach

Given the class imbalance, three modeling approaches were compared:

| Model | Approach | Key Difference |
|-------|----------|-----------------|
| Model 1 | Baseline neural network | [FILL IN — e.g. class weights for imbalance, default threshold 0.5] |
| Model 2 | Tuned neural network | [FILL IN — e.g. adjusted architecture/layers] + threshold tuned to 0.72 (best F1/recall balance) |
| Model 3 | Alternate approach | [FILL IN — e.g. different resampling method, different architecture] |

*(Edit the bracketed cells above with your actual architecture/technique differences — the metrics and confusion matrices are accurate as given.)*

All models were evaluated on **Precision, Recall, F1 Score, and ROC-AUC** rather than raw accuracy alone, since accuracy is misleading on imbalanced data — a model predicting "no fraud" for everything would still score ~95%+ accuracy while catching zero fraud.

---

## Features

| Feature                        | Description                                     |
| ------------------------------ | ----------------------------------------------- |
| `transaction_id`               | Unique transaction identifier                   |
| `amount_usd`                   | Transaction amount in USD                       |
| `merchant_category`            | Category of merchant                            |
| `card_type`                    | Type of card used                               |
| `auth_method`                  | Authentication method                           |
| `channel`                      | Transaction channel                             |
| `device_type`                  | Device used for transaction                     |
| `is_foreign_transaction`       | Whether the transaction is foreign              |
| `hours_since_last_txn`         | Hours since the previous transaction            |
| `txn_count_last_24h`           | Number of transactions in the previous 24 hours |
| `distance_from_home_km`        | Distance of transaction from home               |
| `card_age_months`              | Age of the card                                 |
| `customer_age`                 | Age of customer                                 |
| `account_balance_usd`          | Account balance                                 |
| `is_new_merchant`              | Whether the merchant is new                     |
| `used_vpn`                     | Whether a VPN was used                          |
| `ip_country_mismatch`          | IP country mismatch indicator                   |
| `billing_shipping_mismatch`    | Billing/shipping mismatch indicator             |
| `cvv_retry_count`              | Number of CVV retry attempts                    |
| `velocity_score`               | Transaction velocity score                      |
| `time_of_day_hour`             | Hour of transaction                             |
| `day_of_week`                  | Day of transaction                              |
| `is_ai_generated_scam_attempt` | AI-generated scam indicator                     |
| `merchant_risk_score`          | Merchant risk score                             |
| `prior_disputes`               | Number of previous disputes                     |
| `is_fraud`                     | Target variable                                 |

---

# ⚖️ Class Imbalance

One of the most important characteristics of the dataset is the strong imbalance between legitimate and fraudulent transactions.

The target variable is:

```text
is_fraud
```

where:

```text
0 = Non-Fraud
1 = Fraud
```

Because fraudulent transactions are the minority class, the project explicitly analyzes the class distribution before model training.

---

### Class Distribution

![Class Distribution](Images/class_distribution.png)

---

# 🧹 Data Preprocessing

The dataset contains both numerical and categorical variables.

A preprocessing pipeline was created using Scikit-learn's `ColumnTransformer`.

---

## Numerical Features

Numerical features are scaled using:

```python
StandardScaler()
```

Scaling helps the neural network train more effectively by placing numerical variables on comparable scales.

---

## Categorical Features

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical variables into numerical representations that can be passed to the neural network.

---

## ColumnTransformer

The numerical and categorical preprocessing steps are combined using:

```python
ColumnTransformer
```

Conceptually:

```text
                    Dataset
                       │
             ┌─────────┴─────────┐
             │                   │
       Numerical             Categorical
        Features               Features
             │                   │
      StandardScaler      OneHotEncoder
             │                   │
             └─────────┬─────────┘
                       │
                Processed Data
                       │
                 Neural Network
```

After preprocessing, the dataset contains:

```text
54 processed features
```

---

# 🧠 Neural Network

A feed-forward neural network was developed for binary classification.

The general architecture follows:

```text
Input Features
      │
      ▼
Dense Layer
      │
      ▼
Activation
      │
      ▼
Regularization
      │
      ▼
Dense Layer
      │
      ▼
Activation
      │
      ▼
Output Layer
      │
      ▼
Sigmoid
      │
      ▼
Fraud Probability
```

The output layer produces a probability between:

```text
0 and 1
```

This probability is later converted into a fraud/non-fraud prediction using a classification threshold.

---

# ⚖️ Class Weights

Because the dataset is imbalanced, class weights are used during neural network training.

The purpose is to give greater importance to the minority fraud class.

Conceptually:

```text
Non-Fraud → Lower Weight
Fraud     → Higher Weight
```

This encourages the model to pay more attention to fraudulent transactions during training.

---

# 🏋️ Model Training

The neural networks were trained using:

* Training data
* Validation data
* Class weights
* Binary classification loss
* Early stopping

## Early Stopping

Early stopping was implemented to reduce overfitting.

The model monitors validation performance during training and stops training when further training no longer produces meaningful improvement.

This helps prevent the model from unnecessarily learning noise from the training dataset.

---

# 🧪 Multiple Model Architectures

Instead of relying on a single neural network architecture, three models were experimented with.

```text
Model 1
   │
   ├── Train
   └── Evaluate

Model 2
   │
   ├── Train
   └── Evaluate

Model 3
   │
   ├── Train
   └── Evaluate
```

The purpose of creating multiple models was to determine whether changes in the network architecture could improve fraud classification performance.


---

## 📈 Results

| Metric | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| Accuracy | 93.55% | 95.33% | 92.48% |
| Precision | 17.01% | 21.80% | 15.22% |
| Recall | 72.06% | 67.65% | 75.00% |
| F1 Score | 27.53% | 32.97% | 25.31% |
| ROC-AUC | 0.9290 | 0.9358 | 0.9304 |

### Confusion Matrices

**Model 1:**
```
[[3589  343]
 [  14   54]]
```

**Model 2 — Threshold 0.70:**
```
[[3767  165]
 [  22   46]]
```

**Model 2 — Threshold 0.72 (final):**
```
[[3723  209]
 [  19   49]]
```
> Raising the threshold from 0.70 → 0.72 traded a few missed frauds caught (46 → 49 TP) for more false positives (165 → 209 FP) — the 0.72 threshold was chosen as the better F1/recall balance.

**Model 3:**
```
[[3648  284]
 [  17   51]]
```

---

### Confusion Matrices by Model

| Model 1 | Model 2 | Model 3 |
| :---: | :---: | :---: |
| ![Confusion Matrix Model 1](Images/confusion_matrix_model_1.png) | ![Confusion Matrix Model 2](Images/confusion_matrix_model_2.png) | ![Confusion Matrix Model 3](Images/confusion_matrix_model_3.png) |

---

### ROC-AUC Comparison

![Model Comparison](Images/model_comparison.png)

---

### ROC Curve

![ROC Curve](Images/roc_curve.png)

---

### Final Model Comparison

![Final Model Comparison](Images/final_model_comparison.png)



---

### 🏆 Best Model: Model 2

Model 2 achieved the **best balance** across metrics — highest F1 Score (32.97%) and highest ROC-AUC (0.9358), meaning it separates fraud from non-fraud most reliably while keeping false positives comparatively lower than Model 3.

However, the right model depends on business priority:
- **If minimizing missed fraud is the priority** (recall-first), **Model 3** catches the most fraud cases (75% recall) at the cost of more false alarms.
- **If overall reliability and balance matter most**, **Model 2** is the stronger choice.

---

## 🔍 Key Insight: Why Not Just Use Accuracy?

All three models exceed 92% accuracy — but this is misleading. With such a rare positive class, a naive model predicting "not fraud" every time would still score high accuracy while catching **zero** actual fraud. Precision, Recall, F1, and ROC-AUC give a truer picture of real-world usefulness here.

---

## 🚀 How to Run

```bash
# Clone the repository
git clone [FILL IN: repo URL]
cd [FILL IN: repo name]

# Install dependencies
pip install -r requirements.txt

# Run the notebook / script
[FILL IN: e.g. jupyter notebook fraud_detection.ipynb]
```

---


# 🔬 Current Conclusion

Three neural network models were developed and evaluated.

The current ROC-AUC comparison is:

```text
Model 1 → 0.9290                                                       
Model 2 → 0.9358  ← Current Best
Model 3 → 0.9304
```

Model 2 currently provides the highest ROC-AUC and is therefore the **leading candidate**.

The final decision between Model 1 and Model 2 will be made after comparing their:

* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Threshold

The model is **not deployed yet**. Deployment and productionization are planned as future improvements.

---

## 📂 Repository Structure

```
├── data/                  # dataset (or link if too large for GitHub)
├── notebooks/             # exploration and model training notebooks
├── models/                # saved model files
├── README.md
└── requirements.txt
```
---

## ⭐ Project Highlights

```text
Real-World Classification Problem
            ↓
Highly Imbalanced Dataset
            ↓
Advanced Preprocessing
            ↓
54 Processed Features
            ↓
Neural Network
            ↓
Class Weighting
            ↓
3 Model Experiments
            ↓
ROC-AUC Comparison
            ↓
Threshold Optimization
            ↓
Final Model Selection
```

> **This project demonstrates the complete model-development process rather than simply training a neural network: preprocessing, imbalance handling, experimentation, evaluation, comparison, and threshold-based decision making.**

---

### Areas of Interest

* Data Science
* Machine Learning
* Deep Learning
* Neural Networks
* SQL
* Data Engineering
* Business Analytics

---

## 🔮 Future Improvements

- [ ] Experiment with SMOTE / other resampling techniques if not already used
- [ ] Try ensemble methods combining strengths of multiple models
- [ ] Deploy as an API endpoint for real-time fraud scoring
- [ ] Add SHAP-based explainability for model predictions

## 👤 Author

**Anuj Bhatt** 

---
*This project was built as part of exploring machine learning approaches to imbalanced classification problems in the fraud detection domain.*

