# Helios AI Models

## Overview

**Helios AI Models** contains the artificial intelligence and machine learning models developed to power the intelligent financial analysis capabilities of **Helios**, a broader full-stack financial technology application currently under development.

This repository focuses specifically on the **AI/ML layer** of the project, keeping model development, experimentation, optimization, and evaluation separate from the web application itself.

The repository currently contains two complementary financial AI models:

- **FinGuard** — a supervised machine learning model for financial fraud detection.
- **FinSage** — a probabilistic AI model for credit risk assessment.

Together, these models explore different approaches to financial risk analysis, combining high-performance supervised machine learning with interpretable probabilistic modeling.

---

## Models

### FinGuard — Fraud Detection

**FinGuard** is a supervised machine learning model designed to identify potentially fraudulent financial transactions.

The model uses **XGBoost** for binary classification and is designed around the challenges presented by highly imbalanced fraud datasets, where fraudulent transactions represent only a small percentage of the available observations.

The model prioritizes the detection of fraudulent activity while maintaining reasonable precision to limit unnecessary false alerts.

Key areas explored include:

- Financial fraud detection
- Binary classification
- XGBoost
- Imbalanced datasets
- SMOTE
- Feature preprocessing and scaling
- Hyperparameter optimization
- GridSearchCV
- Precision, recall, and F1 evaluation
- ROC-AUC analysis
- Confusion matrix analysis

The model was optimized using class-balancing techniques and hyperparameter tuning to improve fraud detection while maintaining strong overall classification performance.

Detailed implementation, preprocessing, optimization, evaluation, results, and limitations are documented within the FinGuard directory.

➡️ **[View FinGuard](./finguard/README.md)**

---

### FinSage — Credit Risk Assessment

**FinSage** is a probabilistic AI model designed to evaluate credit risk by modeling relationships between financial attributes and potential credit outcomes.

Unlike conventional classification models, FinSage uses a **Discrete Bayesian Network** to represent dependencies between financial variables and estimate risk through probabilistic inference.

An expert-designed network structure is used to define meaningful relationships between financial attributes. This approach was selected because the dataset used during development is relatively small, making manually defined relationships more appropriate for the scope of the model than relying entirely on automatic structure learning.

Key areas explored include:

- Credit risk assessment
- Bayesian Networks
- Probabilistic graphical models
- Probabilistic inference
- Conditional dependencies
- Feature discretization
- Expert-designed network structures
- Maximum Likelihood Estimation
- Variable elimination
- Model optimization
- Precision, recall, and F1 evaluation
- Confusion matrix analysis
- Interpretable AI

The model was optimized by refining the feature set and network structure to better represent relationships between applicant financial characteristics while preserving the interpretability of the probabilistic framework.

Detailed architecture, preprocessing, probabilistic inference, optimization, evaluation, results, and limitations are documented within the FinSage directory.

➡️ **[View FinSage](./finsage/README.md)**

---

## AI Architecture

The repository intentionally explores two different approaches to financial risk analysis.

| Model        | Financial Problem      | AI Approach                 | Primary Model             |
| ------------ | ---------------------- | --------------------------- | ------------------------- |
| **FinGuard** | Fraud Detection        | Supervised Machine Learning | XGBoost                   |
| **FinSage**  | Credit Risk Assessment | Probabilistic AI            | Discrete Bayesian Network |

### FinGuard

FinGuard learns classification patterns from historical transaction data to distinguish potentially fraudulent activity from legitimate transactions.

Its development emphasizes predictive performance, class imbalance management, fraud recall, and classification optimization.

### FinSage

FinSage represents dependencies between financial variables through a probabilistic graphical structure.

Its development emphasizes uncertainty, conditional relationships, probabilistic reasoning, and interpretability within credit-risk assessment.

Together, the models allow Helios to explore multiple forms of financial intelligence rather than relying on a single modeling methodology.

---

## Model Development

Although FinGuard and FinSage are intended to contribute to the same broader platform, each model is developed as an independent machine learning system.

The development process includes:

**Data → Preprocessing → Model Development → Training → Evaluation → Optimization → Integration**

Each model maintains its own:

- Dataset and preprocessing pipeline
- Feature preparation strategy
- Model architecture
- Training process
- Optimization approach
- Evaluation methodology
- Performance results
- Model-specific documentation

This separation allows each AI component to be tested and improved independently before integration with the broader Helios application.

---

## Model Evaluation

The two models require different evaluation considerations because they address different financial problems and use different AI methodologies.

### FinGuard

Fraud detection involves a highly imbalanced classification problem. Because fraudulent transactions represent only a small portion of the available data, overall accuracy alone is not sufficient for evaluating model performance.

FinGuard therefore considers metrics including:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Particular attention is given to **recall for fraudulent transactions**, since failing to identify actual fraud represents an important risk within a financial system.

At the same time, precision must remain sufficiently strong to prevent the system from producing an excessive number of false fraud alerts.

### FinSage

FinSage evaluates classification performance while also considering the probabilistic and interpretable nature of the Bayesian Network.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Because the model is designed for credit-risk assessment, the ability to correctly identify high-risk cases is an important component of evaluation.

Model results must also be interpreted alongside the limitations of the available dataset and validation methodology.

---

## Model Optimization

Both models were optimized according to the characteristics of their respective machine learning approaches.

### FinGuard Optimization

FinGuard optimization focused on improving fraud detection while maintaining strong overall classification performance.

The optimization process included:

- SMOTE for minority-class representation
- Hyperparameter tuning
- GridSearchCV
- Learning-rate optimization
- Tree-depth optimization
- Estimator optimization
- Sampling-ratio adjustments

The objective was to improve the model's ability to identify fraudulent transactions while reducing unnecessary false-positive classifications.

### FinSage Optimization

FinSage optimization focused on improving the identification of high-risk credit applicants while preserving the interpretability of the Bayesian framework.

The optimization process included:

- Refinement of the feature set
- Additional financial variables
- Improved feature discretization
- Refinement of the Bayesian Network structure
- Additional dependencies between relevant variables

These changes were designed to create a more representative probabilistic structure capable of capturing relationships between financial characteristics and credit risk.

---

## Relationship to Helios

FinGuard and FinSage form part of the **AI/ML layer of the broader Helios project**.

**Helios** is an ongoing full-stack financial technology project designed to integrate intelligent financial analysis into a web-based application.

The models contained in this repository are developed, tested, evaluated, and optimized independently before being integrated into the broader Helios application architecture.

Within the Helios ecosystem:

**FinGuard** provides fraud-detection capabilities by analyzing transaction information for patterns associated with potentially fraudulent activity.

**FinSage** provides credit-risk analysis through probabilistic reasoning over financial and behavioral attributes.

Together, they are designed to provide complementary AI capabilities that power financial risk-analysis functionality within Helios.

### Current Development

The **Helios frontend is currently under active development**.

The goal of the frontend is to provide a user-facing web interface through which the financial intelligence capabilities of FinGuard and FinSage can eventually be accessed as part of the complete Helios platform.

Application development and model development are maintained separately so that the AI components can be independently tested, evaluated, optimized, and versioned without coupling model experimentation directly to frontend development.

This modular approach also provides a clearer path for connecting trained models to backend services and exposing their capabilities to the Helios frontend.

### Helios Repository

The complete Helios application, including frontend development, application architecture, backend functionality, and AI model integration, is maintained in a separate repository.

➡️ **[View the Helios Repository](https://github.com/DonnaIsabel97/helios_ai)**

---

## Deployment Direction

The models are designed with future integration into the Helios application in mind.

### FinGuard

FinGuard can support a transaction-analysis workflow in which financial transactions are submitted to a backend service and evaluated by the trained fraud-detection model.

Potentially suspicious transactions can then be identified for additional review.

### FinSage

FinSage can support a credit-evaluation workflow in which financial information is provided as evidence to the Bayesian Network.

Probabilistic inference can then be used to estimate credit-risk classifications while retaining information about the relationships between the variables involved in the decision.

The models can ultimately be exposed through backend services that connect the AI layer with the Helios web application.

---

## Technologies

### Programming & Data

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- imbalanced-learn
- SMOTE
- GridSearchCV

### Probabilistic AI

- pgmpy
- Discrete Bayesian Networks
- Maximum Likelihood Estimation
- Variable Elimination

### Evaluation & Visualization

- Matplotlib
- ROC-AUC
- Confusion Matrices
- Precision
- Recall
- F1 Score

### Model Management

- Joblib
- Modular Python architecture
- Version control

---

## Repository Structure

```text
helios-ai-models/
│
├── finguard/
│   ├── data/
│   ├── outputs/
│   ├── src/
│   └── README.md
│
├── finsage/
│   ├── data/
│   ├── outputs/
│   ├── src/
│   └── README.md
│
└── README.md