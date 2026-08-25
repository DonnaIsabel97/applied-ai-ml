# Applied AI & ML 

A collection of applied ai and ml projects showcasting practical implementations of ML models, data analysis and intelligent systems.

This repository brings together projects developed throughout my Master’s in Computer Science, with a focus on Artificial Intelligence and Machine Learning, emphasizing not only model implementation but also the reasoning behind model selection, evaluation, and interpretation.

## Purpose

The goals of this repository are to:

**Apply ai and ml concepts to practical problems and give in scenarios**
**Explore different ml algorithms and AI techniques**
**Develop and evaluate predictive models**
**Practice data prepocessing, exploration an feature preparation**
**Compare models using appropiate evaluation metrics**
**Understand the reasoning behing model and algorithm selection**
**Document experiments, results, limitations and lessons learned**
**Demostrate my AI/ML knowledge through practical implementations**

## Areas of Focus

Projects in this repository may explore topics including:

**Supervised Learning**
**Unsupervised Learning**
**Classification**
**Regression**
**Clustering**
**Probabilistic Models**
**Bayesian Networks**
**Natural Language Processing**
**Model Evaluation**
**Feature Engineering**
**Data Preprocessing**
**Exploratory Data Analysis**
**AI/ML Model Selection**

## Technologies

Depending on the project, technologies may include:

**Python**
**Jupyter Notebook**
**Numpy**
**Pandas**
**Matploblib**
**Scikit-learn**
**ML libraries**
**Probabilistic Modeling Tools**

## Repository Structure

Each project is organized independently and includes documentation explaining the problem, methodology, implementation, evaluation and results.

## Projects

### Project 1 - LSTM Time-Series Forecasting & Model Optimization

**Focus:** 
Time-Series Forecasting, Deep Learning, LSTM Networks, Model Optimization, Regularization, Ensemble Learning

**Technologies:**
Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy

**Description:**
Developed and optimized an LSTM-based model for multivariate time-series forecasting using environmental data. The project involved sequential data preprocessing, stacked LSTM architecture design, regularization, learning-rate optimization, and ensemble techniques such as model averaging and bagging. Model performance was evaluated and compared using RMSE and MAE.

### Project 2 - Demographic Clustering

**Focus:**
 Machine Learning, Healthcare Analytics, Demographic Analysis, Clustering, Responsible AI

**Technologies:**
Python, Pandas, Scikit-learn, K-Means, Data Preprocessing


**Description:**
Exploration of demographic and public-health data related to Alzheimer's disease using machine learning techniques. The project examines data preprocessing, demographic clustering, model-selection considerations, and ethical challenges associated with using sensitive attributes in healthcare-oriented machine learning.

### Project 3 - Helios AI Models

**Focus:**
Financial Machine Learning, Fraud Detection, Credit Risk Assessment, Model Optimization, Probabilistic AI

**Technologies:**
Python, Scikit-learn, XGBoost, Bayesian Networks, Pandas, NumPy

**Description:**
Helios AI Models contains the machine learning models developed to power the broader Helios financial intelligence platform. The project focuses on two complementary financial AI systems: **FinGuard** and **FinSage**, each addressing a different aspect of financial risk analysis.

**FinGuard** is a fraud-detection model designed to identify potentially fraudulent credit card transactions. It uses XGBoost for binary classification and includes preprocessing, model training, evaluation, hyperparameter optimization, and model persistence. Performance is analyzed using confusion matrices and ROC-AUC to evaluate the model's ability to distinguish fraudulent from legitimate transactions.

**FinSage** focuses on credit-risk assessment using a Bayesian Network. Rather than relying exclusively on point predictions, the model uses probabilistic inference to represent relationships between financial variables and estimate credit-risk outcomes. The project includes an expert-designed Bayesian Network structure, feature preparation, discretization, probabilistic inference, evaluation, and model optimization.

Together, FinGuard and FinSage form the initial AI layer of **Helios**, combining transaction-level fraud detection with probabilistic credit-risk analysis. These models are being developed separately from the application layer so they can be trained, evaluated, optimized, and validated independently before integration.

The broader **Helios web application is currently under development**, with a frontend being built to provide an interface through which the models and their financial insights can eventually be accessed.

The `helios-ai-models` directory contains the model implementations and documentation, while the separate Helios repository contains the ongoing development of the complete application.


## Project Methodology

While each project has different requirements, I generally document the following stages:

### 1. Problem Definition

Define the problem being addressed and the objective of the AI/ML implementation.

### 2. Data Understanding

Explore the dataset, its features, structure, quality, and potential limitations.

### 3. Data Preparation

Clean and transform the data as required for the selected model or technique.

### 4. Model Selection

Select an appropriate algorithm or AI technique based on the characteristics and requirements of the problem.

### 5. Implementation

Develop and train the model or intelligent system.

### 6. Evaluation

Evaluate performance using metrics appropriate to the problem and analyze the model's behavior.

### 7. Results & Interpretation

Interpret the results rather than relying only on numerical performance.

### 8. Limitations & Improvements

Identify limitations of the implementation and potential approaches for future improvement.

## Evaluation

Depending on the type of problem, projects may use evaluation techniques and metrics such as:

**Accuracy**
**Precision**
**Recall**
**F1 Score**
**Confusion Matrix**
**ROC-AUC**
**Mean Absolute Error (MAE)**
**Mean Squared Error (MSE)**
**Cross-Validation**

The selected metrics depend on the objective and characteristics of each individual project.

## Documentation

Each project includes its own `README.md`, providing an overview of the project's objectives, methodology, technologies, and key results.

More comprehensive technical documentation is also available in `PDF` format, offering an in-depth explanation of the project's development process, implementation, analysis, and findings.

## Continuos Development

This repository serves both as a record of my applied AI/ML work and as a portfolio of my development in ml.

Projects may be revisited over time to improve documentation, compare additional apporaches, optimize implementations or apply techniques learned through continued AI/ML study.
