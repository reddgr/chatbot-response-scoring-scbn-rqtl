# chatbot-response-scoring-scbn-rqtl

This repository contains a Jupyter notebook that builds a dataset scoring chatbot responses from LMSYS Chatbot Arena using SCBN (Specificity, Coherency, Brevity, Novelty) and RQTL (Request vs Question, Test vs Learn) metrics, a benchmark I created to evaluate chatbot responses

- **SCBN**: A framework that scores chatbot responses by measuring Specificity, Coherency, Brevity, and Novelty.
- **RQTL**: A classification system for categorizing user prompts into four types: Request vs Question, Test vs Learn.

- **Context**: This work is part of the [LMSYS – Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena) competition on Kaggle.
- **Notebook**: Original notebook published on [Kaggle](https://www.kaggle.com/code/davidgromero/lmsys-cba-reddgr-scbn-rqtl-v1).
- **Further Reading**:
  - [Introduction to the SCBN in TTCB blog](https://talkingtochatbots.com/predicting-chatbot-arena-votes-with-the-scbn-and-rqtl-benchmarks/)
  - [Introduction to SCBN Chatbot battles in TTCB blog](https://talkingtochatbots.com/talking-to-chatbots/is-philosophy-a-science-chatbot-battle/)
  - [SCBN search term in TTCB blog](https://talkingtochatbots.com/?s=SCBN)
 

# Notebook Overview

This notebook is designed to process, analyze, and fine-tune models based on user prompt data from the Chatbot Arena as part of the LMSYS Chatbot Arena competition on Kaggle. The competition's goal is to predict which chatbot responses users will prefer in head-to-head battles between chatbots powered by large language models (LLMs). The notebook follows a series of steps to prepare data, fine-tune models, and make predictions to address the competition's objectives.

## 0. Input Data and Libraries Import
- Set up of the Kaggle Notebook environment.
- Importing necessary libraries.
- Initial setup of the notebook environment.

## 1. Train and Test Data - Initial Loading, Preparation, and Exploration
- Loading the original train and test data from the LMSYS starter notebook.
- Pre-loading dataframes for both train and test with calculated metrics.
- Exploratory data analysis on the data formats.

## 2. Data Preprocessing (Starter) - Make Pairs and Detect Encoding Errors
- Implementation of the `Make_pairs` function from the starter notebook.
- Identification and exploration of records with UTF-8 encoding issues.
- Exploration of the 'options' feature in the prompt data.

## 3. RQ Prompt Classification (Request vs Question)
- Draft classification tests using Zero-shot distilbert.
- Fine-tuning of the Distilbert model for classifying requests vs. questions.
- Manual labeling and training of the RQ classification model.
- Loading and testing the fine-tuned RQ model.
- Binary text classification for RQ prompts.
- Complete dataset classification and metric calculation for RQ prompts.

## 4. TL Prompt Classification (Test vs Learn)
- Notes and zero-shot tests for TL classification.
- Fine-tuning of the Distilbert model for TL classification.
- Manual labeling and training of the TL classification model.
- Loading and testing the fine-tuned TL model.
- Complete dataset classification and metric calculation for TL prompts.

## 5. RQTL Samples and Statistics
- Generation of random samples and histograms for RQTL prompts.
- Analysis of tie frequencies by prompt class and a 2-D histogram of tie frequencies.

## 6. TF-IDF Features (Novelty Score)
- Definition of TF-IDF features and their theoretical basis.
- Calculation of TF-IDF scores for each prompt and the corresponding pair corpus.
- Calculation and visualization of Novelty scores, including scatter plots, histograms, and hexbin plots.

## 7. SC Features (Specificity Score, Coherency Score)
- Definition and calculation of Specificity and Coherency scores.
- Visualization of Specificity and Coherency scores through histograms and hexbin plots.
- Relative score calculation.

## 8. Token Length Features (Brevity Score)
- Tokenization examples and exploration of disparities in token counts.
- Calculation of Brevity scores and their visualization through histograms and hexbin plots.
- Compilation of SCBN (Specificity, Coherency, Brevity, Novelty) scores.

## 9. PCA and SCBN Scores Evaluation
- Principal Component Analysis (PCA) on SCBN scores and their evaluation.

## 10. Linear Decision Tree Model
- First approximation and feature calibration using a linear decision tree model.

## 11. Logistic Regression
- Implementation of a logistic regression model for classification tasks.

## 12. Neural Network
- Implementation of a neural network model for further classification and analysis.

## 13. Kaggle Submission
- Preparation and submission of the final model and results to Kaggle.

I made this README with Python Code Streamliner, a GPT which you can access [here](https://chatgpt.com/g/g-M4uZyYsUj-python-code-streamliner).
