# chatbot-response-scoring-scbn-rqtl

This repository contains several Jupyter notebooks that classify chatbot prompts and predict human preference on responses using SCBN (Specificity, Coherency, Brevity, Novelty) and RQTL (Request vs Question, Test vs Learn) metrics, a benchmark I created to evaluate chatbot responses based on prompts.

- **SCBN**: A framework that scores chatbot responses by measuring Specificity, Coherency, Brevity, and Novelty.
- **RQTL**: A classification system for categorizing user prompts into four quadrants: Request vs Question, Test vs Learn.

The core foundational ideas of this repository are inspired by the SCBN benchmark first introduced at the [Talking to Chatbots](https://talkingtochatbots.com/) website and a submission to the [LMSYS – Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena) competition on Kaggle.

## Files

- `lmsys-cba-reddgr-scbn-rqtl-codespaces.ipynb`: A Jupyter notebook that classifies chatbot prompts and predicts human preference on responses using SCBN and RQTL metrics. The notebook covers data preprocessing, classification, and model training and evaluation.
- `lmsys-cba-reddgr-scbn-rqtl-kaggle.ipynb`: older version of the notebook that can be run directly on Kaggle
- `zero-shot-and-few-shot-text-classification-examples.ipynb`: text classification process and examples used in the main notebook, using Tensorflow as main framework.
- `zero-shot-and-few-shot-text-classification-examples-torch.ipynb`: text classification process and examples used in the main notebook, using PyTorch as main framework.
- `chat-with-gemma-notebook.ipynb`: A Jupyter notebook that sets up a chat interface with the Gemma model, enabling interaction by sending prompts and receiving responses directly within the notebook. It simplifies testing and experimentation with the model, eliminating the need for external applications or interfaces. Gemma was not used in the original SCBN-RQTL scoring notebook, but this asset is included here as the code may be useful for performing further analysis and improvements to the SCBN-RQTL benchmark.
- `datasets.ipynb`: This notebook downloads prompts and responses from the official LMSYS Chatbot Arena repository hosted on HuggingFace. It requires a HuggingFace token to access the lmsys-chat-1m dataset, retrieves and caches the data locally, and provides tools for exploring specific conversations and displaying samples directly within the notebook.
- `install_dependencies.sh`: A shell script that installs the necessary dependencies to run the Jupyter notebook.
- `requirements.txt`: A file containing the Python dependencies for the Jupyter notebooks.

- **Additional Context**:
  - Original notebook published on [Kaggle](https://www.kaggle.com/code/davidgromero/lmsys-cba-reddgr-scbn-rqtl-v1).
  - [Reddgr models and datasets on HuggingFace](https://huggingface.co/reddgr)
- **Further Reading**:
  - [Introduction to the SCBN in TTCB blog](https://talkingtochatbots.com/predicting-chatbot-arena-votes-with-the-scbn-and-rqtl-benchmarks/)
  - [Introduction to SCBN Chatbot battles in TTCB blog](https://talkingtochatbots.com/talking-to-chatbots/is-philosophy-a-science-chatbot-battle/)
  - [SCBN search term in TTCB blog](https://talkingtochatbots.com/?s=SCBN)
 
## Installation

To install all necessary dependencies, make the script executable and run:

chmod +x install_dependencies.sh
install_dependencies.sh

# Notebook Overview

The notebook lmsys-cba-reddgr-scbn-rqtl-kaggle.ipynb was originally designed to process, analyze, and fine-tune models based on user prompt data from the Chatbot Arena as part of the LMSYS Chatbot Arena competition on Kaggle. The competition's goal was to predict which chatbot responses users will prefer in head-to-head battles between chatbots powered by large language models (LLMs). The notebook follows a series of steps to prepare data, fine-tune models, and make predictions to address the competition's objectives.

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
- Relative scores analysis.

## 8. Token Length Features (Brevity Score)
- Tokenization examples and exploration of the dataset.
- Calculation of Brevity scores and their visualization through histograms and hexbin plots.
- Compilation of SCBN (Specificity, Coherency, Brevity, Novelty) scores.

## 9. PCA and SCBN Scores Evaluation
- Principal Component Analysis (PCA) on SCBN scores and their evaluation.

## 10. Linear Decision Tree Model
- First approximation and feature calibration using a linear decision tree model.

## 11. Logistic Regression
- Implementation of a simplified logistic regression model for predicting response votes, using only SCBN-RQTL-related metrics.

## 12. Neural Network
- Implementation of a neural network model for predicting response votes, using only SCBN-RQTL-related metrics.

## 13. Kaggle Submission
- Preparation and submission of the final model and results to Kaggle.

I made this README with Python Code Streamliner, a GPT which you can access [here](https://chatgpt.com/g/g-M4uZyYsUj-python-code-streamliner).
