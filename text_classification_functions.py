from transformers import pipeline
from tqdm import tqdm
import torch
import numpy as np
from langdetect import detect
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

class Classifier:

    def __init__(self, model_path, label_map, verbose = False):
        self.model_path = model_path
        self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)
        self.label_map = label_map
        if verbose: 
            self.print_device_information()

    def print_device_information(self):
        # Check device information
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_properties = torch.cuda.get_device_properties(0) if device.type == "cuda" else "CPU Device"

        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"Device Name: {device_properties.name}")
            # print(f"Compute Capability: {device_properties.major}.{device_properties.minor}")
            print(f"Total Memory: {device_properties.total_memory / 1e9:.2f} GB")

    def tokenize_and_trim(self, text):
        max_length = self.classifier.tokenizer.model_max_length
        inputs = self.classifier.tokenizer(text, truncation=True, max_length=max_length, return_tensors="tf")
        return self.classifier.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)


    def classify_dataframe_column(self, df, target_column, feature_suffix):

        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(self.tokenize_and_trim)

        results = []
        for text in tqdm(df[f'trimmed_{target_column}'].tolist(), desc="Classifying"):
            result = self.classifier(text)
            results.append(result[0])

        df[f'pred_label_{feature_suffix}'] = [self.label_map[int(result['label'].split('_')[-1])] for result in results]
        df[f'prob_{feature_suffix}'] = [result['score'] for result in results]
        df.drop(columns=[f'trimmed_{target_column}'], inplace=True)
        return df
    
    def test_model_predictions(self, df, target_column):
        """
        Tests model predictions on a given dataframe column and computes evaluation metrics.

        Args:
            df (pd.DataFrame): Input dataframe containing the data.
            target_column (str): The name of the column to classify.
            feature_suffix (str): Suffix to append to the generated prediction columns.

        Requirements:
            - The dataframe must include a 'label' column for comparison with predictions.

        Returns:
            dict: A dictionary containing accuracy, F1 score, cross-entropy loss, 
                  and the confusion matrix.
        """
        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(self.tokenize_and_trim)

        results = []
        for text in tqdm(df[f'trimmed_{target_column}'].tolist(), desc="Classifying"):
            result = self.classifier(text)
            results.append(result[0])

        # Reconstruct probabilities for binary classification
        predicted_probs = []
        for result in results:
            if result['label'] == 'LABEL_0':
                predicted_probs.append([result['score'], 1 - result['score']])
            else:
                predicted_probs.append([1 - result['score'], result['score']])

        predicted_probs = np.array(predicted_probs)
        true_labels = df['label']

        # Calculate metrics
        accuracy = accuracy_score(true_labels, np.argmax(predicted_probs, axis=1))
        f1 = f1_score(true_labels, np.argmax(predicted_probs, axis=1), average='weighted')
        cross_entropy_loss = log_loss(true_labels, predicted_probs)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cross Entropy Loss: {cross_entropy_loss:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, np.argmax(predicted_probs, axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()

        # Cleanup
        df.drop(columns=[f'trimmed_{target_column}'], inplace=True)

        # Return metrics and probabilities for further inspection
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "cross_entropy_loss": cross_entropy_loss,
            "confusion_matrix": cm,
            "predicted_probs": predicted_probs  # Include reconstructed probabilities
        }
    
class LanguageDetector:
    def __init__(self, dataframe):
        """
        Initializes the LanguageDetector with the provided DataFrame.
        """
        self.dataframe = dataframe

    def detect_language_dataframe_column(self, target_column):
        """
        Detects the language of text in the specified column using langdetect and adds 
        a 'detected_language' column to the DataFrame.
        """
        def detect_language(text):
            try:
                return detect(text)
            except Exception:
                return None

        tqdm.pandas()
        self.dataframe['detected_language'] = self.dataframe[target_column].progress_apply(detect_language)

        return self.dataframe
    
