from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import numpy as np
import os
from langdetect import detect
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    
    def classify_single_text(self, text):
        trimmed_text = self.tokenize_and_trim(text)
        result = self.classifier(trimmed_text)[0]
        numeric_label = int(result['label'].split('_')[-1])
        label = self.label_map[numeric_label]
        score = result['score']
        printed_result = f" {score*100:.1f}% {label}"
        return printed_result

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
            df (pd.DataFrame): Input dataframe containing a 'label' column.
            target_column (str): The name of the column to classify.

        Requirements:
            - The dataframe must include a 'label' column for comparison with predictions.

        Returns:
            dict: A dictionary containing accuracy, F1 score, cross-entropy loss, 
                and the confusion matrix.
        """
        # Convert pandas dataframe to Dataset
        dataset = Dataset.from_pandas(df)

        # Define a processing function for tokenization and classification
        def process_data(batch):
            trimmed_text = self.tokenize_and_trim(batch[target_column])
            result = self.classifier(trimmed_text)
            score = result[0]['score']
            label = result[0]['label']
            return {
                'trimmed_text': trimmed_text,
                'predicted_prob_0': score if label == 'LABEL_0' else 1 - score,
                'predicted_prob_1': 1 - score if label == 'LABEL_0' else score,
            }

        # Apply processing with map
        processed_dataset = dataset.map(process_data, batched=False)

        # Convert back to pandas dataframe
        processed_df = processed_dataset.to_pandas()

        # Extract predicted probabilities and true labels
        predicted_probs = processed_df[['predicted_prob_0', 'predicted_prob_1']].values
        true_labels = df['label'].values

        # Calculate metrics
        accuracy = accuracy_score(true_labels, np.argmax(predicted_probs, axis=1))
        if len(self.label_map):
            average='binary'
        else:
            average='weighted'
        f1 = f1_score(true_labels, np.argmax(predicted_probs, axis=1), average=average)
        cross_entropy_loss = log_loss(true_labels, predicted_probs)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cross Entropy Loss: {cross_entropy_loss:.4f}")

        # Confusion matrix
        cm = confusion_matrix(true_labels, np.argmax(predicted_probs, axis=1))
        cmap = plt.cm.Blues
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=cmap)
        plt.show()

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
    

# Classifier with Tensorflow backend
class TensorflowClassifier(Classifier):
    def __init__(self, model_path, label_map, verbose=False):
        super().__init__(model_path, label_map, verbose=False)
        self.is_tensorflow = False
        
        if self._is_tensorflow_model(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Adjust as per training tokenizer
            self.is_tensorflow = True
            if verbose:
                print("Loaded TensorFlow model.")
        else:
            if verbose:
                print("Fallback to HuggingFace pipeline.")

    def _is_tensorflow_model(self, model_path):
        return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb"))

    def classify(self, text):
        if self.is_tensorflow:
            inputs = self.tokenizer(text, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="np")
            logits = self.model.predict([inputs["input_ids"], inputs["attention_mask"]])
            probabilities = tf.nn.softmax(logits).numpy()
            label_id = np.argmax(probabilities, axis=-1).item()
            return {
                "label": f"LABEL_{label_id}",
                "score": probabilities.max()
            }
        else:
            return self.classifier(text)[0]

    def classify_dataframe_column(self, df, target_column, feature_suffix):
        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(
            lambda text: self.tokenizer.decode(
                self.tokenizer(text, truncation=True, max_length=self.tokenizer.model_max_length)["input_ids"],
                skip_special_tokens=True
            )
        )

        if self.is_tensorflow:
            results = [self.classify(text) for text in df[f'trimmed_{target_column}']]
        else:
            results = [self.classifier(text)[0] for text in df[f'trimmed_{target_column}']]

        df[f'pred_label_{feature_suffix}'] = [
            self.label_map[int(result['label'].split('_')[-1])] for result in results
        ]
        df[f'prob_{feature_suffix}'] = [result['score'] for result in results]
        df.drop(columns=[f'trimmed_{target_column}'], inplace=True)
        return df
