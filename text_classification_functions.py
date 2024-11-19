from transformers import pipeline
from tqdm import tqdm
import torch
from langdetect import detect

class Classifier:

    def __init__(self, model_path, label_map):
        self.model_path = model_path
        self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0 if torch.cuda.is_available() else -1)
        self.label_map = label_map
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

    def classify_dataframe_column(self, df, target_column, feature_suffix):
        max_length = self.classifier.tokenizer.model_max_length

        def tokenize_and_trim(text):
            inputs = self.classifier.tokenizer(text, truncation=True, max_length=max_length, return_tensors="tf")
            return self.classifier.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

        tqdm.pandas()
        df[f'trimmed_{target_column}'] = df[target_column].progress_apply(tokenize_and_trim)

        results = []
        for text in tqdm(df[f'trimmed_{target_column}'].tolist(), desc="Classifying"):
            result = self.classifier(text)
            results.append(result[0])

        df[f'label_{feature_suffix}'] = [self.label_map[int(result['label'].split('_')[-1])] for result in results]
        df[f'prob_{feature_suffix}'] = [result['score'] for result in results]
        return df
    


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