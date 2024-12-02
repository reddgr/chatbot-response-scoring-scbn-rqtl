import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, HBox, VBox, Output
import textwrap
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

class LabelingWidget:
    def __init__(self):
        self.labeled_data = pd.DataFrame(columns=["text", "label"])
        self.session_complete = False
    
    def manual_labeling(self, df, classifier, label_map):
        """
        Manual labeling function for user interaction. Returns labeled_data when the session ends.
        """
        output = Output()
        current_index = {"value": 0}  # Track the current index in df

        # Create buttons
        correct_button = Button(description="CORRECT", button_style="success")
        wrong_button = Button(description="WRONG", button_style="danger")
        pass_button = Button(description="PASS", button_style="info")
        end_button = Button(description="END SESSION", button_style="warning")

        # Bind button actions
        def on_correct_button_clicked(_):
            label_text()
            current_index["value"] += 1
            display_text()

        def on_wrong_button_clicked(_):
            label_text(override_label=1 - int(classifier([df.iloc[current_index["value"]]["text"]])[0]["label"].split("_")[-1]))
            current_index["value"] += 1
            display_text()

        def on_pass_button_clicked(_):
            current_index["value"] += 1
            display_text()

        def on_end_button_clicked(_):
            output.clear_output(wait=True)
            with output:
                print("### Labeling Session Ended ###")
                print(f"Total labels recorded: {len(self.labeled_data)}")
                print("Labeled data:")
                display(self.labeled_data)
                self.session_complete = True

        correct_button.on_click(on_correct_button_clicked)
        wrong_button.on_click(on_wrong_button_clicked)
        pass_button.on_click(on_pass_button_clicked)
        end_button.on_click(on_end_button_clicked)

        # Display the interface once
        display(VBox([HBox([correct_button, wrong_button, pass_button, end_button]), output]))

        def display_text():
            """
            Function to display the current text and prediction.
            """
            output.clear_output(wait=True)  # Clear the output area for the current example
            with output:
                if current_index["value"] >= len(df):
                    print("### Labeling Complete ###")
                    print("Labeled data:")
                    display(self.labeled_data)
                    self.session_complete = True
                    return
                text = df.iloc[current_index["value"]]["text"]
                result = classifier([text])[0]
                predicted_label = int(result["label"].split("_")[-1])
                prob = result["score"]
                wrapped_text = textwrap.fill(text, width=120)
                label_str = label_map[predicted_label]
                print(f"### Predicted: {label_str} ({prob:.3f}) ###")
                print(wrapped_text)

        def label_text(override_label=None):
            """
            Function to add the labeled data to the labeled_data DataFrame.
            """
            text = df.iloc[current_index["value"]]["text"]
            result = classifier([text])[0]
            predicted_label = int(result["label"].split("_")[-1])
            correct_label = override_label if override_label is not None else predicted_label
            self.labeled_data = pd.concat(
                [self.labeled_data, pd.DataFrame({"text": [text], "label": [correct_label]})],
                ignore_index=True,
            )

        # Initialize by displaying the first text
        display_text()

    def update_dataset(self, dataset_name, split_name, hf_token, new_dataset_records=None):
        """
        Updates a HuggingFace dataset with the labeled data or a custom dataframe.

        Parameters:
        - dataset_name: The name of the dataset on the HuggingFace Hub.
        - hf_token: The HuggingFace token for authentication.
        - split_name: The split of the dataset to update ('train' or 'test').
        """
        if not new_dataset_records:
            new_dataset_records = Dataset.from_pandas(self.labeled_data)
        else:
            new_dataset_records = new_dataset_records
        dataset = load_dataset(dataset_name, token=hf_token)
        updated_split = concatenate_datasets([dataset[split_name], new_dataset_records])
        updated_dataset = DatasetDict({
            'train': dataset['train'] if split_name == 'test' else updated_split,
            'test': dataset['test'] if split_name == 'train' else updated_split
        })
        updated_dataset.push_to_hub(dataset_name, token=hf_token)
        print(f"Successfully pushed {len(new_dataset_records)} records to {dataset_name} {split_name} split.")   