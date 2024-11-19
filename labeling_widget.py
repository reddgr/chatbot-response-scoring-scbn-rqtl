import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, HBox, VBox, Output
import textwrap

class LabelingWidget:
    def __init__(self):
        self.labeled_data = pd.DataFrame(columns=["text", "label"])
        self.session_complete = False

    def manual_labeling(self, df_extracted, classifier, label_map):
        """
        Manual labeling function for user interaction. Returns labeled_data when the session ends.
        """
        output = Output()
        current_index = {"value": 0}  # Track the current index in df_extracted

        def display_text():
            """
            Function to display the current text and prediction.
            """
            # clear_output(wait=True)
            output.clear_output(wait=True)  # Clear the output area for the current example
            with output:
                if current_index["value"] >= len(df_extracted):
                    print("### Labeling Complete ###")
                    print("Labeled data:")
                    display(self.labeled_data)
                    self.session_complete = True
                    return
                text = df_extracted.iloc[current_index["value"]]["text"]
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
            text = df_extracted.iloc[current_index["value"]]["text"]
            result = classifier([text])[0]
            predicted_label = int(result["label"].split("_")[-1])
            correct_label = override_label if override_label is not None else predicted_label
            self.labeled_data = pd.concat(
                [self.labeled_data, pd.DataFrame({"text": [text], "label": [correct_label]})],
                ignore_index=True,
            )

        def on_correct_button_clicked(_):
            """
            Action for the CORRECT button.
            """
            label_text()
            current_index["value"] += 1
            display_text()

        def on_wrong_button_clicked(_):
            """
            Action for the WRONG button.
            """
            label_text(override_label=1 - int(classifier([df_extracted.iloc[current_index["value"]]["text"]])[0]["label"].split("_")[-1]))
            current_index["value"] += 1
            display_text()

        def on_pass_button_clicked(_):
            """
            Action for the PASS button.
            """
            current_index["value"] += 1
            display_text()

        def on_end_button_clicked(_):
            """
            Action for the END button.
            """
            clear_output(wait=True)
            print("### Labeling Session Ended ###")
            print(f"Total labels recorded: {len(self.labeled_data)}")
            print("Labeled data:")
            display(self.labeled_data)
            self.session_complete = True

        # Create buttons
        correct_button = Button(description="CORRECT", button_style="success")
        wrong_button = Button(description="WRONG", button_style="danger")
        pass_button = Button(description="PASS", button_style="info")
        end_button = Button(description="END SESSION", button_style="warning")

        # Bind button actions
        correct_button.on_click(on_correct_button_clicked)
        wrong_button.on_click(on_wrong_button_clicked)
        pass_button.on_click(on_pass_button_clicked)
        end_button.on_click(on_end_button_clicked)

        # Display the interface
        display(VBox([HBox([correct_button, wrong_button, pass_button, end_button]), output]))
        display_text()