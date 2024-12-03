import os
import pandas as pd
import textwrap
import random
from datasets import load_dataset
from IPython.display import display

class LMSYSChat1MHandler:
    def __init__(self, hf_token, streaming=False, verbose=True):
        self.hf_token = hf_token
        self.streaming = streaming
        self.lmsys_dataset = load_dataset(
            'lmsys/lmsys-chat-1m',
            revision="main",
            token=self.hf_token,
            streaming=self.streaming
        )
        self.verbose = verbose
        if verbose:
            print(self.lmsys_dataset)
        self.df_sample = None
        self.df_prompts = None
        self.unwrapped_turns_df = None

        if not self.streaming and verbose:
            print('Data is cached at:\n')
            for file_info in self.lmsys_dataset['train'].cache_files:
                filename = file_info['filename']
                file_size = os.path.getsize(filename)
                i = int((len(filename) - 41) / 2)
                print(f"Filename: {filename[:i]}*{filename[-41:]}\nSize: {file_size} bytes")

    def extract_df_sample(self, n_samples):
        if not self.streaming:    
            df_sample = self.lmsys_dataset['train'].to_pandas().sample(n_samples)
            print(f"Retrieved {len(df_sample)} conversations from lmsys/lmsys-chat-1m")
            if self.verbose and n_samples>4:
                display(df_sample.head(2))
                print('...')
                display(df_sample.tail(2))
            self.df_sample = df_sample

        if self.streaming:
            # Take a sample from the streamed dataset
            streamed_samples = []
            for i, row in enumerate(self.lmsys_dataset['train']):
                streamed_samples.append(row)
                if i + 1 == n_samples:  # Collect only the desired number of samples
                    break

            # Shuffle and convert the collected samples to a Pandas DataFrame
            random.shuffle(streamed_samples)
            df_sample = pd.DataFrame(streamed_samples)
            self.df_sample = df_sample

        return df_sample
    
    def add_turns_to_conversations(self):
        """
        Adds 'turn' keys to each conversation in the 'conversation' column of the dataframe.
        """
        self.df_sample['conversation'] = self.df_sample['conversation'].apply(
            lambda conv: Conversation(conv).add_turns()
        )
        df_with_turns = self.df_sample
        return df_with_turns
    
    def unwrap_turns(self):
        """
        Creates a dataframe where each row corresponds to a pair of user-assistant messages in a conversation and turn.
        The 'prompt' column contains the user's message, and the 'response' column contains the assistant's message.
        Each row includes a 'turn_id' column, which numbers the turns uniquely per conversation.
        """
        paired_data = []
        for _, row in self.df_sample.iterrows():
            conversation_id = row['conversation_id']
            row_data = row.to_dict()
            row_data.pop('conversation')  # Remove the 'conversation' field as it's being unwrapped

            current_prompt = None
            turn_id = None

            for message in row['conversation']:
                if message['role'] == 'user':
                    current_prompt = message['content']
                    turn_id = f"{conversation_id}{message['turn']:03}"  # Create turn_id
                elif message['role'] == 'assistant' and current_prompt is not None:
                    # Create a new row with the user-assistant pair
                    paired_row = {
                        **row_data,
                        'turn_n': message['turn'],
                        'prompt': current_prompt,
                        'response': message['content'],
                    }
                    paired_data.append(paired_row)
                    current_prompt = None  # Reset after pairing

        unwrapped_turns_df = pd.DataFrame(paired_data)
        unwrapped_turns_df.rename(columns={"turn": "conversation_turns"}, inplace=True) # The naming in the original dataset is ambiguous
        self.unwrapped_turns_df = unwrapped_turns_df
        return unwrapped_turns_df

    def extract_prompts(self, filter_language=None, max_char_length=500):
        """
        Extracts user prompts from the sample dataframe, optionally filtering by language and limiting the character length.

        Parameters:
        - filter_language (list of str or None): A list of specific languages to filter prompts by. If None, no language 
          filter is applied. Examples of valid values include ['English'], ['English', 'Portuguese'], or 
          ['Spanish', 'French', 'German'].
        - max_char_length (int): The maximum character length for user prompts to include. Defaults to 500.

        Returns:
        - pd.DataFrame: A DataFrame containing extracted prompts with columns 'prompt' and 'language'.
        """
        df_sample = self.df_sample
        if filter_language:
            extracted_data = df_sample[df_sample['language'].isin(filter_language)].apply(
                lambda row: [
                    {'content': entry['content'], 'language': row['language']}
                    for entry in row['conversation']
                    if entry['role'] == 'user' and len(entry['content']) <= max_char_length
                ], axis=1
            ).explode().dropna()
        else:
            extracted_data = df_sample.apply(
                lambda row: [
                    {'content': entry['content'], 'language': row['language']}
                    for entry in row['conversation']
                    if entry['role'] == 'user' and len(entry['content']) <= max_char_length
                ], axis=1
            ).explode().dropna()

        df_prompts = pd.DataFrame(extracted_data.tolist())
        df_prompts.rename(columns={'content': 'prompt'}, inplace=True)
        self.df_prompts = df_prompts
        if self.verbose and len(df_sample) > 4:
            display(df_prompts.head(2))
            print('...')
            display(df_prompts.tail(2))
        return df_prompts
    
    def extract_prompt_sample(self):
        prompt_sample = self.df_prompts.sample(1)['prompt'].values[0]
        if self.verbose:
            wrapped_message = textwrap.fill(prompt_sample, width=120)
            print(wrapped_message)
        return prompt_sample
        
    def print_language_counts(self, df):
        language_counts = df['language'].value_counts()
        print("Language Record Counts:")
        print(language_counts.to_frame('Count').reset_index().rename(columns={'index': 'Language'}))


class Conversation:
    def __init__(self, conversation_data):
        """
        Initializes the Conversation object with the conversation data.

        Parameters:
        - conversation_data (list): A list of dictionaries representing a conversation.
        """
        self.conversation_data = conversation_data

    def add_turns(self):
        """
        Adds a 'turn' key to each dictionary in the conversation,
        identifying the turn (pair of user and assistant messages).

        Returns:
        - list: The updated conversation with 'turn' keys added.
        """
        turn_counter = 0
        for message in self.conversation_data:
            if message['role'] == 'user':
                turn_counter += 1
            message['turn'] = turn_counter
        return self.conversation_data