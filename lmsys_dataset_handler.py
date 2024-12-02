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