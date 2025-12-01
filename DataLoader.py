import pandas as pd
import os
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path

BASE_PATH = '/workspaces/CMP9794-Advanced-Artificial-Intelligence/Data/'
FOLDER = 'annotated-question-answer-pairs-for-clinical-notes-in-the-mimic-iii-database-1.0.0'

class Dataset_t(Enum):
    JSON = 1,
    CSV = 2,

class DataLoader:
    def __init__(self, dataset: Dataset_t, create_pie = False):
        self.dataset_t = dataset
        self.creat_pie = create_pie
        if dataset == Dataset_t.JSON:
            self.data, self.code_to_title = self.parse_json_qa()
        elif dataset == Dataset_t.CSV:
            self.data, self.code_to_title = self.parse_csv(self.creat_pie)

    def parse_json_qa(self):
        data = pd.read_json(Path(BASE_PATH)/FOLDER/'test.final.json')
        df = pd.json_normalize(data['data'],
                            record_path=['paragraphs'],
                            meta=['title'],
                            errors='ignore')
        qas = pd.json_normalize(df['qas'].explode())
        df_qas = pd.concat([df.drop(columns=['qas']), qas.reset_index(drop=True)], axis=1)

        df_answers = pd.json_normalize(df_qas['answers'].explode())
        df_qas['answer_text'] = df_answers['text']

        df_final = df_qas[['title', 'context', 'question', 'answer_text']]

        return df_final, None
    
    def parse_csv(self, create_pie=False):
        df = pd.read_csv(Path(BASE_PATH)/'MIMIC-SAMPLED.csv')

        null_rows = df[df.isnull().any(axis=1)]
        print(null_rows.shape)
        df.isnull().sum()

        df_copy = df.copy()
        df_copy.dropna(subset=["ICD9 Diagnosis"], inplace=True)
        df_copy.dropna(subset=["note_category"], inplace=True)
        df_copy.isnull().sum()

        # Calculate the number of rows in the original and modified dataframes
        original_rows = df.shape[0]
        dropped_rows = original_rows - df_copy.shape[0]
        not_dropped_rows = df_copy.shape[0]

        # Create labels and sizes for the pie chart
        labels = ['Dropped Rows', 'Not Dropped Rows']
        sizes = [dropped_rows, not_dropped_rows]
        colors = ['#ff9999','#66b3da']

        if create_pie:
            # Create the pie chart
            plt.figure(figsize=(5, 5))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title('Proportion of Dropped vs Not Dropped Rows')
            plt.savefig("DroppedPie.png")

        df_sampled = df_copy.sample(n=1312, random_state=42) # Shuffle

        unique_pairs = df[['ICD9 Diagnosis', 'SHORT_TITLE']].drop_duplicates()
        code_to_title = dict(unique_pairs.values)

        df_sampled = df_sampled.reset_index()

        return df_sampled, code_to_title



if __name__ == '__main__':
    dataset = DataLoader(Dataset_t.JSON)
    print(dataset.data.head())