import pandas as pd
import argparse
from azureml.core import Run

def data_preprocessing(data_path, output_path):
    run = Run.get_context()
    df = run.input_datasets['raw_data'].to_pandas_dataframe()
    df['chol'].fillna(df['chol'].mean(), inplace=True)
    df['thalach'].fillna(df['thalach'].mean(), inplace=True)
    df.drop(columns=['slope', 'ca', 'thal', 'exang'], inplace=True)
    df.to_csv(output_path, index=False)



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to the data')
parser.add_argument('--output_path', type=str, help='path to the output data')
args = parser.parse_args()
data_preprocessing(args.data_path, args.output_path)


