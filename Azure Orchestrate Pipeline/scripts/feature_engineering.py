import pandas as pd
import argparse

def feature_engineering(data_path, output_path):
    df = pd.read_csv(data_path)
    df['aged_binned'] = pd.cut(df['age'], bins=[0,30,50,70,100], labels=['young', 'Middle-aged', 'Senior', 'Old'])
    df = pd.get_dummies(df, columns=['sex', 'cp', 'restecg', 'aged_binned'])
    df.to_csv(output_path, index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--output_path', type=str, help='path to output data')
args = parser.parse_args()
feature_engineering(args.data_path, args.output_path)
