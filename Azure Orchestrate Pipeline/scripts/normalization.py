import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler

def normalization(data_path, output_path):
    df = pd.read_csv(data_path)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach']
    scaler = StandardScaler()
    df[numeric_cols]=scaler.fit_transform(df[numeric_cols])
    df.to_csv(output_path, index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to the data')
parser.add_argument('--output_path', type=str, help='path to the output data')
args = parser.parse_args()
normalization(args.data_path, args.output_path)
