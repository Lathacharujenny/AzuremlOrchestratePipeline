import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

def train_model(data_path, model_output_path, test_data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_output_path)
     # Save the test set for evaluation
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(test_data_path, index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--model_output_path', type=str, help='path to model output')
parser.add_argument('--test_data_path', type=str, help='path to test data')
args = parser.parse_args()
train_model(args.data_path, args.model_output_path, args.test_data_path)
