import pandas as pd
import argparse
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(test_data_path, model_path, output_path):
    df = pd.read_csv(test_data_path)
    #df = df.dropna()
    X_test = df.drop(columns=['target'], axis=1)
    y_test = df['target']
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # save metrics to a file
    with open(output_path, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision Score: {precision}\n')
        f.write(f'Recall : {recall}\n')

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, help="Path to the test data file")
parser.add_argument('--model_path', type=str, help="Path to the trained model file")
parser.add_argument('--output_path', type=str, help="Path to save evaluation metrics")
args = parser.parse_args()
evaluate_model(args.test_data_path, args.model_path, args.output_path)

