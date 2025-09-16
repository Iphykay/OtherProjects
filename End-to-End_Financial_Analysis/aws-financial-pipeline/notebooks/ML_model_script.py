
# OTHER IMPORTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import boto3
import joblib
import pathlib
import argparse
from os                     import path, environ
import pandas               as pd
# from imblearn.over_sampling import SMOTE
import joblib


def model_fn(model_dir):
    clf = joblib.load(path.join(model_dir, 'model.joblib'))
    return clf
#

def load_data(file_path):
    """
    Load and preprocess the training data.

    Input:
    ------
    file_path: Path to the directory containing the training data.

    Output:
    -------
    Returns:
        features: Features of the training dataset.
        label: Labels of the training dataset.
    """
    df       = pd.read_csv(file_path)
    features = df.iloc[:, :-1]
    label    = df.iloc[:, -1]
    return features, label
#


if __name__ == "__main__":

    print("[INFO] Extracting arguments....")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=2) 
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=42)

    # Data, model and output directories
    parser.add_argument('--model-dir', type=str, default=environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='financial_train_data.csv')
    parser.add_argument('--test-file', type=str, default='financial_test_data.csv')

    args, _ = parser.parse_known_args()

    print(args)

    print('SKLearn ersion:', sklearn.__version__)
    print('Joblib version:', joblib.__version__)

    print("[INFO] Loading the train and test data....")
    train_features, train_labels = load_data(path.join(args.train, args.train_file))
    test_features, test_labels   = load_data(path.join(args.test, args.test_file))

    # train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    # test_df  = pd.read_csv(os.path.join(args.test, args.test_file))

    # Using SMOTE to handle the imbalance
    # smote            = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    # x_smote, y_smote = smote.fit_resample(train_features, train_labels)

    print("Training Random Forest Classifier model....", flush=True)
    model = RandomForestClassifier(n_estimators=args.n_estimators,

                                    max_depth=args.max_depth,
                                    min_samples_split=args.min_samples_split,
                                    min_samples_leaf=args.min_samples_leaf,
                                    random_state=args.random_state)
    model.fit(train_features, train_labels)     
    print("Model trained successfully...",flush=True)

    model_path = path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model persisted at {model_path}", flush=True)

    # Predicting the test data
    print("[INFO] Predicting the test data....", flush=True)
    ypred = model.predict(test_features)
    print("Prediction completed successfully.", flush=True)

    # Other metrics
    print("[INFO] Calculating the metrics....", flush=True)
    accuracy      = accuracy_score(test_labels, ypred)
    precision     = precision_score(test_labels, ypred, average='weighted')
    class_report  = classification_report(test_labels, ypred)   
    confusion_mat = confusion_matrix(test_labels, ypred)
    print("Metrics calculated successfully.", flush=True)
