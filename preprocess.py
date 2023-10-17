import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_dataset(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Perform preprocessing steps
    # Step 1: standardization for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Step 2: label encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Save the preprocessed dataset
    data.to_csv(output_file, index=False)
    print("Dataset preprocessed and saved to", output_file)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Your Dataset")
    parser.add_argument("input_file", help="Input Dataset (CSV)")
    parser.add_argument(
        "output_file", help="Output Preprocessed Dataset (CSV)")
    args = parser.parse_args()

    preprocess_dataset(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
