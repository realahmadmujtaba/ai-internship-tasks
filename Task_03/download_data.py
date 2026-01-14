import pandas as pd
import urllib.request
import os

# Download the UCI Heart Disease dataset
def download_heart_disease_dataset():
    # The dataset is available at UCI ML Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        # Download the data
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')

        # Process the data
        lines = data.strip().split('\n')
        processed_lines = []

        for line in lines:
            if line.strip():
                values = line.split(',')
                processed_line = []
                for val in values:
                    val = val.strip()
                    if val == '?':
                        processed_line.append(float('nan'))
                    else:
                        try:
                            processed_line.append(float(val))
                        except ValueError:
                            processed_line.append(val)
                processed_lines.append(processed_line)

        # Create DataFrame
        df = pd.DataFrame(processed_lines, columns=column_names)

        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = df['target'].apply(lambda x: 0 if x == 0.0 else 1)

        # Save to CSV
        df.to_csv('heart_disease.csv', index=False)
        print(f"Dataset downloaded and saved. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Create a sample dataset for demonstration if download fails
        import numpy as np
        n_samples = 303  # Typical size of the heart disease dataset

        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })

        # Introduce some NaN values to simulate missing data
        nan_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        for col in ['ca', 'thal']:
            col_nan_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
            df.loc[col_nan_indices, col] = np.nan

        df.to_csv('heart_disease.csv', index=False)
        print(f"Sample dataset created. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df

if __name__ == "__main__":
    df = download_heart_disease_dataset()