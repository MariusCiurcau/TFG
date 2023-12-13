import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def train_eval_model(df, split=None, subsample=None):
    """
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    
        # Iterate through the first 5 rows of the DataFrame
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            # Read image
            image_path = row['filename']
            image = Image.open(image_path)
    
            # Plot image
            axes[i].imshow(image)
            axes[i].axis('off')
    
            # Add label below the image
            label = row['label']
            axes[i].text(0.5, -0.05, f'Label: {label}', color='black', size=10, ha='center', va='center', transform=axes[i].transAxes)
    
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()"""

    if split is None:
        split = [0.7, 0.15, 0.15]

    df['data'] = df['data'].apply(lambda x: x.flatten())
    #flattened_data = np.array([item.flatten() for item in df.data.values])
    X_aux, X_test, y_aux, y_test = train_test_split(
        df[['filename', 'data']], df.label.values, test_size=split[1], shuffle=True, random_state=1, stratify=df.label.values
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_aux, y_aux, test_size=split[2] / (1 - split[1]), shuffle=True, random_state=1, stratify=y_aux
    )

    if subsample is not None:
        X_train.reset_index(drop=True,inplace=True)
        y_train_df = pd.DataFrame({'label': y_train})
        df_concat = pd.concat([X_train, y_train_df], axis=1)
        print(df_concat['filename'].isna().sum())
        df_filtered = df_concat[df_concat['filename'].str.endswith('_0.jpg')]
        
        # Identify the minority and majority classes
        minority_class = 1
        majority_class = 0

        # Identify the number of samples in the minority class
        minority_class_samples = y_train_filtered.value_counts()[minority_class]

        # Identify the number of samples you want to subsample from the majority class
        # You can adjust this based on your desired balance
        desired_majority_samples = minority_class_samples * 2  # for a 2:1 ratio

        # Separate the majority and minority class samples
        majority_samples = y_train_filtered[y_train_filtered == majority_class].index
        minority_samples = y_train_filtered[y_train_filtered == minority_class].index

        # Subsample from the majority class
        majority_samples_subsampled = resample(
            majority_samples, replace=False, n_samples=desired_majority_samples, random_state=1
        )

        # Combine the subsampled majority class with the minority class
        y_train_subsampled = y_train_filtered.loc[minority_samples.union(majority_samples_subsampled)]


    #print(np.unique(y_train, return_counts=True))
    names = [
        # "Linear SVM",
        # "RBF SVM",
        # "Decision Tree",
        "Neural Net"
    ]

    classifiers = [
        # SVC(kernel="linear", C=0.025, random_state=42),
        # SVC(gamma=2, C=1, random_state=42),
        # DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(random_state=42, max_iter=300, early_stopping=True)
    ]

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = f"Classification report for classifier {clf}:\n{metrics.classification_report(y_test, predicted)}"
        print(report)
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix for " + name)
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()

    return report, str(disp.confusion_matrix)


if __name__ == "__main__":
    df = pd.read_pickle("../df.pkl")
    train_eval_model(df, split=[0.7, 0.15, 0.15],subsample=1)