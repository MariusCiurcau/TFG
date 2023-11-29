from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_pickle('df.pkl')
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

    flattened_data = np.array([item.flatten() for item in df.data.values])
    X_train, X_test, y_train, y_test = train_test_split(
        flattened_data, df.label.values, test_size=0.2, shuffle=True, random_state=1, stratify=df.label.values
    )

    clf = MLPClassifier(random_state=42, max_iter=300, early_stopping=True)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    print(df['label'].value_counts())


if __name__ == "__main__":
    main()