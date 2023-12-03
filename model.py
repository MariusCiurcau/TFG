from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df_train = pd.read_pickle('df_train.pkl')
    df_test = pd.read_pickle('df_test.pkl')
    df_val = pd.read_pickle('df_val.pkl')
    
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

    flattened_train_data = np.array([item.flatten() for item in df_train.data.values])
    flattened_test_data = np.array([item.flatten() for item in df_test.data.values])
    flattened_val_data = np.array([item.flatten() for item in df_val.data.values])


    names = [
        
        #"Linear SVM",
        #"RBF SVM",
        #"Decision Tree",
        "Neural Net"
    ]

    classifiers = [
        
        #SVC(kernel="linear", C=0.025, random_state=42),
        #SVC(gamma=2, C=1, random_state=42),
        #DecisionTreeClassifier(max_depth=5, random_state=42),
        MLPClassifier(random_state=42, max_iter=300, early_stopping=True)
    ]

    for name, clf in zip(names,classifiers):
        clf.fit(flattened_train_data, df_train.label)
        predicted = clf.predict(flattened_test_data)
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(df_test.label, predicted)}\n"
        )
        disp = metrics.ConfusionMatrixDisplay.from_predictions(df_test.label, predicted)
        disp.figure_.suptitle("Confusion Matrix for " + name)
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()


if __name__ == "__main__":
    main()