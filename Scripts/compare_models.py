import os
from model import predict


reports_folder = '../Reports'

if __name__ == "__main__":
    models = ["../models/resnet18_10_3_ROB", "../models/resnet18_10_3_ROB_AO", "../models/resnet18_10_3_ROB_AO_AQ", "../models/resnet18_10_3_ROB_AO_AQ_MAL"]
    num_classes = [3, 3, 3, 3]
    datasets = ["../Datasets/ROB", "../Datasets/AO", "../Datasets/AQ", "../Datasets/MAL"]

    for model, classes in zip(models, num_classes):
        model_name = os.path.basename(os.path.normpath(model))
        print(f"-------Model: {model_name}, Classes: {classes}-------")
        for dataset in datasets:
            dataset_name = os.path.basename(os.path.normpath(dataset))
            print(f"Dataset: {dataset_name}")
            report, conf_mat = predict(model, dataset + "/images", dataset + "/labels", classes)

            with open(os.path.join(reports_folder, f"report_{model_name}_Dataset_{dataset_name}.txt"), 'w') as txt_file:
                txt_file.write("Model: " + model_name + "\n")
                txt_file.write("Dataset: " + dataset_name + "\n")
                txt_file.write("Classification report:\n")
                txt_file.write("-" * 50 + "\n")
                txt_file.write(report)
                txt_file.write("\n")

                txt_file.write("Confusion matrix:\n")
                txt_file.write("-" * 50 + "\n")
                txt_file.write(conf_mat)