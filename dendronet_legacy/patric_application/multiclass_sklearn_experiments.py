import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import train_valid_split_indices

WEIGHT_CLASSES = False
SEEDS = [0, 1, 2, 3, 4]

model_names = ['RandomForestClassifier', 'KNeighborsClassifier']
models = [RandomForestClassifier(), KNeighborsClassifier()]
summarized_accuracies = list()
summarized_accuracies_std = list()
summarized_aucs = list()
summarized_aucs_std = list()


def calculate_metrics(model, inputs, labels, confusion_matrix):
    prediction_results = model.predict(inputs)
    # auc = roc_auc_score(labels, prediction_results)
    num_correct = 0.0
    for pred, label in zip(prediction_results, labels):
        if False not in np.equal(pred, label) :
            num_correct += 1.0
    accuracy = np.round(num_correct / len(labels), 3
                        )
    return [accuracy, 0.0]

x = list()
y = list()
# loading the data
with open('multiclass_clostridium_samples.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
        else:
            y.append(eval(row[2]))
            x.append(eval(row[-1]))
        line_count += 1
num_features = len(x[0])
x = np.array(x)
y = np.array(y)

for model, name in zip(models, model_names):
    print('---------------')
    print('Running model ' + name)
    curr_accuracies = list()
    curr_aucs = list()
    for seed in SEEDS:
        train_indices, valid_indices = train_valid_split_indices(max_index=len(y), random_seed=seed)
        x_train = list()
        x_valid = list()
        y_train = list()
        y_valid = list()
        for ind in train_indices:
            x_train.append(x[ind])
            y_train.append(y[ind])
        for ind in valid_indices:
            x_valid.append(x[ind])
            y_valid.append(y[ind])
        training_confusion_matrix = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        validation_confusion_matrix = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        model.fit(X=x_train, y=y_train)
        train_results = calculate_metrics(model, x_train, y_train, training_confusion_matrix)
        valid_results = calculate_metrics(model, x_valid, y_valid, validation_confusion_matrix)

        curr_accuracies.append(valid_results[0])
        curr_aucs.append(valid_results[1])

        print('Train results for seed ' + str(seed) + ':')
        # print(training_confusion_matrix)
        print('Accuracy: ' + str(train_results[0]))
        print('ROC: ' + str(train_results[1]))
        print(' --- ')
        print('Validation results for seed' + str(seed) + ':')
        # print(validation_confusion_matrix)
        print('Accuracy: ' + str(valid_results[0]))
        print('ROC: ' + str(valid_results[1]))
        print(' ---')

    summarized_accuracies.append(np.round(np.mean(curr_accuracies), 3))
    summarized_accuracies_std.append(np.round(np.std(curr_accuracies), 4))
    summarized_aucs.append(np.round(np.mean(curr_aucs), 3))
    summarized_aucs_std.append(np.round(np.std(curr_aucs), 4))

for i in range(len(models)):
    print('Summarized accuracy for model ' + model_names[i] + ':')
    print(str(summarized_accuracies[i]) + '+/-' + str(summarized_accuracies_std[i]))
    print('Summarized auc for model ' + model_names[i] + ':')
    print(str(summarized_aucs[i]) + '+/-' + str(summarized_aucs_std[i]))
    print(' ')
