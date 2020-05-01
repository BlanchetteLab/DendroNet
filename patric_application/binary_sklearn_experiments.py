import csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from utils import train_valid_split_indices

WEIGHT_CLASSES = False
SEEDS = [0, 1, 2, 3, 4]

model_names = ['AdaBoost', 'RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier', 'SVC']
models = [AdaBoostClassifier(), RandomForestClassifier(), LogisticRegression(), GradientBoostingClassifier(), SVC()]
summarized_accuracies = list()
summarized_accuracies_std = list()
summarized_aucs = list()
summarized_aucs_std = list()


def calculate_metrics(model, inputs, labels, confusion_matrix):
    prediction_results = model.predict(inputs)
    num_correct = 0
    for i in range(0, len(labels)):
        if prediction_results[i] == 1 and labels[i] == 1:
            confusion_matrix['tp'] += 1
            num_correct += 1
        elif prediction_results[i] == 0 and labels[i] == 0:
            confusion_matrix['tn'] += 1
            num_correct += 1
        elif prediction_results[i] == 1 and labels[i] == 0:
            confusion_matrix['fp'] += 1
        else:
            confusion_matrix['fn'] += 1
    auc = roc_auc_score(labels, prediction_results)
    accuracy = float(num_correct) / float(len(labels))
    return [accuracy, auc]


x = list()
y = list()
# loading the data
with open('clostridium_samples_combined_resistance.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
        else:
            if row[2] == 'Resistant':
                y.append(1.0)
            else:
                y.append(0.0)
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
        print(training_confusion_matrix)
        print('Accuracy: ' + str(train_results[0]))
        print('ROC: ' + str(train_results[1]))
        print(' --- ')
        print('Validation results for seed' + str(seed) + ':')
        print(validation_confusion_matrix)
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
    # print(str(summarized_accuracies[i]))
    print('Summarized auc for model ' + model_names[i] + ':')
    print(str(summarized_aucs[i]) + '+/-' + str(summarized_aucs_std[i]))
    # print(str(summarized_aucs[i]))
    print(' ')
