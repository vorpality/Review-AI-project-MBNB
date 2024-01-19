import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import csv

def load_data(directory, max_files_per_class=None):
    data = []
    labels = []
    for label in ["pos", "neg"]:
        path = os.path.join(directory, label)
        files = os.listdir(path)
        random.shuffle(files)
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(path, file)
                with open(file_path, 'r', encoding='utf-8') as file_open:
                    data.append(file_open.read())
                    labels.append(1 if label == "pos" else 0)
    return data, labels


def predict_review(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return "Positive" if prediction[0] == 1 else "Negative"


def plot_metrics(data, metric,output_dir):
    train_info = []
    test_info =[]
    files = []
    for test in data:
        if test['type'] == 'test':
            test_info.append(test[metric])
        else:
            train_info.append(test[metric])
            files.append(test['files'])
        
    plt.figure()
    plt.plot(files, test_info, label='Test', marker='o')
    plt.plot(files, train_info, label='Train', marker='o')
    plt.title(f'{metric} plot')
    plt.xlabel('Files')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/{metric}_files.png')
    plt.close()
        
def create_tables(data, output_dir):
    with open(f'{output_dir}/tables.csv', 'w', newline='') as table_file:
        writer = csv.writer(table_file)
        header = ['Test', 'Precision', 'Recall', 'Accuracy', 'F1', 'Files']
        writer.writerow(header)
        for test in data:
          row = [
            test['test'],
            test['precision'],
            test['recall'],
            test['accuracy'],
            test['f1_score'],
            test['files']
            ]
          writer.writerow(row)
        



def main():
    #output dir
    output_dir = r'../results/part b'
    #preparing training sizes
    starting_size = 100
    size_step = 100
    step_amount = 20

    test_size = 1000
    
    end_size = starting_size + (step_amount*size_step)
    results = []
    evaluation_counter = 1
    for n_files in range (starting_size, end_size, size_step):

        train_data, train_labels = load_data(r'../assets/includes/aclImdb/train', max_files_per_class=n_files)
        test_data, test_labels = load_data(r'../assets/includes/aclImdb/test', max_files_per_class=test_size)

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_data)
        X_test = vectorizer.transform(test_data)

        model = MultinomialNB()
        model.fit(X_train, train_labels)

        predicted = model.predict(X_test)
        
        # Evaluate on Test data
        test_accuracy = accuracy_score(test_labels, predicted)
        test_precision = precision_score(test_labels, predicted)
        test_recall = recall_score(test_labels, predicted)
        test_f1 = f1_score(test_labels, predicted)

        # Evaluate on Training data
        train_predicted = model.predict(X_train)
        train_accuracy = accuracy_score(train_labels, train_predicted)
        train_precision = precision_score(train_labels, train_predicted)
        train_recall = recall_score(train_labels, train_predicted)
        train_f1 = f1_score(train_labels, train_predicted)

        results.append({
            'test' : f'test_{evaluation_counter}',
            'type' : 'test',
            'accuracy' :test_accuracy,
            'precision' :test_precision,
            'recall' : test_recall,
            'f1_score' : test_f1,
            'files' : n_files
            })

        results.append({
            'test' :f'train_{evaluation_counter}',
            'type': 'train',
            'accuracy' : train_accuracy,
            'precision' :train_precision,
            'recall' :train_recall,
            'f1_score' :train_f1,
            'files' : n_files
            })
        evaluation_counter += 1

    create_tables(results, output_dir)
    for metric in ['accuracy','precision','recall','f1_score']:
        plot_metrics(results, metric, output_dir)
    print("done")



# Run the main function
if __name__ == "__main__":
    main()
