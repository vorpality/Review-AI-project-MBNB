import matplotlib.pyplot as plt
import os
import csv
import sys

  
def create_new_dir(base_dir):
    counter = 0
    while True:
        new_dir = f"{base_dir}_{counter}" if counter > 0 else base_dir
        if not os.path.exists(new_dir):
            # Create the directory if it doesn't exist
            os.makedirs(new_dir)
            print(f"Directory created: {new_dir}")
            break
        else:
            counter += 1
    return new_dir          
#JSON style formating because what else fits 
def parse_data(files):
    data ={
            'test' :{
                'positive': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []},
                'negative': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []}
            },
            'train' :{
                'positive': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []},
                'negative': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []}
            }
        }
    for file_path in files :
        with open(file_path, 'r') as file:
            lines = file.readlines()
            #current_number = None
            current_type = None
            current_class = None
            #if it works, it works..
            for line in lines:
                if 'number' in line:
                    current_number = line.split('number:')[1]
                elif 'type' in line:
                    current_type = line.split('type:')[1].strip()
                elif 'class' in line:
                    current_class = line.split('class:')[1].strip()
                if 'Precision' in line:
                    (data[current_type])[current_class]['precision'].append(float(line.split(': ')[1]))
                elif 'Recall' in line:
                    data[current_type][current_class]['recall'].append(float(line.split(': ')[1]))
                elif 'F1' in line:
                    data[current_type][current_class]['f1'].append(float(line.split(': ')[1]))
                elif 'Accuracy' in line:
                    data[current_type][current_class]['accuracy'].append(float(line.split(': ')[1]))
                elif 'Files' in line:
                    data[current_type][current_class]['files'].append(int(line.split(': ')[1]))
    return data


def plot_metrics(data, metric_name, data_class, dir):
    plt.figure()
    plt.plot(data['train'][data_class]['files'], data['train'][data_class][metric_name], 'o-', label='Train ' + metric_name)
    plt.plot(data['test'][data_class]['files'], data['test'][data_class][metric_name], 'o-', label='Test ' + metric_name)
    plt.title(f'{metric_name} plot')
    plt.xlabel('Files')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir}/{metric_name}_plot.png')
    plt.close()

def plot_accuracy(data, dir):
    accuracies = {
        'test' : [],
        'train' : []
        }
    for _type in ['test','train']:
        for index in range(0,len(data[_type]['positive']['accuracy'])):
            total_acc = (data[_type]['positive']['accuracy'][index] + data[_type]['negative']['accuracy'][index])/2
            accuracies[_type].append(total_acc)
    plt.figure()
    plt.plot(data['train']['negative']['files'], accuracies['train'], 'o-', label='Train data')
    plt.plot(data['test']['negative']['files'], accuracies['test'], 'o-', label='Test data')
    plt.title('Accuracy plot')
    plt.xlabel('Files')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir}/Train_Test_accuracy.png')
    plt.close()

def create_tables(data, dir):
    with open(dir, 'w', newline='') as table_file:
        writer = csv.writer(table_file)
        _class='positive'
        header = ['Test', 'Precision', 'Recall', 'Accuracy', 'F1', 'Files']
        writer.writerow(header)
        for index in range(0,len(data['test']['positive']['accuracy'])):
            for _type in ['test','train']:
                row = [
                    f'{_type}_{index+1}',
                    data[_type][_class]['precision'][index],
                    data[_type][_class]['recall'][index],
                    data[_type][_class]['accuracy'][index],
                    data[_type][_class]['f1'][index],
                    data[_type][_class]['files'][index]
                    ]
                writer.writerow(row)
        return

        
    
def main():
    args = sys.argv[1:]
    if (len(args) < 2):
        files = {'../output/training_data_accuracy.txt','../output/test_data_accuracy.txt'}
    else :
        files={args[0],args[1]}
    directory = create_new_dir('../output/plots/plots')
    data = parse_data(files)
    file_path = os.path.join(directory, 'tables.csv')
    create_tables(data, file_path)
    plot_accuracy(data,directory)
    for metric in ['precision', 'recall', 'f1', 'accuracy']:
        plot_metrics(data, metric,'positive', directory)
            
if __name__ == "__main__":
    main()

