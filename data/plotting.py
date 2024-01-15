import matplotlib.pyplot as plt
import os
import csv

def create_new_dir(base_dir):
    counter = 0
    while True:
        # Create a new directory name by appending a counter
        new_dir = f"{base_dir}_{counter}" if counter > 0 else base_dir
        
        # Check if the directory exists
        if not os.path.exists(new_dir):
            # Create the directory since it doesn't exist
            os.makedirs(new_dir)
            print(f"Directory created: {new_dir}")
            break  # Exit the loop
        else:
            # Increment the counter if the directory exists
            counter += 1
    return new_dir          

def parse_data(file_path):
    data = {
        'positive': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []},
        'negative': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'files': []}
    }
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_test = None
        for line in lines:
            if 'positive' in line:
                current_test = 'positive'
            elif 'negative' in line:
                current_test = 'negative'
            if 'Precision' in line:
                data[current_test]['precision'].append(float(line.split(': ')[1]))
            elif 'Recall' in line:
                data[current_test]['recall'].append(float(line.split(': ')[1]))
            elif 'F1' in line:
                data[current_test]['f1'].append(float(line.split(': ')[1]))
            elif 'Accuracy' in line:
                data[current_test]['accuracy'].append(float(line.split(': ')[1]))
            elif 'Files' in line:
                data[current_test]['files'].append(int(line.split(': ')[1]))
    return data

def create_dataframe(data):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    # Reorder the DataFrame columns to match the order you want in the table
    column_order = ['Training Size', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 
                    'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1']
    df = df[column_order]
    return df

def save_table(df, filename):
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Table saved to {filename}")

def plot_metrics(data, metric_name, dir):
    plt.figure()
    plt.plot(data['positive']['files'], data['positive'][metric_name], 'o-', label='Train ' + metric_name)
    plt.plot(data['negative']['files'], data['negative'][metric_name], 'o-', label='Test ' + metric_name)
    plt.title(f'{metric_name} vs. Training Size')
    plt.xlabel('Training Size')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir}/{metric_name}_vs_Training_Size.png')
    plt.close()

def main():
    directory = create_new_dir('plots')
    files = { 'test_data_accuracy.txt', 'training_data_accuracy.txt'}
    for file in files :
        data = parse_data(file)
        
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            plot_metrics(data, metric, directory)
            
if __name__ == "__main__":
    main()

