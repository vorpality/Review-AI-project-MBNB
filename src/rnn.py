import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from tensorflow.keras.utils import plot_model
import random
import csv

(x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train_imdb])
x_test_imdb = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test_imdb])


from tqdm import tqdm

train_doc_length = 0
for doc in tqdm(x_train_imdb):
  tokens = str(doc).split()
  train_doc_length += len(tokens)

print('\nTraining data average document length =', (train_doc_length 
                                                  / len(x_train_imdb)))

VOCAB_SIZE = 100000
SEQ_MAX_LENGTH = 250
vectorizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, 
                                               output_mode='int', 
                                               ngrams=1, name='vector_text',
                                               output_sequence_length=SEQ_MAX_LENGTH)
with tf.device('/CPU:0'):
  vectorizer.adapt(x_train_imdb)

  vector_model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(1,), dtype=tf.string),
      vectorizer
])
vector_model.predict([['awesome movie']])

dummy_embeddings = tf.keras.layers.Embedding(1000, 5)
dummy_embeddings(tf.constant([1, 2, 3])).numpy()

from tensorflow.keras.utils import plot_model

def get_rnn(num_layers=1, emb_size=64, h_size=64):
  inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='txt_input') # ['awesome movie']
  x = vectorizer(inputs) # [1189, 18, 0, 0, 0, 0, ...]
  x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),
                                output_dim=emb_size, name='word_embeddings',
                                mask_zero=True)(x)
  for n in range(num_layers):
    if n != num_layers - 1:
      x = tf.keras.layers.SimpleRNN(units=h_size, 
                                    name=f'rnn_cell_{n}', 
                                    return_sequences=True)(x)
    else:
      x = tf.keras.layers.SimpleRNN(units=h_size, name=f'rnn_cell_{n}')(x)

  x = tf.keras.layers.Dropout(rate=0.5)(x)
  o = tf.keras.layers.Dense(units=1, activation='sigmoid', name='lr')(x)
  return tf.keras.models.Model(inputs=inputs, outputs=o, name='simple_rnn')

def calculate_f1(precision, recall):
    f1 = 2*(precision * recall)/(precision + recall)
    return f1

def plot_epoch_results(epochs, output_dir):
    for epoch_array in epochs:
        current_epochs = []
        current_loss = []
        for i in range (len(epoch_array['array'])):
            current_epochs.append(epoch_array['array'][i]['amount'])
            current_loss.append(epoch_array['array'][i]['loss'])
        size = epoch_array['files']
        plt.plot(current_epochs,current_loss, label=f'Files : {size}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs / Loss')
    plt.savefig(f'{output_dir}/epoch_loss.png')

def plot_metric(data, metric, output_dir):
    files =[]
    info = []
    for entry in data:
        files.append(entry['files'])
        info.append(entry[metric])
    plt.figure()
    plt.plot(files,info)
    plt.xlabel('Files')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} plot')
    plt.savefig(f'{output_dir}/{metric}_training_size.png')
    plt.close()

def create_tables(data, output_dir):
  with open(f'{output_dir}/tables.csv', 'w', newline='') as table_file:
    writer = csv.writer(table_file)
    header = ['Test', 'Precision', 'Recall', 'Accuracy', 'F1', 'Files']
    writer.writerow(header)
    counter = 1
    for test in data:
      row = [
        counter,
        test['precision'],
        test['recall'],
        test['accuracy'],
        test['f1_score'],
        test['files']
        ]
      writer.writerow(row)
      counter += 1


#output directory
output_dir = r'../results/part c'
#preparing training sizes
starting_size = 100
size_step = 100
step_amount = 2
epochs_per_model = 10

end_size = starting_size + (step_amount*size_step)
train_results=[]
test_results= []
epoch_results= []
for training_size in range (starting_size, end_size, size_step):
#shuffling data in unison and resizing
    x_train_imdb,y_train_imdb = shuffle(x_train_imdb,y_train_imdb)
    current_x = x_train_imdb[:training_size]
    current_y = y_train_imdb[:training_size]
    
    imdb_rnn = get_rnn()
    imdb_rnn.summary()

    imdb_rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                     optimizer=tf.keras.optimizers.SGD(),
                     metrics=[
                         tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.Precision(name='precision'),
                         ]
                     )
    

    history = imdb_rnn.fit(x=current_x, y=current_y,
                 epochs=epochs_per_model, verbose=1, batch_size=32)

    epoch_losses = history.history['loss']
    #saving epoch results in array of arrays for plotting
    epoch_loss=[]
    epoch_amount = 1
    for loss in epoch_losses:
        epoch_loss.append({
            'loss' : loss,
            'amount' : epoch_amount
            })
        epoch_amount +=1
    epoch_results.append({'array' : epoch_loss, 'files' : training_size})
    #evaluating and saving against train/test data
    test = imdb_rnn.evaluate(current_x, current_y)
    test_results.append({
        'files' : training_size,
        'accuracy' : test[1],
        'precision': test[2],
        'recall': test[3],
        'f1_score': calculate_f1(test[2],test[3])
        })
                  
        

plot_epoch_results(epoch_results, output_dir)

for metric in ['accuracy','precision','recall','f1_score']:
    plot_metric(test_results, metric, output_dir)

create_tables(test_results, output_dir)
    
