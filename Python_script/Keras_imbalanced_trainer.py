import argparse
import datetime
import itertools
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split

#python -m Keras_trainer --batch_size 500 --epochs 10 --run_iter 0 --exp_name 'TEMP' --model_id 'v1'
print(tf.__version__)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=500,         type=int,    help="Batch size")
parser.add_argument("--epochs",     default=10,          type=int,    help="Number of training epochs")
parser.add_argument("--run_iter",   default=0,           type=int,    help="Run iteration")
parser.add_argument("--model_id",   default='10000epochs',        type=str,    help="Name of the model")
parser.add_argument("--exp_name",   default='Keras_models/10000ephocs_10_500batch',      type=str,    help="Folder name for results")

def read_data(path):
    file_path = path
    data = pd.read_csv(file_path)
    colnames = list(data['Unnamed: 0'])
    data = data.T
    data.columns =  colnames
    data = data.drop(axis=0, labels='Unnamed: 0')
    return data

def read_labels(path, select_cl):
    file_path = path
    labels = pd.read_csv(file_path)
    labels.index = list(labels['Unnamed: 0'])
    labels = labels.drop(axis=1, labels='Unnamed: 0')
    new_factors = np.arange(len(select_cl)).tolist()
    cls = select_cl
    refactor_cls = []
    for items in labels.cl:
        if items in cls: 
            index = cls.index(items)
            refactor_cls = refactor_cls + [new_factors[index]]
        else:
            refactor_cls = refactor_cls + [np.nan]
    labels["old_factor_cl"] = labels["factor_cl"]
    labels["factor_cl"] = refactor_cls 
    return labels

def split_data_intwo(data, labels, test_size, cvset):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                              test_size = test_size,
                                              random_state = cvset)

    return train_data, test_data, train_labels, test_labels

def build_model_regular(n_features, n_cluster, HL_size, dropout_rate, 
                        learning_rate, HL_multiply, regularization=0.01):
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(n_features,)),
            keras.layers.Dropout(dropout_rate, noise_shape=None, seed=None),
            keras.layers.Dense(HL_size * HL_multiply, 
                               kernel_regularizer= regularizers.l2(regularization), 
                               bias_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(n_cluster, activation=tf.nn.softmax)
    ])
    optim = keras.optimizers.Adam(lr= learning_rate, decay=0.001)
    model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def main(batch_size=500, epochs=10, 
    run_iter=0, model_id='VISP',exp_name='TEMP'):
    base_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/patchseq-work-dir/Patchseq_vs_FACs_cre_analysis/mouse_patchseq_VISp_20181220_collapsed40_cpm/'
    results_path = base_path + '/' + exp_name +'/'
    #Path(results_path).mkdir(parents=True, exist_ok=True)
    fileid = model_id + '_ri_' + str(run_iter)
    facs_output_id = "facs_membership_" + str(run_iter)

    V1_cl = pd.read_csv(base_path + "select_cl.csv")['x'].tolist()

    FACs_data = read_data(base_path + "FACs_norm.csv")
    FACs_labels = read_labels(base_path + "FACs_correct_labels.csv", V1_cl)
    FACs_labels = FACs_labels['factor_cl']
    FACs_cells = FACs_data.index.tolist()
    FACs_membership = pd.DataFrame(0, index=FACs_cells, columns=V1_cl)
    
    train_data , test_data, train_labels, test_labels = split_data_intwo(FACs_data, FACs_labels, 
                                                                         test_size = 1464, cvset=run_iter)
    test_cells = test_data.index.tolist()

    start_time = timeit.default_timer()
    model = build_model_regular(n_features=4020, n_cluster=93, HL_size=10,
                                dropout_rate=0.6,  regularization=0.01, learning_rate=0.01,
                                HL_multiply=1)
    results = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,  verbose=0)
    
    facs_memb = model.predict(test_data)
    facs_memb = pd.DataFrame(facs_memb, index= test_cells, columns=V1_cl)
    FACs_membership.loc[test_cells] = FACs_membership.loc[test_cells] + facs_memb.loc[test_cells]
    
    print(datetime.datetime.now())

    elapsed = timeit.default_timer() - start_time
    score, acc = model.evaluate(test_data, test_labels,
                       batch_size=500, verbose=0)

    print('Test accuracy:', acc)
    print('-------------------------------')
    print('Training time:', elapsed)
    print('-------------------------------')

    FACs_membership.to_csv(results_path + facs_output_id + '.csv')
    model.save(results_path + fileid + '.h5')

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
