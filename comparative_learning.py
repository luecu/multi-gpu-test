import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plot_helper import plot_confusion_matrix
from sklearn.metrics import classification_report
import os

def tensorboard_dir(filename):    
    origin_filename = "tensorboard-logs/" + filename
    filename = origin_filename
    counter = 0
    while os.path.isdir(filename):
        counter += 1
        filename = origin_filename + "-" + str(counter)
    print("tensorboard directory " + filename)
    return filename

def fit(model, occasi_data, nr_of_epochs, model_summary, batch_size=32, title='', filename='', verbose=0, save_model=True, initial_epoch=0):
    model_summary.at['title'] = title
    model_summary.at['epochs'] = initial_epoch + nr_of_epochs
    model_summary.at['tensorboard_dir'] = nr_of_epochs
    model_summary.at['model_json'] = model.to_json()
    
    #tensorboard_logdir = tensorboard_dir(filename)
    #model_summary.at['tensorboard_dir'] = tensorboard_logdir
    #tensorboard = TensorBoard(log_dir=(tensorboard_logdir), histogram_freq=1, write_grads=True)

    steps_per_epoch = occasi_data.steps_for_all_inputs(batch_size) # TODO define properly
    model_summary.at['steps_per_epoch'] = steps_per_epoch
    model_summary.at['batch_size'] = batch_size

    cv_user_features, cv_img1_features, cv_img2_features, cv_y = occasi_data.cross_validation_triplets()
    history = model.fit_generator(occasi_data.train_triplets_generator(batch_size),
                                  validation_data=([cv_user_features, cv_img1_features, cv_img2_features], cv_y),
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=nr_of_epochs,
                                  verbose=verbose,
                                  workers=6,
                                  use_multiprocessing=True
                                  #use_multiprocessing=True
                                 )
                                  #callbacks=[tensorboard])
    
    model_summary.at['history'] = history
    
    print(history.history['loss'])
    end_loss = round(history.history['loss'][-1],5)
    min_val_loss = round(min(history.history['val_loss']),5)
    max_val_acc = round(max(history.history['val_acc']),5)
    print("end loss = " + str(end_loss))
    print("min_val_loss = " + str(min_val_loss))
    print("max_val_acc = " + str(max_val_acc))
    
    filename = filename + '_withendloss_' + str(end_loss)
    pd.Series(history.history['loss']).plot(logy=True)
    pd.Series(history.history['val_loss']).plot(logy=True)
    pd.Series(history.history['acc']).plot(logy=True)
    pd.Series(history.history['val_acc']).plot(logy=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')
    plt.title(title)
    plt.suptitle(filename)
    plt.savefig("images/loss-" + filename + ".jpeg")
    plt.show()

    if save_model:
        filename = filename  + '.h5'
        model.save("models/" + filename)
        model_summary.at['model_filename'] = filename
    
    return model

    