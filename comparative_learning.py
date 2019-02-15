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
    
    return model

    