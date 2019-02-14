from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dot, Subtract, Lambda, Dense, Activation, Dropout, BatchNormalization, Flatten,MaxPooling2D, Conv2D
from keras.optimizers import Nadam, Adam
from keras import backend as K
import keras

import pandas as pd
import numpy as np
from keras.utils import multi_gpu_model

def simple_architectur_with_flattening(image_vector_input_size, 
                       user_vector_input_size, 
                       learning_rate,
                       model_summary,
                       activation_function_dense = 'relu',
                       amount_of_nodes=512,
                       dropout_feature_extract=0,
                       dropout=0,          
                       distance_activation_functions = ['sigmoid', None, 'sigmoid'], # activation_diff_fc, activation_diff_fc (inference only), activation_end
                      nr_of_gpus=1):
    model_summary.at['model'] = 'simple_architectur'
    model_summary.at['learning_rate'] = learning_rate
    model_summary.at['activation_feature_extract'] = activation_function_dense
    
    model = private_simple_architectur_with_flattening(image_vector_input_size, user_vector_input_size, learning_rate, model_summary, activation_function_dense, inference=False,amount_of_nodes=amount_of_nodes, dropout_feature_extract=dropout_feature_extract, dropout=dropout, distance_activation_functions=distance_activation_functions,nr_of_gpus=nr_of_gpus)
    
    inference_model = private_simple_architectur_with_flattening(image_vector_input_size, user_vector_input_size, learning_rate, model_summary, activation_function_dense, inference=True,amount_of_nodes=amount_of_nodes, dropout_feature_extract=dropout_feature_extract, dropout=dropout, distance_activation_functions=distance_activation_functions,nr_of_gpus=nr_of_gpus)
    
    return model, inference_model

def private_simple_architectur_with_flattening(image_vector_input_size, 
                       user_vector_input_size,         
                       learning_rate,
                       model_summary,
                       activation_function_dense = 'relu',
                       inference = False,
                       amount_of_nodes=512,
                       dropout_feature_extract=0,
                       dropout=0, 
                       distance_activation_functions=['sigmoid', None, 'sigmoid'],
                       nr_of_gpus=1): 
    multi_dimensional = len(list(image_vector_input_size)) > 2
    img1_input = Input(shape=list(image_vector_input_size[1:]), name='img1_features')
    user_input = Input(shape=[user_vector_input_size], name='user_preferences')
    
    img2_input = Input(shape=list(image_vector_input_size[1:]), name='img2_features')
    
    model_summary.at['dropout_feature_extract'] = dropout_feature_extract
    model_summary.at['img_layers'] = [amount_of_nodes]
    
    
    img_cnn_flatten = Flatten(name='img_cnn_flatten')
    
    img_dnn_fc_1 = Dense(amount_of_nodes, name='img_dnn_dense_1')
    img_dnn_bn_1 = BatchNormalization(name="img_dnn_bn_1")
    img_dnn_ac_1 = Activation(activation_function_dense, name='img_dnn_ac_1')
    img_dnn_dropout_1 = Dropout(dropout_feature_extract, name='img_dnn_dropout_1')

    model_summary.at['user_layers'] = [amount_of_nodes,amount_of_nodes]
    user_dnn_fc_1 = Dense(amount_of_nodes, name='user_dnn_dense_1')
    user_dnn_bn_1 = BatchNormalization(name="user_dnn_bn_1")
    user_dnn_ac_1 = Activation(activation_function_dense, name='user_dnn_ac_1')
    user_dnn_dropout_1 = Dropout(dropout_feature_extract, name='user_dnn_dropout_1')
    user_dnn_fc_2 = Dense(amount_of_nodes, name='user_dnn_dense_2')
    user_dnn_bn_2 = BatchNormalization(name="user_dnn_bn_2")
    user_dnn_ac_2 = Activation(activation_function_dense, name='user_dnn_ac_2')
    user_dnn_dropout_2 = Dropout(dropout_feature_extract, name='user_dnn_dropout_2')

    pooled_img1_input = img1_input
    pooled_img2_input = img2_input
    
    conv_layer_counter = 1
    
    flattened_size = np.prod(image_vector_input_size[1:])
    max_feature_vector_size = 2048
    dimension = image_vector_input_size[3]
    while flattened_size > max_feature_vector_size:
        if dimension > 32:
            dimension = int(dimension / 2)
        img_cnn_conv = Conv2D(filters=dimension, kernel_size=(2,2), name='img_cnn_conv_' + str(conv_layer_counter))
        img_cnn_max_pooling = MaxPooling2D(pool_size=(2, 2), name='img_cnn_max_pooling_' + str(conv_layer_counter))
        img_cnn_bn = BatchNormalization(name='img_cnn_bn_' + str(conv_layer_counter))


        pooled_img1_input = img_cnn_bn(img_cnn_max_pooling(img_cnn_conv(pooled_img1_input)))
        pooled_img2_input = img_cnn_bn(img_cnn_max_pooling(img_cnn_conv(pooled_img2_input)))
        print(pooled_img1_input.shape)
        flattened_size = np.prod(pooled_img1_input.shape[1:])
        conv_layer_counter = conv_layer_counter + 1
     
    if multi_dimensional:
        pooled_img1_input = img_cnn_flatten(pooled_img1_input)
        pooled_img2_input = img_cnn_flatten(pooled_img2_input)
    
    img1_vector_model = img_dnn_dropout_1(img_dnn_ac_1(img_dnn_bn_1(img_dnn_fc_1(pooled_img1_input))))
    img2_vector_model = img_dnn_dropout_1(img_dnn_ac_1(img_dnn_bn_1(img_dnn_fc_1(pooled_img2_input))))


    user_vector_model = user_dnn_dropout_2(user_dnn_ac_2(user_dnn_bn_2(user_dnn_fc_2(user_dnn_dropout_1(user_dnn_ac_1(user_dnn_bn_1(user_dnn_fc_1(user_input))))))))

    model_summary.at['dropout_diff_fc'] = dropout
    shared_fc_layer = Dense(1, name='shared_diff_fc_layer') # fc_top and fc_bottom (Evt. fixed weights, damit es nicht lernt)
    shared_fc_dropout_layer = Dropout(dropout, name='shared_diff_dropout_layer')
    if distance_activation_functions[0] is not None:
        shared_fc_activation = Activation(distance_activation_functions[0], name='shared_diff_activation')
    
    # Merge Layers for img_1 and user
    img1_user_model = Subtract(name='user_img1_diff')([img1_vector_model, user_vector_model])
    img1_user_model = Lambda(lambda x: x ** 2, name='user_img1_sqr')(img1_user_model)
    img1_user_model = shared_fc_dropout_layer(shared_fc_layer(img1_user_model))
    model_summary.at['activation_diff_fc'] = distance_activation_functions[0]
    if distance_activation_functions[0] is not None:
        img1_user_model = shared_fc_activation(img1_user_model)
    
    if not inference:
        img2_user_model = Subtract(name='user_img2_diff')([img2_vector_model, user_vector_model])
        img2_user_model = Lambda(lambda x: x ** 2, name='user_img2_sqr')(img2_user_model)
        img2_user_model = shared_fc_dropout_layer(shared_fc_layer(img2_user_model))
        if distance_activation_functions[0] is not None:
            img2_user_model = shared_fc_activation(img2_user_model)
            
        # merge img_1 and img_2 merged layers
        diff_3 = Subtract(name='diff3')([img1_user_model, img2_user_model])
        
        model_summary.at['activation_end'] = distance_activation_functions[2]
        if distance_activation_functions[2]:
            diff_3 = Activation(distance_activation_functions[2], name="activation_end")(diff_3)
        
        model = keras.Model([user_input, img1_input, img2_input], diff_3)
        
    if inference:
        model_summary.at['activation_diff_fc_inference'] = distance_activation_functions[1]
        if distance_activation_functions[1]:
            img1_user_model = Activation(distance_activation_functions[1], name='diff_fc_only_inference_activation')(img1_user_model)
        
        model = keras.Model([user_input, img1_input], img1_user_model)
    
    
    optimizer = Adam(lr=learning_rate)
    model_summary.at['optimizer'] = 'Adam'
    
    if nr_of_gpus > 1:
        model = multi_gpu_model(model, gpus=nr_of_gpus)
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model_summary.at['loss_function'] = 'binary_crossentropy'
    
    return model