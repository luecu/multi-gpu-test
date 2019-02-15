import occasi_data as occ_data
import models
import comparative_learning

import sklearn
print(sklearn.__version__)

#VGG-Layer-Test
#('features_vgg16_layer_global_max_pooling2d_3.npy',0),
#image_inputs = ['features_vgg16_layer_block5_pool.npy', 'features_vgg16_layer_block4_pool.npy', 'features_vgg16_layer_block3_pool.npy', 'features_vgg16_layer_block2_pool.npy', 'features_vgg16_layer_block1_pool.npy']
image_inputs = ['features_vgg16_layer_block1_pool.npy']
nr_of_epochs = 1
for image_input in image_inputs:
    title = 'vgg16-layers-' + image_input
    filename = title
    
    occasi_data = occ_data.data(image_features_file='data/' + image_input, feature_scaling=False)
    model_summary = occasi_data.model_summary
	
    
	
    model, inference_model = models.simple_architectur_with_flattening(image_vector_input_size=occasi_data.image_features.shape,
                                                 user_vector_input_size=occasi_data.user_vector_input_size(),
                                                 learning_rate=0.001,
                                                 amount_of_nodes=512,
                                                 model_summary=model_summary,
                                                 dropout=0.5
                                                 )
    #model.summary()
    model = comparative_learning.fit(model,
                      occasi_data,
                      nr_of_epochs,
                      model_summary,
                      title=title, 
                      filename=filename,
                      save_model=True,
					  batch_size=8*64,
                      verbose=1)

