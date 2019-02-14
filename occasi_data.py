import plot_helper

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

class data:

    def __init__(self, 
                 image_features_file="data/2018-12-17_features_resnet_model_dont_include_top.npy",
                 ratings_train_file = "data/ratings-train.csv",
                 ratings_test_file = "data/ratings-test.csv",
                 user_features_file = "data/2018-12-17_collaborative_filtering_user_preferences.npz",
                 original_images_file = "data/data_occasi-images.npz",
                 image_input_type = 'pretrained_feature', # can be 'pretrained_feature' OR 'pixel_image'
                 feature_scaling=True):
        self.image_input_type = image_input_type
        
        print("init image features from " + image_features_file)
        # import images
        self.image_features = np.load(image_features_file)
        #Reshape, Da Densenet 1314,7,7,1920 Shape hat
        if(feature_scaling and 'DenseNet' in image_features_file): # TODO reshape würde ich bereits vor dem Speichern in ein File machen.
            print(self.image_features.shape)
            nsamples, nx, ny, nz = self.image_features.shape
            self.image_features = self.image_features.reshape((nsamples,nx*ny*nz))
        
        if feature_scaling:
            print("feature scaling")
            self.image_featuers_scaler = StandardScaler().fit(self.image_features)
            self.image_features = self.image_featuers_scaler.transform(self.image_features)
        print("loadad image features with shape " + str(self.image_features.shape))
        
        print("load ratings from " + ratings_train_file)
        ratings = pd.read_csv(ratings_train_file,sep=',',names="user_id,item_id,rating,user_ext_id,item_ext_id".split(","))
        self.train_ratings, self.cross_validation_ratings= train_test_split(ratings, test_size=0.1)
        print("loaded " + str(len(self.train_ratings)) + " train ratings and " + str(len(self.cross_validation_ratings)) + " cross validation ratings")
        
        print("load test ratings from " + ratings_test_file)
        self.test_ratings = pd.read_csv(ratings_test_file,sep=',',names="user_id,item_id,rating,user_ext_id,item_ext_id".split(","))
        print("loaded " + str(len(self.test_ratings)) + " test ratings")
        
        print("load user features from " + user_features_file)
        loaded_file = np.load(user_features_file)
        self.user_features = loaded_file['user_preferences']
        if feature_scaling:
            self.user_features_scaler = StandardScaler().fit(self.user_features)
            self.user_features = self.user_features_scaler.transform(self.user_features)
        print("loaded user features with shape " + str(self.user_features.shape))
        
        print("load original images from " + original_images_file)
        imported_file = np.load(original_images_file)
        self.original_images = imported_file['imgs']
        print("loaded " + str(self.original_images.shape[0]) + " images")
        
        self.model_summary = pd.Series({'user_vector':user_features_file,
                                        'img_vector':image_features_file,
                                        'image_input_type':image_input_type})
        
    def image_features(self):
        return self.image_features
    
    def image_inputs(self):
        return self.image_features if self.image_input_type == 'pretrained_feature' else self.original_images
    
    def train_triplets_with_imageids(self):
        return ratings_to_triplets(self.train_ratings, self.train_ratings.user_id.unique())
    
    def steps_for_all_inputs(self, batch_size):
        # Returns the number of steps it need with the given batch_size to see each input at least once
        train_triplets = ratings_to_triplets(self.train_ratings, self.train_ratings.user_id.unique())
        return int(len(train_triplets) / batch_size) 
    
    def train_triplets_generator(self, batch_size):
        while True:
            train_triplets = ratings_to_triplets(self.train_ratings, self.train_ratings.user_id.unique())
            
            for i in range(0, len(train_triplets), batch_size):
                end_index = i + batch_size
                end_index = min(end_index, len(train_triplets))
                
                user_features, img1_features, img2_features, y = triplets_to_inputs(train_triplets[i:end_index], self.image_inputs(), self.user_features)

                yield ([np.array(user_features), np.array(img1_features), np.array(img2_features)], np.array(y))  
        
    def test_triplets_generator(self, nr_of_steps):
        user_ids = self.test_ratings.user_id.values
        img_ids = self.test_ratings.item_id.values
        y_ratings = self.test_ratings.rating.values
        
        nr_examples = len(user_ids)
        
        batch_size = int(nr_examples / (nr_of_steps-1))
        
        for i in range(0,nr_examples,batch_size):
            end_index = min(i + batch_size, nr_examples)
            yield [
                [self.get_user_features(user_id) for user_id in user_ids[i:end_index]], 
                [self.get_image_features(img_id) for img_id in img_ids[i:end_index]]
            ]
        
        
    def cross_validation_triplets(self):
        cv_triplets = ratings_to_triplets(self.cross_validation_ratings, self.train_ratings.user_id.unique())
        user_features, img1_features, img2_features, y = triplets_to_inputs(cv_triplets, self.image_inputs(), self.user_features)
        return user_features, img1_features, img2_features, y
    
    def user_vector_input_size(self):
        return self.user_features.shape[1]
    
    def image_vector_input_size(self):
        return [self.image_features.shape[1]] # TODO remove array or add array on user_vector_input_size method
    
    def get_user_features(self, user_id):
        return self.user_features[user_id]
    
    def get_image_features(self, item_id):
        return self.image_inputs()[item_id]
    
    def user_id_of(self, user_ext_id):
        user_ratings = self.train_ratings[self.train_ratings['user_ext_id'] == user_ext_id].user_id
        if len(user_ratings) == 0:
            print("could not find user " + user_ext_id)
            return
        
        return user_ratings.iloc[0]
    
    def plot_original_image(self, img_id):
        img = self.original_images[img_id]
        plot_helper.plot_image(img)
        
def triplets_to_inputs(triplets, image_features, user_features):
    #print("start 1 " + str(len(triplets)))
    user_vector = []
    img1_vector = []
    img2_vector = []
    y = []
    
    #print("start " + str(user_vector.shape))
    for idx, triplet in enumerate(triplets):
        if not user_features[triplet['user']].any():
            continue # ignore triplet if all values are 0 for user
            
        user_vector.append(user_features[triplet['user']])

        if round(random.uniform(0, 1)) == 1:
            # img1 as positiv image => y equals 0 TODO correct?
            img1_vector.append(image_features[triplet['image_pos']])
            img2_vector.append(image_features[triplet['image_neg']])
            y.append(0)
        else:
            img1_vector.append(image_features[triplet['image_neg']])
            img2_vector.append(image_features[triplet['image_pos']])
            y.append(1)
        
    #print("end " + str(user_vector.shape))
    return np.array(user_vector), np.array(img1_vector), np.array(img2_vector), np.array(y)
 
 
def ratings_to_triplets(ratings, all_user_ids):
    triplets = []

    for user_id in all_user_ids:
        user_ratings = ratings[ratings['user_id'] == user_id]

        rat_0 = user_ratings[user_ratings['rating'] == 0].item_id.values
        rat_1 = user_ratings[user_ratings['rating'] == 1].item_id.values
        rat_2 = user_ratings[user_ratings['rating'] == 2].item_id.values

        for img_id in rat_2:
            triplets.append({
              'user': user_id,
              'image_pos': img_id,
              'image_neg': random.choice(rat_1) if len(rat_1) > 0 else random.choice(rat_0)
            })  

        # for each rating 1 add a rating 0 image as negativ example
        for img_id in rat_1:
            if (len(rat_0) < 1):
                break
            triplets.append({
              'user': user_id,
              'image_pos': img_id,
              'image_neg': random.choice(rat_0)
          })
        
    np.random.shuffle(triplets)
    return triplets