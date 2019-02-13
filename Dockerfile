FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

#RUN git clone https://github.com/luecu/multi-gpu-test.git
#RUN cd multi-gpu-test

WORKDIR /workdir

RUN apt-get update
#RUN apt-get install -y locales
#RUN locale-gen en_US.UTF-8
#ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
RUN apt-get install -y python3 python3-pip python3-tk python3-numpy python3-pandas python3-sklearn libglib2.0-0 libsm6 libxext6
RUN pip3 install Keras-Applications==1.0.6 keras==2.2.4 image scikit-learn matplotlib h5py tensorflow-gpu==1.12.0 opencv-python img_rotate
# download data to data folder

ADD comparative_learning.py /workdir/comparative_learning.py
ADD models.py /workdir/models.py
ADD occasi_data.py /workdir/occasi_data.py
ADD plot_helper.py /workdir/plot_helper.py
ADD run.py /workdir/run.py

ADD data/2018-12-17_collaborative_filtering_user_preferences.npz /workdir/data/2018-12-17_collaborative_filtering_user_preferences.npz
ADD data/data_occasi-images.npz /workdir/data/data_occasi-images.npz
ADD data/ratings-test.csv /workdir/data/ratings-test.csv
ADD data/ratings-train.csv /workdir/data/ratings-train.csv
ADD data/features_vgg16_layer_block1_pool.npy /workdir/data/features_vgg16_layer_block1_pool.npy


ENTRYPOINT ["python3", "run.py"]