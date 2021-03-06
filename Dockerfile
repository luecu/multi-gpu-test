FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update

WORKDIR /multi-gpu-test

VOLUME /multi-gpu-test

RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
RUN apt-get install -y python3 python3-pip python3-tk python3-numpy python3-pandas python3-sklearn libglib2.0-0 libsm6 libxext6
RUN pip3 install Keras-Applications==1.0.6 keras==2.2.4 image scikit-learn matplotlib h5py tensorflow-gpu==1.12.0 opencv-python img_rotate
# download data to data folder

# Install coremltools from source to support Keras 2.2.4 (as of dec, 6th 2018)
# Install Protocol Buffer
RUN apt-get install -y git build-essential cmake zlib1g-dev wget unzip python3-dev python-dev
ENV PROTOBUF_VERSION 3.6.0
RUN wget -O /multi-gpu-test/protoc-$PROTOBUF_VERSION.zip https://github.com/google/protobuf/releases/download/v$PROTOBUF_VERSION/protoc-$PROTOBUF_VERSION-linux-x86_64.zip && \
	unzip /multi-gpu-test/protoc-$PROTOBUF_VERSION.zip -d /multi-gpu-test/protoc && \
	rm /multi-gpu-test/protoc-$PROTOBUF_VERSION.zip
RUN mv /multi-gpu-test/protoc/bin/* /usr/local/bin/ && \
	mv /multi-gpu-test/protoc/include/* /usr/local/include/ && \
	rm -rf /multi-gpu-test/protoc
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN git clone https://github.com/apple/coremltools.git && \
	cd coremltools && \
	cmake . -DPYTHON=$(which python) -DPYTHON_CONFIG=$(which python-config) && \
	make && \
	pip3 install -e .
	
ADD comparative_learning.py /workdir/comparative_learning.py	
ADD models.py /workdir/models.py	
ADD occasi_data.py /workdir/occasi_data.py	
ADD plot_helper.py /workdir/plot_helper.py	
ADD run.py /workdir/run.py

ADD data/2018-12-17_collaborative_filtering_user_preferences.npz /multi-gpu-test/data/2018-12-17_collaborative_filtering_user_preferences.npz
ADD data/data_occasi-images.npz /multi-gpu-test/data/data_occasi-images.npz
ADD data/ratings-test.csv /multi-gpu-test/data/ratings-test.csv
ADD data/ratings-train.csv /multi-gpu-test/data/ratings-train.csv
ADD data/features_vgg16_layer_block1_pool.npy /multi-gpu-test/data/features_vgg16_layer_block1_pool.npy


ENTRYPOINT ["python3", "run.py"]