FROM horovod/horovod:0.17.0.post1-tf1.14.0-torch1.2.0-mxnet1.5.0-py3.6

RUN pip install jupyterlab

RUN pip install git+https://www.github.com/keras-team/keras-contrib.git

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    python -c 'import keras; keras.datasets.fashion_mnist.load_data()' && \
    ldconfig

COPY . /tutorial

WORKDIR /tutorial

ENV SHELL=/bin/bash

ENTRYPOINT ["/bin/bash", "-c", "(tensorboard --logdir . &) && (jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='horovod_lab')"]

