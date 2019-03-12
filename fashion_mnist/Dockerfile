FROM horovod/horovod:0.16.0-tf1.12.0-torch1.0.0-mxnet1.4.0-py3.5

RUN pip install jupyterlab

RUN pip install git+https://www.github.com/keras-team/keras-contrib.git

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    python -c 'import keras; keras.datasets.fashion_mnist.load_data()' && \
    ldconfig

# Until https://github.com/horovod/horovod/pull/858 is merged
RUN ln -s /usr/local/bin/mpirun /usr/local/bin/horovodrun

COPY . /tutorial

WORKDIR /tutorial

ENV SHELL=/bin/bash

ENTRYPOINT ["/bin/bash", "-c", "(tensorboard --logdir . &) && (jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='horovod_lab')"]

