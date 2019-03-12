# Creating a VM for the Fashion MNIST Tutorial

These instructions describe how to install a VM for the [Fashion MNIST Tutorial](README.md).

## Prerequisites

This installation assumes that the VM will be based on *Ubuntu Server 16.04 LTS*.

## Install NVIDIA driver

```
$ sudo bash
# cd /root
# apt-get update && apt-get install -y wget git build-essential linux-headers-`uname -r` dkms
# wget http://us.download.nvidia.com/tesla/410.79/NVIDIA-Linux-x86_64-410.79.run
# chmod 755 NVIDIA-Linux-x86_64-410.79.run
# ./NVIDIA-Linux-x86_64-410.79.run --dkms
# rm -f NVIDIA-Linux-x86_64-410.79.run
```

## Install Docker

```
$ sudo bash
# apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common lsb-core
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
# add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
# apt-get update
# apt-get install -y docker-ce docker-ce-cli containerd.io
```

## Install nvidia-docker2

```
$ sudo bash
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list > /etc/apt/sources.list.d/nvidia-docker.list
# apt-get update
# apt-get install -y nvidia-docker2
# pkill -SIGHUP dockerd
```

## Build model Docker

```
$ git clone https://github.com/horovod/tutorials && cd tutorials/fashion_mnist
$ sudo docker build -t fashion_mnist_tutorial .
```

## Auto-start model Docker

```
$ (echo '#!/bin/sh -e'; echo 'while true; do nvidia-docker run --rm --privileged --network=host --shm-size 4G fashion_mnist_tutorial; sleep 60; done') | sudo tee /etc/rc.local
$ sudo systemctl enable rc-local.service
$ sudo reboot
```

