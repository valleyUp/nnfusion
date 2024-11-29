FROM ngc.nju.edu.cn/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment
RUN  sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list \
     && sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git wget vim

RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN <<EOT 
cat <<EOF > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
EOT

RUN conda install python=3.7 pip numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN conda install -c pytorch magma-cuda110
RUN conda install -c conda-forge onnx
RUN conda install -c conda-forge protobuf=3.9
RUN pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple
RUN pip install future

RUN git clone https://ghproxy.net/https://github.com/snuspl/nimble \
  && sed -i 's|https://github.com|https://ghproxy.net/https://github.com|g' nimble/.gitmodules \
  && cd nimble && git submodule update --init --recursive \
  && cd ..

# COPY nimble /nimble

RUN cd /nimble && USE_NINJA=OFF BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install

RUN pip install --no-dependencies torchvision==0.8 timm pillow

COPY ./artifacts scripts/

WORKDIR scripts
