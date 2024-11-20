FROM ngc.nju.edu.cn/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

RUN  sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list \
     && sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget vim \
  libgtest-dev libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev llvm-dev \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

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

RUN conda install python~=3.10 pip cmake && conda clean --all

RUN pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

RUN pip install --no-cache-dir --default-timeout=1000 torch==1.12 torchvision==0.13 timm==0.5.4 einops \
    onnx==1.12 onnxruntime-gpu==1.12 onnxconverter_common==1.12 \
    attrs cloudpickle decorator psutil synr tornado xgboost==1.5.0 regex pandas pytest \
    && pip install "numpy<2" \
    && rm -rf ~/.cache/pip

RUN git clone https://mirror.ghproxy.com/https://github.com/nox-410/tvm --recursive -b welder \
  && sed -i 's|https://github.com|https://mirror.ghproxy.com/https://github.com|g' tvm/.gitmodules \
  && mkdir tvm/build && cd tvm/build && cp ../cmake/config.cmake . \
  && echo "set(USE_LLVM ON)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake \
  && cmake .. && make -j2
ENV PYTHONPATH /root/tvm/python:$PYTHONPATH

RUN git clone https://mirror.ghproxy.com/https://github.com/nox-410/nnfusion -b welder \
  && mkdir nnfusion/build \
  && cd nnfusion/build && cmake .. && make -j2
ENV PATH /root/nnfusion/build/src/tools/nnfusion:$PATH

RUN git clone https://mirror.ghproxy.com/https://github.com/nox-410/cutlass -b welder
ENV CPLUS_INCLUDE_PATH /root/cutlass/include:$CPLUS_INCLUDE_PATH

COPY . welder/
ENV PYTHONPATH /root/welder/python:$PYTHONPATH

CMD bash
