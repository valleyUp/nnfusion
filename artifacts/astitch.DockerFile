FROM docker.udayun.com/bladedisc/bladedisc:latest-runtime-torch1.7.1-cu110

RUN pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple
RUN pip install einops timm transformers

COPY . scripts/

WORKDIR scripts
