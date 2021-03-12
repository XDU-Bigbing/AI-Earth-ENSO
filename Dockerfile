# docker 基础镜像构建
# https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586973.0.0.b17922323IJaDN&postId=67720
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3  

USER root
# 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

# 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

# 装包
# 会卸载自带的，默认的版本不支持
RUN pip --no-cache-dir install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip --no-cache-dir install tqdm -i https://mirrors.aliyun.com/pypi/simple
RUN pip --no-cache-dir install numpy -i https://mirrors.aliyun.com/pypi/simple
RUN pip --no-cache-dir install netCDF4 -i https://mirrors.aliyun.com/pypi/simple
RUN pip --no-cache-dir install zipfile36 -i https://mirrors.aliyun.com/pypi/simple

# 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
