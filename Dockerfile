# docker 基础镜像构建
# https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586973.0.0.b17922323IJaDN&postId=67720
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3  

# 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

# 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

# 装包
RUN pip --no-cache-dir install tqdm -i https://mirrors.aliyun.com/pypi/simple
RUN pip --no-cache-dir install numpy -i https://mirrors.aliyun.com/pypi/simple

# 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
