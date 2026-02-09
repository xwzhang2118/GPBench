# =========================
# 1) 基础镜像：带 conda 的 Miniconda
# =========================
FROM continuumio/miniconda3

# =========================
# 2) 工作目录：容器中项目放哪
#    为什么：后续 COPY/运行命令都基于这个目录，路径统一，不容易出错
# =========================
WORKDIR /app

# =========================
# 3) 复制 environment.yml 并创建 conda 环境（先做环境，后拷代码）
#    为什么放这里：
#    - Docker 有缓存机制：如果你代码改了但依赖没变，就不需要重装环境，build 更快
#    - 依赖安装是最耗时的一步，放前面最大化利用缓存
# =========================
COPY environment.yml /app/environment.yml

# 这里要改成你的 environment.yml 里的 name:
ARG CONDA_ENV=Benchmark
ENV CONDA_ENV=${CONDA_ENV}
ENV PATH=/opt/conda/envs/${CONDA_ENV}/bin:$PATH

RUN conda env create -f /app/environment.yml && \
    conda clean -a -y

# =========================
# 4) 复制项目代码
#    为什么放这里：代码变动频繁，把 COPY . 放后面，避免每次改代码都触发重新装依赖
# =========================
COPY . /app

# =========================
# 5) 运行方式设计（关键）
#    你的项目有很多“可运行文件”，不能写死 CMD ["python", "xxx.py"]
#    做法：
#    - 统一提供一个默认命令：打印用法提示
#    - 真正运行哪个入口，由你 docker run 时传参决定
# =========================
SHELL ["bash", "-lc"]

CMD echo "Image is ready." && \
    echo "Run a specific entry like:" && \
    echo "  docker run --rm gpbench:1.0 python method_class/BayesA/BayesA_class.py [args...]" && \
    echo "  docker run --rm gpbench:1.0 python method_reg/BayesA/BayesA.py [args...]" && \
    echo "" && \
    echo "Current /app:" && \
    ls -la