# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ARG PYTHONPATH="tmp"

RUN : \
    && apt-get update \
    # && add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-venv \
        python3.7-venv \
        libglfw3 \
        libglfw3-dev \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        freeglut3-dev \
        git-all \
        xvfb \
        mesa-utils \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python -m venv /venv --system-site-packages
ENV PATH=/venv/bin:$PATH

RUN python --version

COPY requirements.txt requirements.txt
RUN git clone https://github.com/duckietown/gym-duckietown.git

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

RUN pip install --no-cache-dir -e gym-duckietown

ENV LD_LIBRARY_PATH=/Tmp/glx:$LD_LIBRARY_PATH
ADD runxvfb.sh /runxvfb.sh
RUN chmod a+x /runxvfb.sh
ENV DISPLAY=:1

COPY --link . /workspaces


ENV PYTHONPATH=/workspaces:$PYTHONPATH

RUN mkdir /scratch

WORKDIR /workspaces
CMD ["/runxvfb.sh"]
