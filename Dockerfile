FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
SHELL ["/bin/bash", "-c"]

# Source code
ADD . /app
WORKDIR /app
ENV PYTHONPATH=/app/src:$PYTHONPATH

# APT dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    git \
    curl \
    sysstat \
    wget \
    unzip \
	# for fpocket
    libnetcdf-dev

# Install miniconda3 to /miniconda
RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh
RUN bash Miniconda3-4.5.12-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-4.5.12-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y -q conda
RUN conda create -y -n deeplytough python=3.6
RUN conda install -y -n deeplytough -c openbabel -c anaconda -c acellera -c psi4 biopython openbabel htmd
RUN source activate deeplytough; pip install --upgrade pip; pip install --no-cache-dir -r /app/requirements.txt

RUN conda create -y -n deeplytough_mgltools
RUN conda install -y -n deeplytough_mgltools -c bioconda mgltools

# rot covariant convolutions (includes also the 'experiments' code)
RUN source activate deeplytough; git clone https://github.com/mariogeiger/se3cnn && cd se3cnn && git reset --hard 6b976bea4ea17e1bd5655f0f030c6e2bb1637b57 && mv experiments se3cnn; sed -i "s/exclude=\['experiments\*'\]//g" setup.py && python setup.py install && cd .. && rm -rf se3cnn;
RUN source activate deeplytough; git clone https://github.com/AMLab-Amsterdam/lie_learn && cd lie_learn && python setup.py install && cd .. && rm -rf lie_learn

# fpocket2
RUN curl -O https://netcologne.dl.sourceforge.net/project/fpocket/fpocket2.tar.gz && \
    tar -xvzf fpocket2.tar.gz && rm fpocket2.tar.gz && cd fpocket2 && \
    sed -i 's/\$(LFLAGS) \$\^ -o \$@/\$\^ -o \$@ \$(LFLAGS)/g' makefile && make && \
    mv bin/fpocket bin/fpocket2 && mv bin/dpocket bin/dpocket2 && mv bin/mdpocket bin/mdpocket2 && mv bin/tpocket bin/tpocket2
ENV PATH=/app/fpocket2/bin:${PATH}
