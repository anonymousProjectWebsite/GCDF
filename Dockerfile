# build a image for developing in ubuntu 20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV TZ=America/New_York
ARG DEBIAN_FRONTEND=noninteractive
# install some necessary tools
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    git \
    curl \
    wget \
    lsb-release \
    sudo \
    pybind11-dev \
    gnupg2 \
    build-essential \
    libglpk-dev \
    libglvnd0\
    mesa-utils \
    cmake \
    libeigen3-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    terminator \
    locales && \
    locale-gen en_US.UTF-8\
    && rm -rf /var/lib/apt/lists/*

# ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all    
ENV NVIDIA_DRIVER_CAPABILITIES=all


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -


RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN rosdep init && rosdep update


ENV ROS_DISTRO=noetic
ENV ROS_PACKAGE_PATH=/opt/ros/$ROS_DISTRO/share
ENV PATH=/opt/ros/$ROS_DISTRO/bin:$PATH
ENV PYTHONPATH=/opt/ros/$ROS_DISTRO/lib/python3/dist-packages:{$PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/ros/$ROS_DISTRO/lib:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all    
ENV NVIDIA_DRIVER_CAPABILITIES=all

ARG user=robot
ARG group=robot
ARG uid=1000
ARG gid=1000
ARG home=/home/${user}
RUN mkdir -p /etc/sudoers.d \
    && groupadd -g ${gid} ${group} \
    && useradd -d ${home} -u ${uid} -g ${gid} -m -s /bin/bash ${user} \
    && echo "${user} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudoers_${user}
USER ${user}
RUN sudo usermod -a -G video ${user}
WORKDIR ${home}
# RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" > /home/${user}/.bashrc \
#     && chown ${user}:${group} /home/${user}/.bashrc
RUN git clone https://github.com/coin-or/CppAD.git \
    && cd CppAD \
    && mkdir build && cd build \
    && cmake .. \
    && sudo make install \
    && cd /home/robot

RUN git clone https://github.com/joaoleal/CppADCodeGen.git \
    && cd CppADCodeGen \
    && mkdir build \
    && cd build \
    && cmake .. \
    && sudo make install \
    && cd /home/robot

# RUN git clone https://github.com/PREDICT-EPFL/piqp.git \
#     && git checkout v0.5.0 \
#     && cd piqp \
#     && mkdir build \
#     && cd build \
#     && cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
#     && cmake --build . --config Release \
#     && cmake --install . --config Release \
#     && cd /home/robot





# RUN git clone https://github.com/PREDICT-EPFL/piqp.git \
#     && cd piqp \
#     && git checkout v0.5.0 \
#     && mkdir build \
#     && cd build \
#     && cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
#     && cmake --build . --config Release \
#     && sudo cmake --install . --config Release
# and open shell
# CMD ["/bin/bash"]
CMD ["terminator"]



