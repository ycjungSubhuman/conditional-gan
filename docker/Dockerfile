FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Yucheol Jung <ycjung@postech.ac.kr>

COPY ./run_jupyter_tensorboard.sh run_jupyter_tensorboard.sh
RUN chmod +x run_jupyter_tensorboard.sh

RUN pip install scikit-optimize tensorboard Pillow
ENTRYPOINT ["./run_jupyter_tensorboard.sh"]

