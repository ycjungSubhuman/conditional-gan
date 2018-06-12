#!/bin/bash

nvidia-docker run -it --rm \
	      -v $PWD/code/:/notebooks/code/ \
	      -v $PWD/dataset/:/notebooks/dataset/ \
	      -v $PWD/checkpoints/:/notebooks/checkpoints/ \
	      -v $PWD/summary/:/notebooks/summary/ \
	      -p 8888:8888 \
	      -p 8008:8008 \
	      -p 6006:6006 \
	      yucheol/tensorflow-auto \
	      "$@"
