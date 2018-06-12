#!/bin/bash

cd ./docker/
nvidia-docker build -t yucheol/tensorflow-auto:latest -t yucheol/tensorflow-auto:0.0.0 .
