#!/bin/bash

(tensorboard --logdir=summary/ &) && jupyter notebook --allow-root

