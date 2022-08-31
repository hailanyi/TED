#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3,4 nohup python3 -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch > log-test.txt &

