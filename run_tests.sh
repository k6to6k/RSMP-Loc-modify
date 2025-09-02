#!/bin/bash
echo "运行上下文感知动态原型模块单元测试"
CUDA_VISIBLE_DEVICES=0 python -W ignore ./test_dynamic_prototype.py

