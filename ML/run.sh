#!/bin/bash

action=$1

if [ $action = "make" ]; then
	rm -rf data;
	python3 scripts/make_dataset.py
	bash scripts/scale.sh
elif [ $action = "svm" ]; then
	rm -rf libsvm_output
	mkdir libsvm_output
	nohup bash scripts/run_svm.sh &> libsvm_output/log &
elif [ $action = "xgboost" ]; then
	rm -rf xgboost_output
	mkdir -p xgboost_output
	nohup bash scripts/run_xgboost.sh &> xgboost_output/log &
else
	echo "Invalid option: $action, should be one of 'make' 'svm' 'xgboost' "
fi