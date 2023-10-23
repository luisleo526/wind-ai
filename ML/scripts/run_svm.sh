for c in 0.01 0.1 1 10 100
do
	for k in 0 1 2 3 4
	do
		echo "-------------------------------------------"
		echo "C=$c"
		./libsvm/svm-train -s 3 -t 2 -c $c -h 0 -q data/fold$k/train.scale libsvm_output/libsvm.model
		./libsvm/svm-predict data/fold$k/test.scale libsvm_output/libsvm.model "libsvm_output/${c}.${k}.pred"
	done
done
