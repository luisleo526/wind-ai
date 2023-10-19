for c in 0.01 0.1 1 10 100 1000
do
	echo "-------------------------------------------"
	echo "C=$c"
	./libsvm/svm-train -s 3 -t 2 -c $c -h 0 -q train.scale
	./libsvm/svm-predict test.scale train.scale.model output
done
