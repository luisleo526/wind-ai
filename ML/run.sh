for c in 300 500 1000
do
	echo "c=$c"
	./libsvm/svm-train -s 3 -h 0 -c $c -q train.scale
	./libsvm/svm-predict test.scale train.scale.model output	
done
