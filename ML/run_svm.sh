for c in 0.01 0.1 1 10 100
do
	echo "-------------------------------------------"
	echo "C=$c"
	./libsvm/svm-train -s 3 -t 2 -c $c -h 0 -v 5 data/all.scale
done
