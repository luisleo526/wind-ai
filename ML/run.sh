for c in 1 10 100 300
do
	echo "c=$c"
	./libsvm/svm-train -s 3 -h 0 -c $c -v 5 wind.scale 
done
