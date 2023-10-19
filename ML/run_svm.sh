executable=./libsvm/svm-train

for c in 0.1 1 10 100 1000
do
	echo "-------------------------------------------"
	$executable -s 3 -t 2 -c $c -v 5 -e 0.0001 -h 0 wind.scale
done
