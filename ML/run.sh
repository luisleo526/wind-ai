for c in 1 10 100 1000
do
	for g in 0.001 0.01 0.1 1
	do
		echo "-------------------------------------------"
		./thundersvm/build/bin/thundersvm-train -s 3 -c $c -g $g -v 5 wind.scale
	done
done
