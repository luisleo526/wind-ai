for c in 100 300 1000
do
	for g in 0.001 0.0003 0.0001
	do
		echo "-------------------------------------------"
		./thundersvm/build/bin/thundersvm-train -s 3 -c $c -g $g -v 5 wind.scale
	done
done
