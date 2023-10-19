./libsvm/svm-scale -y 0 1 -s range wind.dat > wind.scale
./libsvm/svm-scale -r range train.dat > train.scale
./libsvm/svm-scale -r range test.dat > test.scale
