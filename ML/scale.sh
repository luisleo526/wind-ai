rm -f data/all.txt
cat data/*.txt > data/all.txt
./libsvm/svm-scale -y 0 1 -s data/range data/all.txt > data/all.scale

for i in 0 1 2 3 4
do
	rm -rf data/fold$i
	mkdir -p data/fold$i
	find ./data -maxdepth 1 -iname '[0-9].txt' -not -name "${i}.txt" -exec cat {} +> "data/fold${i}/train.txt"
	cat data/$i.txt >> data/fold$i/test.txt
	./libsvm/svm-scale -y 0 1 -s data/range data/fold$i/train.txt > data/fold$i/train.scale
	./libsvm/svm-scale -y 0 1 -s data/range data/fold$i/test.txt > data/fold$i/test.scale
done