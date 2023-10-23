TRF="train.scale?format=libsvm"
TSF="test.scale?format=libsvm"

for depth in 16 24 32 40 48
do  
    for k in 0 1 2 3 4
    do
        echo "Depth=${depth}, fold=${k}"
        DIR="data/fold${k}"
        xgboost xgboost.conf max_depth=$depth data=$DIR/$TRF eval[test]=$DIR/$TSF
    done
done