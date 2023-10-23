TRF="train.scale?format=libsvm"
TSF="test.scale?format=libsvm"

rm -rf xgboost_output
mkdir -p xgboost_output

for depth in 16 24 32 40 48
do  
    for min_child_weight in 5 10 30 50
    do
        for k in 0 1 2 3 4
        do
            echo "Depth=${depth}, min_child_weight=${min_child_weight}, fold=${k}"
            DIR="data/fold${k}"
            ARGS="max_depth=${depth} min_child_weight=${min_child_weight}"
            xgboost xgboost.conf $ARGS data=$DIR/$TRF eval[test]=$DIR/$TSF model_out=xgboost.model
            xgboost xgboost.conf $ARGS task=pred test:data=$DIR/$TSF name_pred="xgboost_output/${depth}.${min_child_weight}.${k}.pred" model_in=xgboost.model
        done
    done
done