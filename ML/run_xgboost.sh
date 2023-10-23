for depth in 16 24 32 40
do  
    sed -i -e "6 c max_depth=${depth}" ./xgboost.conf
    for bin in 64 128 192 256
    do
        echo "max_depth=${depth}, max_bin=${bin}"
        sed -i -e "8 c max_bin=${bin}" ./xgboost.conf
        xgboost xgboost.conf
    done
done

