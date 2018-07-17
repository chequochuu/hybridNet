python3 -i hybridNet.py\
        --hidden 200 \
        --lu elu \
        --final_activation leaky \
        --batch_norm True\
        --n_res_block 1 \
        --n_fully 1 \
        --learning_rate 1e-3 \
        --init_xavier True\
        --batch_size 512 \
        --cost_func MSE \
        --reuse_weight False\
        --iter_load 10000 \

