python3 -i hybridNet.py\
        --description prioritied \
        --hidden 1500 \
        --lu leaky \
        --final_activation leaky \
        --batch_norm True\
        --n_res_block 0 \
        --n_fully 0 \
        --learning_rate 1e-4 \
        --init_xavier True\
        --batch_size 64 \
        --cost_func MSE \
        --reuse_weight False\
        --iter_load 10000 \

