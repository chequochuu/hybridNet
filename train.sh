python3 -i hybridNet.py\
        --hidden 1000 \
        --lu leaky \
        --final_activation leaky \
        --batch_norm True\
        --n_res_block 3 \
        --n_fully 3 \
        --learning_rate 5e-4 \
        --init_xavier False\
        --batch_size 64 \
        --cost_func MSE \
        --reuse_weight False\
        --iter_load 10000 \

