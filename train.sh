python -i hybridNet.py\
        --hidden 1000 \
        --lu leaky \
        --final_activation leaky \
        --batch_norm True\
        --n_res_block 2
        --n_fully 2
        --learning_rate 5e-4 \
        --init_xavier True\
        --batch_size 64 \
        --cost_func BCE \
        --reuse_weight False\
        --iter_load 10000 \

