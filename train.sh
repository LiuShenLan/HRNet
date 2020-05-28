#！/bin/bash

# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/home/lzh/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/home/lzh/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lzh/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/home/lzh/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<

conda activate hrnet

export PYTHONPATH=~/anaconda3/envs/hrnet/lib/python3.6/site-packages/

nohup CUDA_VISIBLE_DEVICES=0,1 python tools/dist_train.py --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml  > print.log 2>&1