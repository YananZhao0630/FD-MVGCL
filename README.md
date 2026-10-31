# FD-MVGCL: Adaptive Multi-view Graph Contrastive Learning via Fractional Continuous Dynamics



This repository contains the code for our NeurIPS 2026 accepted paper, *Adaptive Multi-view Graph Contrastive Learning via Fractional Continuous Dynamics*.



## Table of Contents

- [Requirements](#requirements)
- [Framework](#framework-overview)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)



## Requirements

To install the required dependencies, refer to the `environment.yml` file.


## Framework overview

![Framework overview](images/framework.png)


## Reproducing Results

To reproduce the results in Table 2 and Table 3, run the following commands:





```bash

python GRACE_multi_pde_selected_path.py --dataset chameleon-filtered --add_source --hidden_dim 2048 --time 3 --lr 0.01 --function laplacian --block constant_alpha --runtime 1 --classifier_time 10 --classifier_epochs 500 --cuda 1 --method predictor --classifier_lr 0.01 --classifier_decay 0.001 --batch_norm --step_size 1 --num_encoders 5 --loss cosmean_plus --lambda 0.3 --epoch 2 --alpha_lr_2 0.05 --alpha_lr_3 0.1 --alpha_lr_4 0.1 --alpha_lr_5 0.001 --a1 0.5 --a2 0.03 --a3 0.01 --a4 0.01 --a5 0.45 --dropout 0.2 --seed 1469

python GRACE_multi_pde_selected_path.py --dataset actor --add_source --hidden_dim 2048 --time 1.5 --lr 0.01 --function laplacian --block constant_alpha --runtime 1 --classifier_time 10 --classifier_epochs 2000 --cuda 3 --method predictor --classifier_lr 0.01 --classifier_decay 0.00001 --batch_norm --step_size 0.15 --loss cosmean_plus --lambda 0.01 --epoch 5 --num_encoders 5 --dropout 0.01 --alpha_lr_1 0.01 --alpha_lr_2 0.015 --alpha_lr_3 0.01 --alpha_lr_4 0.02 --alpha_lr_5 0.001 --a1 0.1 --a2 0.1 --a3 0.05 --a4 0.05 --a5 0.7 --classifier_seed 15 --seed 10

python GRACE_multi_pde_selected_path.py --dataset squirrel-filtered --add_source --hidden_dim 4096 --time 3 --lr 0.01 --function laplacian --block constant_alpha --runtime 1 --classifier_time 10 --classifier_epochs 300 --cuda 0 --method predictor --classifier_lr 0.01 --classifier_decay 0.0015 --batch_norm --step_size 1 --num_encoders 5 --loss cosmean_plus --lambda 0.05 --epoch 3 --alpha_lr_2 0.05 --alpha_lr_3 0.025 --alpha_lr_4 0.12 --alpha_lr_5 0.005 --a1 0.465 --a2 0.005 --a3 0.015 --a4 0.015 --a5 0.5 --dropout 0.2 --seed 1422 --classifier_seed 45


```


More comments can be found in log_multi file. 



## Citation

If you find our helpful, consider to cite us:



```bibtex

@INPROCEEDINGS{Zhao:C26b,
    author = {Yanan Zhao and Feng Ji and Jingyang Dai and Jiaze Ma and Keyue Jiang and Kai Zhao and Wee Peng Tay},
    title = {Adaptive Multi-view Graph Contrastive Learning via Fractional Continuous Dynamics},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {},
    pages = {},
    month = {Dec.},
    year = {2026},
    address = {Sydney, Australia},
}

```

