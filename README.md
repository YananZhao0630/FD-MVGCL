# \# FD-MVGCL:	Adaptive Multi-view Graph Contrastive Learning via Fractional Continuous Dynamics



This repository contains the code for our NeurIPS 2026 accepted paper, \[\_Adaptive Multi-view Graph Contrastive Learning via Fractional Continuous Dynamics\_](#).



\## Table of Contents

\- \[Requirements](#requirements)

\- \[Reproducing Results](#reproducing-results)

\- \[Reference](#reference)

\- \[Citation](#citation)



\## Requirements

To install the required dependencies, refer to the `environment.yml` file.



\## Reproducing Results

To reproduce the results in Table 2 and Table 3, run the following commands:





```bash

python GRACE_multi_pde_selected_path.py --dataset chameleon-filtered --add_source --hidden_dim 2048 --time 3 --lr 0.01 --function laplacian --block constant_alpha --runtime 1 --classifier_time 10 --classifier_epochs 500 --cuda 1 --method predictor --classifier_lr 0.01 --classifier_decay 0.001 --batch_norm --step_size 1 --num_encoders 5 --loss cosmean_plus --lambda 0.3 --epoch 2 --alpha_lr_2 0.05 --alpha_lr_3 0.1 --alpha_lr_4 0.1 --alpha_lr_5 0.001 --a1 0.5 --a2 0.03 --a3 0.01 --a4 0.01 --a5 0.45 --dropout 0.2 --seed 1469



python ACM\_GCN\_heterphlic.py --dataset wisconsin --graph\_type new --prob\_lambda 0.15 --train\_epoch 20 --seed 2



python ACM\_GCN\_heterphlic.py --dataset squirrel --wd 5e-5 --graph\_type new --prob\_lambda 0.0001 --train\_epoch 60 --seed 1



python ACM\_GCN\_heterphlic.py --dataset cornell --graph\_type new --prob\_lambda 0.01 --train\_epoch 30 --seed 1



python ACM\_GCN\_heterphlic.py --dataset chameleon --wd 2e-5 --graph\_type new --prob\_lambda 0.08 --train\_epoch 30 --seed 10



python ACM\_GCN\_heterphlic.py --dataset actor --wd 1e-4 --graph\_type new --prob\_lambda 0.1 --seed 10 --train\_epoch 20



```


\## Reference



Our proposed simple tricks such as adding edges and normalization for node classification are based on ACM-GCN model, developed based on the following repo: https://github.com/SitaoLuan/ACM-GNN/tree/main



\## Citation

If you find our helpful, consider to cite us:



```bibtex

@inproceedings{JiZhaZha:C25,

&#x09;author = {Feng Ji and Yanan Zhao and Kai Zhao and Hanyang Meng and Jielong Yang and Wee Peng Tay},

&#x09;title={Rethinking Graph Neural Networks From A Geometric Perspective Of Node Features},

&#x09;booktitle={Proc. International Conference on Learning Representations},

&#x09;month = {Apr.},

&#x09;year={2025},

&#x09;address = {Singapore},

}

```

