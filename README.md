## Introduction

This repo includes an implementation of the AiPOD and E-AiPOD algorithm presented in the paper
 [_Alternating Implicit Projected SGD and Its Efficient Variants for Equality-constrained Bilevel Optimization_](https://arxiv.org/abs/2211.07096).
 
 The algorithms solve the _bilevel optimization problem_ with equality constraint.
 The bilevel (optimization) problem enjoys a wide range of applications; e.g., federated learning, meta-learning, image processing, hyper-parameter optimization, and reinforcement learning.


### Dependencies

The combination below works for us.
- Python = 3.8
- [torch = 1.12.1](https://pytorch.org/get-started/locally/)
- [opencv-python=4.6.0.66](https://pypi.org/project/opencv-python/)
- [scipy=1.9.0](https://pypi.org/project/scipy/)
- [yaml = 6.0](https://pypi.org/project/PyYAML/)
- cuda = 11.3

## Running the code

### Toy problem
The problem is described in the 'synthetic experiments' section of the [paper](https://arxiv.org/abs/2211.07096).

To run this experiment, navigate to `./toy example/` and run in console:

`python alset_skip.py`


### Federated representation learning
The problem is described in the 'Federated representation learning' section of the [paper](https://arxiv.org/abs/2211.07096).

To run this experiment (recover figure 1 blue curve), navigate to `./hyper-representation/` and run in console:
```
python main_hr.py --num_users 50 --p 0.1 --inner_ep 20 \
--size 60000 --local_bs 256 \
--hlr 0.01 --lr 0.05 --outer_tau 1 \
--epoch 500 --round 10000000 --frac 1 \
--optim sgd --gpu 0
```

### Federated learning from imbalanced data
The problem is described in the 'Federated learning from imbalanced data' section of the [paper](https://arxiv.org/abs/2211.07096).

To run this experiment (recover figure 2 blue curve), navigate to `./imbalance/` and run in console:
```
python main_imbalance.py --num_users 50 --p 0.3 --inner_ep 20 \
--size 24000 --local_bs 256 --neumann 3 \
--hlr 0.01 --lr 0.04 --outer_tau 1 \
--epoch 3000 --round 4000 --frac 1 \
--optim sgd --gpu 0
```



## Citation

If you find this repo helpful, please cite the [paper](https://arxiv.org/abs/2211.07096).

```latex
@article{xiao2022alternating,
  title={Alternating Implicit Projected SGD and Its Efficient Variants for Equality-constrained Bilevel Optimization},
  author={Xiao, Quan and Shen, Han and Yin, Wotao and Chen, Tianyi},
  journal={arXiv preprint arXiv:2211.07096},
  year={2022}
}
```
