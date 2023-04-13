The code was run in a env built by anaconda with packages:
Python==3.8, torch==1.12.1 (cuda), opencv-python==4.6.0.66, scipy==1.9.0, numpy==1.23.1, etc.

An example of running the code (figure 2 blue line):
python main_imbalance.py --num_users 50 --p 0.3 --inner_ep 20 \
--size 24000 --local_bs 256 --neumann 3 \
--hlr 0.01 --lr 0.04 --outer_tau 1 \
--epoch 3000 --round 4000 --frac 1 \
--optim sgd --gpu 0

Note the complete list of parameters is defined in ./utils/options.py
