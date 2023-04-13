The code was run in a env built by anaconda with packages:
Python==3.8, torch==1.12.1 (cuda), opencv-python==4.6.0.66, scipy==1.9.0, numpy==1.23.1, etc.

An example of running the code (figure 1 blue line):
python main_hr.py --num_users 50 --p 0.1 --inner_ep 20 \
--size 60000 --local_bs 256 \
--hlr 0.01 --lr 0.05 --outer_tau 1 \
--epoch 500 --round 10000000 --frac 1 \
--optim sgd --gpu 0

Note the complete list of parameters is defined in ./utils/options.py
