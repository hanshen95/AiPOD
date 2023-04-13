import matplotlib.pyplot as plt
import yaml
import numpy as np

file_list = ["save/fedsgd_mnist_mlp_1000_C0.1_iidFalse_1642757617.yaml",
             "save/fedsgd_mnist_mlp_1000_C0.1_iidTrue_1642757619.yaml",
             "save/fedsvrg_mnist_mlp_1000_C0.1_iidFalse_1642757619.yaml",
             "save/fedsvrg_mnist_mlp_1000_C0.1_iidTrue_1642757614.yaml"]
name_list = ["sgd non iid", "sgd iid", "svrg non iid", "svrg iid"]

dict_list=[]
for file in file_list:
    f=open(file,mode='r')
    d=yaml.load(f,Loader=yaml.FullLoader)
    dict_list.append(d)

plt.cla()
for name,result in zip(name_list,dict_list):
    plt.plot(result["test_acc"])
plt.legend(name_list)
plt.ylim((60,100))
#plt.xlim((-5,300))
plt.grid('--')
plt.savefig('save/figs/test_acc.pdf')

plt.cla()
for name,result in zip(name_list,dict_list):
    plt.plot(result["test_loss"])
plt.legend(name_list)
plt.ylim((0,1))
#plt.xlim((-5,300))
plt.grid('--')
plt.savefig('save/figs/test_loss.pdf')
