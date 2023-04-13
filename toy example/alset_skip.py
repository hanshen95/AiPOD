import numpy as np
import scipy as sp
import yaml
import time
start_time = int(time.time())

# while True:
#     A = np.random.rand(90,100)
#     rank = np.linalg.matrix_rank(A)
#     if rank == A.shape[0] and rank < A.shape[1]:
#         print('A rank is', rank)
#         np.save('A', A)
#         break

# while True:
#     B = np.random.rand(80,100)
#     Z = np.zeros(B.shape)
#     B = np.concatenate([B,Z],axis=0)
#     rank = np.linalg.matrix_rank(B)
#     if rank < B.shape[0]:
#         print('B rank is', rank)
#         np.save('B', B)
#         break

# while True:
#     C = np.random.rand(100,100)
#     rank = np.linalg.matrix_rank(C)
#     if rank == C.shape[0] and rank == C.shape[1]:
#         print('C rank is', rank)
#         np.save('C', C)
#         break
    
# D = np.random.rand(100)
# np.save('D', D)
# E = np.random.rand(100)
# np.save('E', E)
# h = np.random.rand(90,100)
# np.save('h', h)



class Logger():
    def __init__(self, filename) -> None:
        self.filename=filename
        self.logs={"k": [], "norm": [], "ygap":[], "proj count":[]}
        self.counter=0
    def logging(self,k,count,ygap,norm):
        self.logs["k"].append(k)
        self.logs["proj count"].append(count)
        self.logs["ygap"].append(ygap)
        self.logs['norm'].append(norm)
        self.counter+=1
    def save(self):
        #if self.counter%100==0:
        f = open(self.filename, mode="w+")
        yaml.dump(self.logs, f)
        f.close()
        
def proj(x,A,b=None):
    I = np.eye(x.shape[0])
    A_ = np.linalg.pinv(A)
    if b is None:
        return np.matmul(I-A_ @ A,x)
    else:
        return np.matmul(I-A_ @ A,x)-A_@b

if __name__ == '__main__':
    A = np.load('A.npy')
    B = np.load('B.npy')
    C = np.load('C.npy')
    D = np.load('D.npy')
    E = np.load('E.npy')
    h = np.load('h.npy')
    A_ = np.linalg.pinv(A)
    
    K=500
    S=5
    T=2
    lr=0.1
    hlr=0.02
    p=1
    sig=0.1
    seed=1
    
    logs = Logger(f'./save/alsetskip_K(p=1){K}_S{S}_T{T}_lr{lr}_hlr{hlr}_p{p}_sig{sig}_seed{seed}_{start_time}.yaml') 
    rng = np.random.default_rng(seed)
    x = rng.random(100)
    x = proj(x,B)
    y =rng.random(100)
    y = proj(y,A,h@x)
    r = np.zeros(y.shape)
    Ix = np.eye(x.shape[0])
    # fix y_opt and delete this
    C=Ix
    Px=Ix-np.linalg.pinv(B)@B
    
    proj_count = 0
    
    # compute metric at first round
    y_opt = C@((Ix-np.linalg.pinv(A@C)@(A@C))@x-np.linalg.pinv(A@C)@(h@x))
    y_gap = np.square(np.linalg.norm(y-y_opt,2))
    v=sp.linalg.null_space(A)
    df_dy = np.cos(D@x+E@y)*(E.T)+1/(x.dot(x)+y.dot(y)+1) * y
    dg_dyy=np.eye(y.shape[0])
    temp = np.linalg.inv(v.T@dg_dyy@v)
    w = v@temp@(v.T)@df_dy
    df_dx=np.cos(D@x+E@y)*(D.T)+1/(x.dot(x)+y.dot(y)+1) * x
    dh = h.T
    dg_dxy=-C.T
    dg_dyy=np.eye(y.shape[0])
    hg = (dh @ (A_.T) @ dg_dyy - dg_dxy) @ w
    dF = df_dx+hg
    norm =np.square(np.linalg.norm(Px@dF,2))
    print('step',0,'proj',0,'norm {:.2f}'.format(norm),'y gap {:.2f}'.format(y_gap))
    logs.logging(0,0,y_gap.item(),norm.item())
    logs.save()
    
    Kp = np.min((int(K/p),4000))
    for k in range(Kp):
        for s in range(S):
            dg_dy=y-C@x
            n = rng.normal(0,sig,dg_dy.shape) 
            z=y-lr*(dg_dy+n-r)
            theta = rng.binomial(1,p)
            if theta==1:
                y = z-lr/p * r
                y = proj(y,A,h@x)
                r=r+p/lr * (y-z)
            else:
                y=z
            proj_count+=theta
        # this is problematic when C is not I
        y_opt = C@((Ix-np.linalg.pinv(A@C)@(A@C))@x-np.linalg.pinv(A@C)@(h@x))
        y_gap = np.square(np.linalg.norm(y-y_opt,2))
        
        v=sp.linalg.null_space(A)
        df_dy = np.cos(D@x+E@y)*(E.T)+1/(x.dot(x)+y.dot(y)+1) * y
        # df_dy+=rng.normal(0,sig,df_dy.shape)
        dg_dyy=np.eye(y.shape[0])
        # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
        temp = np.linalg.inv(v.T@dg_dyy@v)
        w = v@temp@(v.T)@df_dy
        for t in range(T):
            df_dx=np.cos(D@x+E@y)*(D.T)+1/(x.dot(x)+y.dot(y)+1) * x
            # df_dx+=rng.normal(0,sig,df_dx.shape)
            dh = h.T
            # dh+=rng.normal(0,sig,dh.shape)
            dg_dxy=-C.T
            # dg_dyy+=rng.normal(0,sig,dg_dxy.shape)
            dg_dyy=np.eye(y.shape[0])
            # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
            hg = (dh @ (A_.T) @ dg_dyy - dg_dxy) @ w
            dF = df_dx+hg
            n=rng.normal(0,sig,dg_dy.shape)
            x = x - hlr*(dF+n)
        x=proj(x,B)
        norm =np.square(np.linalg.norm(Px@dF,2))
        print('step',k+1,'proj',proj_count,'norm {:.2f}'.format(norm),'y gap {:.2f}'.format(y_gap))
        logs.logging(k+1,proj_count,y_gap.item(),norm.item())
        logs.save()
        
        