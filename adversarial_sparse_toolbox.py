import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist

import pdb


class SSC_Model_separable(nn.Module):
    def __init__(self,n,m,k,N,lambd,margin):
        super(SSC_Model_separable, self).__init__()
        
        (D,mu) = init_dictionary(n,m)
        self.D = D
        self.mu = mu
        w = torch.randn(m)
        self.w = w/torch.norm(w)
        self.lambd = lambd
        self.OPT = 1
        self.T = 1000
        self.margin = margin
        self.k = k
        
        # step_size
        self.eta = 1/(np.linalg.norm(self.D.numpy(),2)**2)        
        
        # generating data
        (x,labels) = generate_data_separable(self,N)
        self.x = x
        self.labels = labels
        
        # self-check
        if checkMargin(self) > self.margin:
            print('Margin verified and satisfied.')
        else:
            print('Margin NOT satisfied - something went wrong.')

            
    def forward(self,x):
        
        (N,n) = x.shape
        z = encode(self,x)
        y = torch.matmul(self.w,z)
        pred = torch.cat((y.view(N,1) , - y.view(N,1)),dim=1)
        
        return pred

# -------------------------------------------------------------------------------
    
class SSC_Model_NonSep(nn.Module):
    def __init__(self,n,m,k,N,lambd,margin):
        super(SSC_Model_NonSep, self).__init__()
        
        (D,mu) = init_dictionary(n,m)
        self.D = D
        self.mu = mu
        w = torch.randn(m)
        self.w = w/torch.norm(w)
        self.lambd = lambd
        self.OPT = 1
        self.T = 1000
        self.margin = margin
        self.k = k
        
        # step_size
        self.eta = 1/(np.linalg.norm(self.D.numpy(),2)**2)        
        
        # generating data
        (x,labels) = generate_data_nonsep(self,N)
        self.x = x
        self.labels = labels
        
        # self-check
        if checkMargin(self) > self.margin:
            print('Margin verified and satisfied.')
        else:
            print('Margin NOT satisfied - something went wrong.')

            
    def forward(self,x):
        
        (N,n) = x.shape
        z = encode(self,x)
        y = torch.matmul(self.w,z)
        pred = torch.cat((y.view(N,1) , - y.view(N,1)),dim=1)
        
        return pred
    
#-------------------------------------------------------------------------------


class SSC_Model(nn.Module):
    def __init__(self,n,m,k,N,lambd,margin):
        super(SSC_Model, self).__init__()
        
        (D,mu) = init_dictionary(n,m)
        self.D = nn.Parameter(D,requires_grad=False)
        self.mu = mu
        w = torch.randn(m)
        self.w = nn.Parameter(w/torch.norm(w),requires_grad=False)
        self.lambd = lambd
        self.OPT = 1
        self.T = 1000
        self.margin = margin
        self.k = k
        
        # step_size
        self.eta = 1/(np.linalg.norm(self.D.numpy(),2)**2)        
        
        # generating data
        (x,labels) = generate_data(self,N)
        self.x = x
        self.labels = labels

            
    def forward(self,x):
        
        (N,n) = x.shape
        z = encode(self,x)
        y = torch.matmul(self.w,z)
        pred = torch.cat((y.view(N,1) , - y.view(N,1)),dim=1)
        
        return pred

# -------------------------------------------------------------------------------


# class SSC_Model_learn(nn.Module):
#     def __init__(self,n,m,k,N,lambd,margin):
#         super(SSC_Model, self).__init__()
        
#         (D,mu) = init_dictionary(n,m)
#         self.D = nn.Linear(n,m, bias = False)
#         self.D.weight.data = D.t()
#         self.mu = mu
        
#         self.w = nn.Linear(m,10)
        
#         self.w.weight.data = w/torch.norm(w)
#         self.lambd = lambd
#         self.OPT = 1
#         self.T = 1000
#         self.margin = margin
#         self.k = k
        
#         # step_size
#         self.eta = 1/(np.linalg.norm(self.D.numpy(),2)**2)        
            
#     def forward(self,x):
        
#         (N,n) = x.shape
#         z = encode_data(self,x)
#         y = torch.matmul(self.w,z)
#         pred = torch.cat((y.view(N,1) , - y.view(N,1)),dim=1)
        
#         return pred

    

# -------------------------------------------------------------------------------


def generate_data_separable(model,N):
        
    # sampling data in two rounds, in order to guarantee enough data with given margin
    N0 = N
    Gmin = 1
    Gmax = 2
    (n,m) = model.D.shape

    for _ in range(2):    
        Gamma = np.zeros((m,N))
        for i in range(N):
            S = np.random.permutation(m)[0:model.k]
            Gamma[S,i] = np.ones(model.k)*Gmin + (Gmax-Gmin)*np.random.rand(model.k)
        Signs = np.random.binomial(1,.5,(m,N)) * 2 - 1
        Gamma = torch.from_numpy(Gamma * Signs).float()
            
        # generating signals
        x = torch.matmul(model.D,Gamma)
        x = torch.Tensor(normalize(x.numpy(),axis=0))
        # enforcing margin
        z = encode(model,x.t())
        proj = torch.matmul(model.w,z) 
        ind = (torch.abs(proj)>model.margin)
        N_pass = torch.sum(ind)
        
        
        if N_pass == 0 :
            print('Margin specified too large - decrease marging or modify penalty parameter')
            return('')
        else: # accomodate N for next round.
            N = N**2/N_pass * 2

    if N_pass>=N0: # if enough samples, keep
        x, proj = x[:,ind], proj[ind]
        labels = torch.sign(proj)
        x = x[:,0:N0]
        labels = labels[0:N0]
    else:
        print("Not enough samples / margin too large")
        return ('')

    
    return (x.t(),labels)
        

# -------------------------------------------------------------------------------


def generate_data_nonsep(model,N):
        
    # sampling data in two rounds, in order to guarantee enough data with given margin
    N0 = N
    Gmin = 1
    Gmax = 2
    (n,m) = model.D.shape

    for _ in range(2):    
        Gamma = np.zeros((m,N))
        for i in range(N):
            S = np.random.permutation(m)[0:model.k]
            Gamma[S,i] = np.ones(model.k)*Gmin + (Gmax-Gmin)*np.random.rand(model.k)
        Signs = np.random.binomial(1,.5,(m,N)) * 2 - 1
        Gamma = torch.from_numpy(Gamma * Signs).float()
            
        # generating signals
        x = torch.matmul(model.D,Gamma)
        x = torch.Tensor(normalize(x.numpy(),axis=0))
        # enforcing margin
#         z = encode(model,x.t())
#         proj = torch.matmul(model.w,z)
        Gamma_ = Gamma + .1*torch.randn((m,N))
        proj = torch.matmul(model.w,Gamma_)
        ind = (torch.abs(proj)>model.margin)
        N_pass = torch.sum(ind)
            
        if N_pass == 0 :
            print('Margin specified too large - decrease marging or modify penalty parameter')
            return('')
        else: # accomodate N for next round.
            N = N**2/N_pass * 2

    if N_pass>=N0: # if enough samples, keep
        x, proj = x[:,ind], proj[ind]
        labels = torch.sign(proj)
        x = x[:,0:N0]
        labels = labels[0:N0]
    else:
        print("Not enough samples / margin too large")
        return ('')

    
    return (x.t(),labels)
        

# -------------------------------------------------------------------------------



def generate_data(model,N):
        
    # sampling data in two rounds, in order to guarantee enough data with given margin
    N0 = N
    Gmin = 1
    Gmax = 2
    (n,m) = model.D.shape

    for _ in range(2):    
        Gamma = np.zeros((m,N))
        for i in range(N):
            S = np.random.permutation(m)[0:model.k]
            Gamma[S,i] = np.ones(model.k)*Gmin + (Gmax-Gmin)*np.random.rand(model.k)
        Signs = np.random.binomial(1,.5,(m,N)) * 2 - 1
        Gamma = torch.from_numpy(Gamma * Signs).float()

        # enforcing some margin
        proj = torch.matmul(model.w,Gamma) 
        ind = (torch.abs(proj)>model.margin)
        N_pass = torch.sum(ind)
        
        if N_pass == 0 :
            print('Margin specified too large - decrease marging or modify penalty parameter')
            return('')
        else: # accomodate N for next round.
            N = N**2/N_pass * 2

        
    if N_pass>=N0: # if enough samples, keep        
        # generating signals
        x = torch.matmul(model.D,Gamma)
        x = torch.Tensor(normalize(x.numpy(),axis=0))
        x, proj = x[:,ind], proj[ind]
        labels = torch.sign(proj)
        x = x[:,0:N0]
        labels = labels[0:N0]
    else:
        print("Not enough samples / margin too large")
        return ('')
        
    
    return (x.t(),labels)

# -------------------------------------------------------------------------------

def generate_data_Gaussian(model,N):
        
    # sampling data in two rounds, in order to guarantee enough data with given margin
    N0 = N
    std = 1
    (n,m) = model.D.shape

    for _ in range(2):    
        Gamma = np.zeros((m,N))
        for i in range(N):
            S = np.random.permutation(m)[0:model.k]
            Gamma[S,i] = std * np.random.randn(model.k)
        Gamma = torch.from_numpy(Gamma).float()

        # enforcing some margin
        proj = torch.matmul(model.w,Gamma) 
        ind = (torch.abs(proj)>model.margin)
        N_pass = torch.sum(ind)
        
        if N_pass == 0 :
            print('Margin specified too large - decrease marging or modify penalty parameter')
            return('')
        else: # accomodate N for next round.
            N = N**2/N_pass * 2

        
    if N_pass>=N0: # if enough samples, keep        
        # generating signals
        x = torch.matmul(model.D,Gamma)
        x = torch.Tensor(normalize(x.numpy(),axis=0))
        x, proj = x[:,ind], proj[ind]
        labels = torch.sign(proj)
        x = x[:,0:N0]
        labels = labels[0:N0]
    else:
        print("Not enough samples / margin too large")
        return ('')
        
    
    return (x.t(),labels)


# -------------------------------------------------------------------------------
       
    
def encode(model,x):                
    n = x.shape[1]
    N = x.shape[0]
    x = x.t()
        
    tol = 1e-10
    D = model.D.detach()
    lambd_eta = lambd=model.eta*model.lambd
    Dtx = torch.matmul(D.t(),x).detach()
    z_ = F.softshrink( model.eta * Dtx , lambd_eta)
    for i in range(model.T):
        res =  torch.matmul(D,z_) - x
        Dtres = torch.matmul(D.t(), res ).detach()
        z = F.softshrink( z_ - model.eta * Dtres,lambd=model.lambd*model.eta)
        if torch.norm(z - z_)<tol:
            break
        else: 
            z_ = z
        
    # computing closed form solution given sign pattern
    alfa = z
    SIGNS = torch.sign(z)
    for i in range(N):
        S = (z[:,i]!=0)
        if np.sum(S.numpy()) == 0:
            break
        else:
            Gs = torch.matmul(model.D[:,S].t() , model.D[:,S] )
            b = (torch.matmul(model.D[:,S].t() , x[:,i]) - model.lambd * SIGNS[S,i] )
            alfa[S,i] = torch.matmul(torch.inverse(Gs),b)
        
    return alfa
    

# -------------------------------------------------------------------------------

def checkMargin(model):
    z = encode(model,model.x)
    scores = torch.matmul(model.w,z)
    margin = torch.min(torch.abs(scores)).numpy()
    return margin

# -------------------------------------------------------------------------------

def getSoftMargin(model):
    z = encode(model,model.x)
    scores = torch.matmul(model.w,z)
    N = len(scores)
    y = model.labels
    
    scoresf = scores
    scoresf[y<0] = -1 * scoresf[y<0]
    smax = torch.max(scoresf)
    
    thres = np.linspace(0,smax,50)
    Accs = np.zeros(50)
    for i in range(50):
        Accs[i] = np.sum(scoresf.numpy()>= thres[i])/N
        
    return Accs,thres

# -------------------------------------------------------------------------------

def init_dictionary(n,m):
    
    # computes an approximate Grasmanian frame
    
    # Dictionary Initialization 
    D = torch.randn(n,m)
    D = D/torch.norm(D,dim=0)
    Gram = torch.matmul(D.t(),D) 
    mu = np.sqrt((m-n)/n/(m-1))

    dd1 = 0.8
    dd2 = 0.9
    Iter = 200
    Res = torch.zeros((Iter,3))
    for k in range(Iter):
        
        # shrink the high inner products
        ind = int(np.round(dd1*(m**2-m)))
        thresh = torch.Tensor([np.partition(Gram.abs().flatten(),ind)[ind]])
        ind_G = (torch.abs(Gram)>thresh) & (torch.abs(Gram-1)>1E-6)
        Gram[ind_G] = dd2 * Gram[ind_G]

        # reduce the rank back to N
        u, s, v = torch.svd(Gram)
        s[n:] = 0
        Gram = torch.mm(u, torch.mm(s.diag(), v.t()))

        # Normalize the columns
        eng = Gram.diag().sqrt() 
        G_tmp = Gram/eng
        Gram = G_tmp.t()/eng

        # Show status
        thresh = torch.Tensor([np.partition(Gram.abs().flatten(),ind)[ind]])
        ind_G = (torch.abs(Gram)>thresh) & (torch.abs(Gram-1)>1E-6)
        # Res[k,:]=torch.tensor((mu,torch.mean(torch.abs(Gram[ind_G])),torch.max(torch.abs(Gram[ind_G]))))
                 
    thresh = torch.Tensor([np.partition(Gram.abs().flatten(),ind)[ind]])
    ind_G = (torch.abs(Gram)>thresh)& (torch.abs(Gram-1)>1E-6)
    mu_G = torch.max(torch.abs(Gram-torch.eye(m)))
    print('The emperical mu is %.5f, while the theoretical bound is %.5f.'%(mu_G, np.sqrt((m-n)/n/(m-1))))
    
    u,s,v = torch.svd(Gram)
    D = torch.mm(s[:n].diag().sqrt(), v[:,:n].t())

    return(D,mu_G)
    


# -------------------------------------------------------------------------------

def get_encoderGap(model,x,k):
    
    z = encode(model,x)
    x = x.t()
    proj = model.lambd - torch.abs(torch.matmul(model.D.t(),x - torch.matmul(model.D,z) ))
    proj_sorted, indices = torch.sort(proj,dim=0)
    proj_mins,_ = torch.min(proj_sorted,dim=1)
    
    return proj_mins[k],proj_mins


# -------------------------------------------------------------------------------


def get_babel(model):
    
    D = model.D/torch.norm(model.D,dim=0).detach()
    Gram = torch.matmul(D.t(),D)
    (GS,_) = torch.sort(torch.abs(Gram),dim=0)
    mus = torch.zeros(len(GS)-1)
    for i in range(2,len(GS)+1):
        mus[i-2] = torch.max(torch.sum(GS[-i:-1,:],dim=0))
    
    return mus