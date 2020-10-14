# Simple pytorch implementation of Dictionary Learning based on stochastic gradient descent
#
# June 2018
# Jeremias Sulam


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import pdb


####################################
    ## Dict. Learning ##
####################################

# UNSUPERVISED MODEL

class DictLearn(nn.Module):
    def __init__(self,m, SC, lambd, Fista_Iter = 50):
        super(DictLearn, self).__init__()

        self.SC = SC
        self.lambd = lambd
        self.FistaIter = Fista_Iter
        self.W = nn.Parameter(torch.randn(28*28, m, requires_grad=False))
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
    def forward(self, Y):
        
        self.W.requires_grad_(False)
        # Sparse Coding
        if self.SC == 'IHT':
            Gamma,residual, errIHT = IHT(Y,self.W,self.lambd)
        elif self.SC == 'fista':
            Gamma,residual, errIHT = FISTA(Y,self.W,self.lambd,self.FistaIter)
        else: print("Oops!")
                
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(Gamma,self.W.transpose(1,0))
        
        # sparsity
        NNZ = np.count_nonzero(Gamma.cpu().data.numpy())/Gamma.shape[0]
#         print(NNZ)
        return X, Gamma, errIHT
        
        
    def normalize(self):
        self.W.data = NormDict(self.W.data)
        

# ------------------------------------------------
        
class DictLearnCIFAR(nn.Module):
    def __init__(self,m, SC, lambd, Fista_Iter = 50):
        super(DictLearnCIFAR, self).__init__()

        self.SC = SC
        self.lambd = lambd
        self.FistaIter = Fista_Iter
        self.W = nn.Parameter(torch.randn(32*32*3, m, requires_grad=False))
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
    def forward(self, Y):
        
        self.W.requires_grad_(False)
        # Sparse Coding
        if self.SC == 'IHT':
            Gamma,residual, errIHT = IHT(Y,self.W,self.lambd)
        elif self.SC == 'fista':
            Gamma,residual, errIHT = FISTA(Y,self.W,self.lambd,self.FistaIter)
        else: print("Oops!")
                
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(Gamma,self.W.transpose(1,0))
        
        # sparsity
        NNZ = np.count_nonzero(Gamma.cpu().data.numpy())/Gamma.shape[0]
#         print(NNZ)
        return X, Gamma, errIHT
        
        
    def normalize(self):
        self.W.data = NormDict(self.W.data)
        


# SUPERVISED LEARNING CLASS

class DictLearn_supervised(nn.Module):
    def __init__(self,m,SC,Fiter = 20):
        super(DictLearn_supervised, self).__init__()
        
        self.SC = SC
        self.FistaIter = Fiter
        self.lambd = nn.Parameter(torch.zeros(1,1))
        self.lambd.requires_grad_(False)
        self.W = nn.Parameter(torch.randn(28*28, m, requires_grad=True))
        self.Wclass = nn.Linear(m, 10)
        self.learn_flag = True
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
        
    def init_dictionary(self,W):
        
        device = self.W.device
        self.W.requires_grad_(False)
        self.W.data = NormDict(W.data).to(device)
        self.W.requires_grad_(True)
    
    
    def forward(self, Y):
        
        if self.learn_flag == True:
            # Sparse Coding
            if self.SC == 'IHT':
                Gamma,residual, errIHT = IHT(Y,self.W,self.lambd)
            elif self.SC == 'fista':
                Gamma,residual, errIHT = FISTA(Y,self.W,self.lambd,self.FistaIter)
            else: print("Oops!")
            
        else: # if not in training, perform final LS step for exact LASSO solution
            Gamma = encode(self,Y)           
            
        self.last_avg_nnz = (np.count_nonzero(Gamma.cpu().detach().numpy())/Y.shape[0])
        
        # Classification
        out = self.Wclass(Gamma)
        out = F.log_softmax(out,dim=1)
        
        return out
        
        
        # normalizing Dict
    def normalize(self):
        self.W.requires_grad_(False)
        self.W.data = NormDict(self.W.data)
        self.W.requires_grad_(True)
        
## ------------------------------------------------------


class DictLearn_supervised_CIFAR(nn.Module):
    def __init__(self,m,SC,Fiter = 20):
        super(DictLearn_supervised_CIFAR, self).__init__()
        
        self.SC = SC
        self.accuracy_test = 0
        self.FistaIter = Fiter
        self.lambd = nn.Parameter(torch.zeros(1,1))
        self.lambd.requires_grad_(False)
        self.W = nn.Parameter(torch.randn(32*32*3, m, requires_grad=True))
        self.Wclass = nn.Linear(m, 10)
        self.learn_flag = True
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
        
    def init_dictionary(self,W):
        
        device = self.W.device
        self.W.requires_grad_(False)
        self.W.data = NormDict(W.data).to(device)
        self.W.requires_grad_(True)
    
    def only_linear_training_on(self):
        self.W.requires_grad = False
        self.W.requires_grad_(False)
        print('\nWARNING: only training linear classifier on representations\n')
        
    def only_linear_training_off(self):
        self.W.requires_grad = True
        print('\nNormal end-to-end training set up\n')
        
    
    def forward(self, Y):
        
        if self.learn_flag == True: # if yes, simple do fista for approximate LASSO solution
            # Sparse Coding
            if self.SC == 'IHT':
                Gamma,residual, errIHT = IHT(Y,self.W,self.lambd)
            elif self.SC == 'fista':
                Gamma,residual, errIHT = FISTA(Y,self.W,self.lambd,self.FistaIter)
            else: print("Oops!")
            
        else: # if not in training, perform final LS step for exact LASSO solution
            Gamma = encode(self,Y)           
            
        self.last_avg_nnz = (np.count_nonzero(Gamma.cpu().detach().numpy())/Y.shape[0])
        
        # Classification
        out = self.Wclass(Gamma)
        out = F.log_softmax(out,dim=1)
        
        return out
        
                
        # normalizing Dict
    def normalize(self):
        
        if self.W.requires_grad == True: # if training end-to-end
            self.W.requires_grad_(False)
            self.W.data = NormDict(self.W.data)
            self.W.requires_grad_(True)      
        
        
        
#--------------------------------------------------------------
#         Auxiliary Functions
#--------------------------------------------------------------


def encode(model,Y):    
    
    # Sparse Coding
    if model.SC == 'IHT':
        Gamma,residual, errIHT = IHT(Y,model.W,model.lambd)
    elif model.SC == 'fista':
        Gamma,residual, errIHT = FISTA(Y,model.W,model.lambd,model.FistaIter)
    else: print("Oops!")
    
    N = Y.shape[0]
    D = model.W.data
    # computing closed form solutions given support
    alfa = Gamma.detach()
    SIGNS = torch.sign(Gamma.detach())
    for i in range(N):
        S = (alfa[i,:]!=0)
        if torch.sum(S) == 0:
            break
        else:
            Gs = torch.matmul(D[:,S].t() , D[:,S] )
            b = (torch.matmul(D[:,S].t() , Y[i,:]) - model.lambd * SIGNS[i,S] )
            alfa[i,S] = torch.matmul(torch.inverse(Gs),b)
        
    return Gamma
    
#--------------------------------------------------------------

def getSoftMargin(model,x,y):
    z = encode(model,x)
    out = model.Wclass(z)
    scores = 10**F.log_softmax(out,dim=1)
    N = len(scores)
    
    # compute accuracy as function of margin
    bins = 100
    max_out = np.max(out.cpu().detach().numpy())
    Mrgs = np.linspace(0,max_out,num=bins)
    sample_margin = np.zeros(N)
    Accs = np.zeros(bins)
    
    for j in range(bins):
        mrgs = np.zeros(N)
        for i in range(N):
            si = out[i,:].detach()
            mrgs[i] = si[y[i]] - torch.max( si[torch.arange(len(si)) != y[i]] )  # margin of the ith sample
    
        Accs[j] = np.sum(mrgs>Mrgs[j])/N
        
    return Accs, Mrgs

#--------------------------------------------------------------


def get_encoderGap(model,x,s):
    
    z = encode(model,x).detach().t()
    proj = model.lambd - torch.abs(torch.matmul(model.W.t(), x.t() - torch.matmul(model.W,z))).detach()
    proj_sorted, indices = torch.sort(proj,dim=0)
    proj_mins,_ = torch.min(proj_sorted,dim=1)
    
    return proj_mins[s],proj_mins.detach()

#--------------------------------------------------------------


def get_ClassifierConstant(model):
    
    W = model.Wclass.weight
    # computing pairwise distances
    NC = W.shape[0]   # number of classes
    
    PD = torch.zeros(NC,NC)
    for i in range(NC):
        for j in range(NC):
            PD[i,j] = torch.norm(W[i,:]-W[j,:])
    C = torch.max(PD).cpu().detach().numpy()
    return C



#--------------------------------------------------------------

def get_mu(model):
    
    W = model.W.detach()
    W.data = NormDict(W.data)
    Gram = torch.matmul(W.t(),W).cpu().numpy()
    mu_G = np.max(np.abs(Gram)-np.eye(Gram.shape[0]))
    
    return mu_G

#--------------------------------------------------------------

def get_babel(model):
    
    model.W.data = NormDict(model.W.data)
    Gram = torch.matmul(model.W.t(),model.W).detach()
    (GS,_) = torch.sort(torch.abs(Gram),dim=0)
    mus = torch.zeros(len(GS)-1)
    for i in range(2,len(GS)+1):
        mus[i-2] = torch.max(torch.sum(GS[-i:-1,:],dim=0))
    
    return mus
    

#--------------------------------------------------------------


def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(X.device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(X.device)
    Gamma = Gamma * mask
    return Gamma

#--------------------------------------------------------------


def soft_threshold(X, lamda):
#     pdb.set_trace()
#     Gamma = X.clone()
    Gamma = torch.sign(X) * F.relu(torch.abs(X)-lamda)
    
    return Gamma


#--------------------------------------------------------------


def IHT(Y,W,lambd):
    
    c = PowerMethod(W)
    eta = 1/c
    Gamma = hard_threshold_k(torch.mm(Y,eta*W),lambd)    
    residual = torch.mm(Gamma, W.transpose(1,0)) - Y
    IHT_ITER = 50
    
    norms = np.zeros((IHT_ITER,))

    for i in range(IHT_ITER):
        Gamma = hard_threshold_k(Gamma - eta * torch.mm(residual, W), lambd)
        residual = torch.mm(Gamma, W.transpose(1,0)) - Y
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------


def FISTA(Y,W,lamda,FISTA_ITER):
    
    c = PowerMethod(W)
    eta = (1/c)
    norms = np.zeros((FISTA_ITER,))
#     Y = Y.double()
#     W = W.double()
        
    DtY = torch.mm(Y,eta*W)
    Gamma = soft_threshold(DtY,lamda)
    Z = Gamma.clone()
    Gamma_1 = Gamma.clone()
    t = torch.Tensor([1]).type(Y.type()).to(Y.device)
    c = torch.Tensor([c]).type(Y.type()).to(Y.device)
    
    for i in range(FISTA_ITER):
        Gamma_1 = Gamma.clone()
        residual = torch.mm(Z, W.transpose(1,0)) - Y
        Gamma = soft_threshold(Z - eta * torch.mm(residual, W), lamda/c )
        
        t_1 = t
        t = (1+torch.sqrt(1 + 4*t**2))/2
        #pdb.set_trace()
        Z = Gamma + ((t_1 - 1)/t * (Gamma - Gamma_1))
        
        norms[i] = np.linalg.norm(residual.cpu().detach().numpy(),'fro')/ np.linalg.norm(Y.cpu().detach().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).type(W.type()).to(W.device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm
    
    return nm

#--------------------------------------------------------------


def showFilters(W,ncol,nrows):
    p = int(np.sqrt(W.shape[0]))+2
    Nimages = W.shape[1]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[:,indx].reshape(p-2,p-2)
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im,(1,1),mode='constant')
            indx += 1
            
    return Mosaic

