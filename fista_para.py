# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:59:42 2019

@author: 61995
"""
#import pandas
#import tensorflow as tf
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas
import numpy as np
import scipy.io as scio
import cv2
#from data import *
import torch
import time
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#def dpcn_detect(input_img,A,B,C,d_x1,d_u1,batch_size):
#    img = np.zeros([batch_size,20,20,1])
#    img[:,:,:,0] =input_img
#    imgout = np.zeros([batch_size,2,2,144])
#    for kkk in range(batch_size):
#        for iii in range(6,20,8):
#            for jjj in range(6,20,8):
#                imgout[kkk,int((iii-6)/8),int((jjj-6)/8),:]=(img[kkk,iii-6:iii+6,jjj-6:jjj+6,0]).flatten()
#    imgout = np.transpose(imgout,[3,1,2,0])
#    
#    x_hat = np.zeros([d_x1,2,2,batch_size])
#    cause1,state_cell1 = fista(x_hat,C,B,A,imgout,0,d_x1,d_u1,batch_size)
#    return cause1        
def dpcn_detect(input_img,A,B,C,d_x1,d_u1,batch_size):

    #img = torch.zeros([batch_size,20,20,1])#batch_size
    img = input_img.unsqueeze(-1)
    #imgout = torch.zeros([batch_size,2,2,144])
    
    imgout = img.unfold(1, 12, 8).unfold(2, 12, 8).squeeze()
    imgout = imgout.permute(3,4,1,2,0)
    
    imgout = imgout.reshape([144,-1])
    
    x_hat = torch.zeros([d_x1,2,2,batch_size])

    
    cause1 = fista(x_hat,C,B,A,imgout,0,d_x1,d_u1,batch_size)

    return cause1        
def pre_process(im):

    mn = im.mean()

    sd = im.std()

   

    k = np.ones([5,5])

    k = k/k.sum()

   

    im = (im-mn)/sd

   

    lmn = cv2.filter2D(im,-1,k)

    lmnsq = cv2.filter2D(im**2,-1,k)

    lvar = lmnsq - lmn**2

    lvar[lvar<0] = 0

    lstd = np.sqrt(lvar)

    lstd[lstd<1] = 1

   

    im = im - lmn

    im = im / lstd

    return im        
        
        
        
        
        
        
        
        
        
        
        
#def fista(x_hat,C,B,A,imgout,lamda1,d_x1,d_u,batch_size):
#    state_cell = []
#    for rl in range(2):
#        for cl in range(2):
#            d= {'mu':0.01/d_x1,'L': np.ones([1,batch_size]),'eta':2 , 'tk' :1,'x_hat':x_hat[:,rl,cl,:],'xk':x_hat[:,rl,cl,:],'z_k':x_hat[:,rl,cl,:],'tk_1':1,'CTy':np.matmul(np.transpose(C),imgout[:,rl,cl,:]),'y':imgout[:,rl,cl,:],'xk_1':x_hat[:,rl,cl,:]}
#            
#            state_cell.append(d)
#    CTC = np.matmul(np.transpose(C),C)
#    sumx1 = np.abs(state_cell[0]['xk'])+np.abs(state_cell[1]['xk'])+np.abs(state_cell[2]['xk'])+np.abs(state_cell[3]['xk'])
#    #sumx1 = np.maximum(np.maximum(np.maximum(np.abs(state_cell[0]['xk']),np.abs(state_cell[1]['xk'])),np.abs(state_cell[2]['xk'])),np.abs(state_cell[3]['xk']))
#    keep_going = 1;
#    cause= {'L': np.ones([1,batch_size]),'eta':1.5 , 'tk' :1,'uk':np.zeros([d_u,batch_size]),'uk_1':np.zeros([d_u,batch_size]),'z_k':np.zeros([d_u,batch_size]),'tk_1':1,'x':sumx1,'gamma_bar':1e-3,'gamma':0.5}
#    gama1output = (1+np.exp(-np.matmul(B,cause['uk'])))/2
#    keep_going = 1;
#    nz_x=np.zeros([4,d_x1,batch_size])
#    for reg in range(4):
#        nz_x[reg,:,:] = (np.abs(state_cell[reg]['xk']) >= 2.2204e-15)
#    maxprotect = 0
#    while keep_going:
#        for reg in range(4):
##            alpha = (state_cell[reg]['z_k']-state_cell[reg]['x_hat'])/state_cell[reg]['mu']#(d_x1,4)
##            alpha[alpha > 1] =1;
##            alpha[alpha < -1] = -1
#            grad_zk =(np.matmul(CTC,state_cell[reg]['z_k']) - state_cell[reg]['CTy'])#+ lamda1*alpha;
#            const = 0.5*np.sum((np.matmul(C,state_cell[reg]['z_k'])-state_cell[reg]['y'])**2,0)# + lamda1*np.sum(np.abs(state_cell[reg]['z_k'] - state_cell[reg]['x_hat']),0);
#            stop_linesearch = np.zeros([batch_size,1])
#            protect = 0
#            
#            while np.sum(stop_linesearch) != batch_size:
#                gk=state_cell[reg]['z_k'] - grad_zk/state_cell[reg]['L']
#                state_cell[reg]['xk'] = np.sign(gk)*np.maximum((np.abs(gk)-gama1output/state_cell[reg]['L']),0)
#                
#                #lossx.append(0.5*np.sum((np.matmul(C,state_cell[reg]['xk'])-state_cell[reg]['y'])**2) + lamda1*np.sum(np.abs(state_cell[reg]['xk'] - state_cell[reg]['x_hat']))+0*1/batch_size*np.sum(gama1output*cause['x']))
#                #lossx.append(np.sum(const))
#                temp1 = 0.5*np.sum((state_cell[reg]['y'] - np.matmul(C,state_cell[reg]['xk']))**2,0)# + lamda1*np.sum(abs(state_cell[reg]['xk'] - state_cell[reg]['x_hat']),0) #(y-x)**2 x(144,5)-->(1,5)
#                temp2 = const + np.sum((state_cell[reg]['xk'] - state_cell[reg]['z_k'])*grad_zk,0) + np.sum((state_cell[reg]['L']/2),0)*np.sum(((state_cell[reg]['xk'] - state_cell[reg]['z_k']))**2,0)#(1,5)+(1,5).*sum((x-x')**2)
#                indx = (temp1<= temp2) 
#                stop_linesearch[indx] = True;
#                decay = np.ones([batch_size,1])
#                decay = (1-stop_linesearch)*state_cell[reg]['eta']
#                decay[decay==0] = 1
#                state_cell[reg]['L'] = np.transpose(decay)*state_cell[reg]['L']
#                protect += 1 
#                if protect>5:
#                    break
#                #print(count)
#
#        
#                
#                #state_cell[reg]['L'][indx] = self.eta*state_cell[reg]['L'](~indx);
#            state_cell[reg]['tk_1'] = (1 + np.sqrt(4*state_cell[reg]['tk']**2 + 1))/2
#            state_cell[reg]['z_k'] = state_cell[reg]['xk'] + (state_cell[reg]['tk'] - 1)/(state_cell[reg]['tk_1'])*(state_cell[reg]['xk'] - state_cell[reg]['xk_1']);
#            state_cell[reg]['xk_1'] = state_cell[reg]['xk']
#            state_cell[reg]['tk'] = state_cell[reg]['tk_1'];
#            X = state_cell[reg]['xk']
#        cause['x']=np.abs(state_cell[0]['xk'])+np.abs(state_cell[1]['xk'])+np.abs(state_cell[2]['xk'])+np.abs(state_cell[3]['xk'])
#        #sess.run(u1_int, feed_dict={u1_p_r:cause['z_k'],})
#        exp_func = np.exp(-np.matmul(B,cause['z_k']))/2 #np.array(sess.run(gama1t))[:,0,0,:]-0.5#exp(-Para.B*self.z_k)/2;
#             
#        grad_zk = -np.matmul(np.transpose(B),(exp_func*cause['x']))#'*(exp_func.*self.x);
#        const = np.sum(cause['x']*np.exp(-np.matmul(B,cause['z_k']))/2,0)#sum(alpha*(self.x.*exp(-Para.B*self.z_k)/2),1);
#        stop_linesearch = np.zeros([batch_size,1]);
#        protect = 0
#        while np.sum(stop_linesearch) != batch_size:
#            gk=cause['z_k'] - grad_zk/cause['L']
#            cause['uk'] = np.sign(gk)*np.maximum((np.abs(gk)-cause['gamma']/cause['L']),0)
#            
#            #lossu.append(0*0.5*np.sum((np.matmul(C,state_cell[reg]['xk'])-state_cell[reg]['y'])**2,0) + 0*lamda1*np.sum(np.abs(state_cell[reg]['xk'] - state_cell[reg]['x_hat']),0)+1/batch_size*np.sum(gama1output*cause['x']))
#            #lossu.append(np.sum(const))
#            #sess.run(u1_int, feed_dict={u1_p_r:cause['uk'],})
#            temp1 = np.sum(cause['x']*np.exp(-np.matmul(B,cause['uk']))/2,0)
#            temp2 = const+np.sum((cause['uk'] - cause['z_k'])*grad_zk,0)+ np.sum((cause['L']/2),0)*np.sum(((cause['uk'] - cause['z_k']))**2,0)
#            indx = (temp1<= temp2) 
#            stop_linesearch[indx] = True;
#            decay = np.ones([batch_size,1])
#            decay = (1-stop_linesearch)*cause['eta']
#            decay[decay==0] = 1
#            cause['L'] = np.transpose(decay)*cause['L']
#            protect += 1
#            if protect>5:
#                break
#
# 
#        cause['gamma'] = np.maximum(0.99*cause['gamma'],cause['gamma_bar']);
#        cause['tk_1'] = (1 + np.sqrt(4*cause['tk']**2 + 1))/2
#        cause['z_k'] = cause['uk'] + (cause['tk'] - 1)/(cause['tk_1'])*(cause['uk'] - cause['uk_1']);
#        cause['uk_1'] = cause['uk']
#        cause['tk'] = cause['tk_1'];
#            
#        gama1output = (1+np.exp(-np.matmul(B,cause['uk'])))/2
#       
#            
#        
#        
#        #keep_going -= 1
#        #print(keep_going)
#        #U = cause['uk']
# 
#        nz_x_prev = nz_x
#        nz_x=np.zeros([4,d_x1,batch_size])
#        for reg in range(4):
#            nz_x[reg,:,:] = (np.abs(state_cell[reg]['xk']) >= 2.2204e-15)
#            
#        num_changes_active = 0
#        for reg in range(4):
#            num_changes_active = num_changes_active+np.sum(nz_x[reg,:,:] != nz_x_prev[reg,:,:])
#        num_nz_x = np.sum(nz_x)
#        if num_nz_x >= 1:
#            criterionActiveSet = num_changes_active / num_nz_x
#            keep_going = (criterionActiveSet > 0.01)
#        maxprotect += 1
#        if maxprotect >=20:
#            keep_going=0
#            
#    return cause,state_cell  
def fista(x_hat,C,B,A,imgout,lamda1,d_x1,d_u,batch_size):
    
    state_cell = []
    for rl in range(1):
        for cl in range(1):
            d= {'mu':0.01/d_x1,'L': torch.ones([1,int(batch_size*4)]),'eta':2 , 'tk' :1,'x_hat':torch.zeros([d_x1,int(batch_size*4)]),'xk':torch.zeros([d_x1,int(batch_size*4)]),'z_k':torch.zeros([d_x1,int(batch_size*4)]),'tk_1':1,'CTy':torch.mm(C.T,imgout),'y':imgout,'xk_1':torch.zeros([d_x1,int(batch_size*4)])}
            
            state_cell.append(d)
    
    
    CTC = torch.mm(C.T,C)
    sumx1 = torch.abs(state_cell[0]['xk'])+torch.abs(state_cell[0]['xk'])+torch.abs(state_cell[0]['xk'])+torch.abs(state_cell[0]['xk'])
    
    
    cause= {'L': torch.ones([1,batch_size]),'eta':1.5 , 'tk' :1,'uk':torch.zeros([d_u,batch_size]),'uk_1':torch.zeros([d_u,batch_size]),'z_k':torch.zeros([d_u,batch_size]),'tk_1':1,'x':sumx1,'gamma_bar':1e-3,'gamma':0.5}
    gama1output = torch.zeros([d_x1,2,2,batch_size])
    for rl in range(2):
        for cl in range(2):
            gama1output[:,rl,cl,:] = (1+torch.exp(-torch.mm(B,cause['uk'])))/2
    gama1output = gama1output.reshape([d_x1,-1])
    keep_going = 1
    nz_x=torch.zeros([4,d_x1,batch_size])
    for reg in range(4):
        nz_x[reg,:,:] = (state_cell[0]['xk'].reshape([d_x1,4,batch_size])[:,reg,:] >= 2.2204e-15)
    maxprotect = 0

    while keep_going:
        
        for reg in range(1):#4
            
#            alpha = (state_cell[reg]['z_k']-state_cell[reg]['x_hat'])/state_cell[reg]['mu']#(d_x1,4)
#            alpha[alpha > 1] =1;
#            alpha[alpha < -1] = -1
            grad_zk =(torch.mm(CTC,state_cell[reg]['z_k']) - state_cell[reg]['CTy'])#+ lamda1*alpha;
            const = 0.5*torch.sum((torch.mm(C,state_cell[reg]['z_k'])-state_cell[reg]['y'])**2,dim = 0)# + lamda1*torch.sum(torch.abs(state_cell[reg]['z_k'] - state_cell[reg]['x_hat']),dim = 0);
            stop_linesearch = torch.zeros([int(4*batch_size),1])
            protect = 0
            #print(reg)
            
            while torch.sum(stop_linesearch) != int(batch_size*4):
                gk=state_cell[reg]['z_k'] - grad_zk/state_cell[reg]['L']
                #mks, _ = torch.max((torch.abs(gk)-gama1output/state_cell[reg]['L']),dim = 0)
                mks = torch.abs(gk)-gama1output/state_cell[reg]['L']
                mks[mks<0] = 0

                state_cell[reg]['xk'] = torch.sign(gk)*mks
                
#                lsx = 0.5*torch.sum((torch.mm(C,state_cell[reg]['xk'])-state_cell[reg]['y'])**2)# + lamda1*torch.sum(torch.abs(state_cell[reg]['xk'] - state_cell[reg]['x_hat']))+0*1/batch_size*torch.sum(gama1output*cause['x'])
#                lossx.append(np.asscalar(torch.sum(lsx).cpu().numpy()))
                temp1 = 0.5*torch.sum((state_cell[reg]['y'] - torch.mm(C,state_cell[reg]['xk']))**2,dim = 0)# + lamda1*torch.sum(abs(state_cell[reg]['xk'] - state_cell[reg]['x_hat']),dim = 0) #(y-x)**2 x(144,5)-->(1,5)
                temp2 = const + torch.sum((state_cell[reg]['xk'] - state_cell[reg]['z_k'])*grad_zk,0) + torch.sum((state_cell[reg]['L']/2),dim = 0)*torch.sum(((state_cell[reg]['xk'] - state_cell[reg]['z_k']))**2,dim = 0)#(1,5)+(1,5).*sum((x-x')**2)
                indx = (temp1<= temp2)
                stop_linesearch[indx] = True
                decay = torch.ones([int(batch_size*4),1])
                decay = (1-stop_linesearch)*state_cell[reg]['eta']
                decay[decay==0] = 1
                state_cell[reg]['L'] = (decay.T)*state_cell[reg]['L']
                protect += 1
                if protect>5:
                    break 
                #print('xk')
            state_cell[reg]['tk_1'] = (1 + np.sqrt(4*state_cell[reg]['tk']**2 + 1))/2
            state_cell[reg]['z_k'] = state_cell[reg]['xk'] + (state_cell[reg]['tk'] - 1)/(state_cell[reg]['tk_1'])*(state_cell[reg]['xk'] - state_cell[reg]['xk_1']);
            state_cell[reg]['xk_1'] = state_cell[reg]['xk']
            state_cell[reg]['tk'] = state_cell[reg]['tk_1']

                
        stc = state_cell[0]['xk'].reshape([d_x1,4,batch_size])
        cause['x']=torch.abs(stc[:,0,:])+torch.abs(stc[:,1,:])+torch.abs(stc[:,2,:])+torch.abs(stc[:,3,:])          
        
        exp_func = torch.exp(-torch.mm(B,cause['z_k']))/2
        grad_zk = -torch.mm(B.T,(exp_func*cause['x']))
        const = torch.sum(cause['x']*torch.exp(-torch.mm(B,cause['z_k']))/2,dim = 0)
        stop_linesearch = torch.zeros([batch_size,1])
        protect = 0

        
        while torch.sum(stop_linesearch) != batch_size:
            #print('uk')
            gk=cause['z_k'] - grad_zk/cause['L']
            #mks, _ = torch.max((torch.abs(gk)-cause['gamma']/cause['L']),dim = 0)
            
            mks = torch.abs(gk)-cause['gamma']/cause['L']
            mks[mks<0] = 0
        
            cause['uk'] = torch.sign(gk) * mks
            
            
#            lsu = (1/batch_size*torch.sum(gama1output*cause['x']))
#            lossu.append(np.asscalar(torch.sum(const).cpu().numpy()))
            
            temp1 = torch.sum(cause['x']*torch.exp(-torch.mm(B,cause['uk']))/2,dim = 0)
            temp2 = const+torch.sum((cause['uk'] - cause['z_k'])*grad_zk,dim = 0)+ torch.sum((cause['L']/2),dim = 0)*torch.sum(((cause['uk'] - cause['z_k']))**2,dim =0)
            indx = (temp1<= temp2) 
            stop_linesearch[indx] = True;
            decay = torch.ones([batch_size,1])
            decay = (1-stop_linesearch)*cause['eta']
            decay[decay==0] = 1
            cause['L'] = (decay.T)*cause['L']
            protect += 1
            if protect>5:
                break 
            
        cause['gamma'] = np.maximum(0.99*cause['gamma'],cause['gamma_bar']);
        cause['tk_1'] = (1 + np.sqrt(4*cause['tk']**2 + 1))/2
        cause['z_k'] = cause['uk'] + (cause['tk'] - 1)/(cause['tk_1'])*(cause['uk'] - cause['uk_1']);
        cause['uk_1'] = cause['uk']
        cause['tk'] = cause['tk_1'];
                
        gama1output = torch.zeros([d_x1,2,2,batch_size])
        for rl in range(2):
            for cl in range(2):
                gama1output[:,rl,cl,:] = (1+torch.exp(-torch.mm(B,cause['uk'])))/2
        gama1output = gama1output.reshape([d_x1,-1])
        nz_x_prev = nz_x
        nz_x=torch.zeros([4,d_x1,batch_size])
        for reg in range(4):
            nz_x[reg,:,:] = (torch.abs(state_cell[0]['xk'].reshape([d_x1,4,batch_size])[:,reg,:]) >= 2.2204e-15)
        num_changes_active = 0
        for reg in range(4):
            num_changes_active = num_changes_active+torch.sum(nz_x[reg,:,:] != nz_x_prev[reg,:,:])
        num_nz_x = torch.sum(nz_x)
        if num_nz_x >= 1:
            criterionActiveSet = num_changes_active / num_nz_x
            keep_going = (criterionActiveSet > 0.2)
        maxprotect += 1
        if maxprotect >=20:
            keep_going=0
    return cause


def Gamma(test_image,k=np.array([1, 20, 1, 30, 1, 35], dtype=float),mu = np.array([4, 4, 4, 4, 4, 4], dtype=float)):
    test_image = foai.ImageObject(test_image)
    foveation_prior = foac.matlab_style_gauss2D(test_image.modified.shape, 300)
    kernel = foac.gamma_kernel(test_image,k = k,mu = mu)
    image_height = 240
    image_width = 256
    rankCount = 8  # Number of maps scans to run
    img_sz = 32
    processed_patch = np.empty((1, rankCount, img_sz, img_sz, 3))
    processed_location = np.empty((1, rankCount, 2))
    
    foac.convolution(test_image, kernel, foveation_prior)
    map1 = foas.salience_scan(test_image, rankCount=rankCount)
    processed_patch = (test_image.patch).astype(np.uint8)
    processed_location = (test_image.location).astype(np.int)
    #map1 = (test_image.salience_map).astype(np.int)
    return processed_patch,processed_location, map1

#scio.savemat('state21.mat',mdict={'state': processed_patch})

def DPCN_score(processed_patch,d_x1,d_u1,d_input1,batch_size,A,B,C,ind_ac):
    length = len(processed_patch)
    state = processed_patch/255
    state = 0.2989 * state[:,:,:,0] + 0.5870 * state[:,:,:,1] + 0.1140 * state[:,:,:,2]
    for i in range(length):
        state[i,:,:] = pre_process(state[i,:,:])
    img_sz = 32
    AA = np.zeros([length,img_sz-20,20,20,img_sz-20])
    for ss in range(length):
        for i in range(img_sz-20):
            for j in range(img_sz-20):
                AA[ss,i,:,:,j] = state[ss,i:i+20,j:j+20]
    AA = np.transpose(AA,[0,2,3,1,4])
    AA = np.reshape(AA,[length,20,20,-1])      
    AA = np.transpose(AA,[0,3,1,2])
    AA = np.reshape(AA,[-1,20,20])
#    output = np.zeros([d_u1,1,batch_size])
    
    for i in range(1):
        #print(i)
        #input_img = AA[:,:,:,i]
        
        input_img = torch.Tensor(AA)  
        
        U = dpcn_detect(input_img,A,B,C,d_x1,d_u1,int(length*batch_size))

#        output[:,i,:] = U['uk']
    output = np.reshape(U['uk'].cpu().numpy(),[d_u1,length,batch_size])

    #output = np.reshape(output,[60,12,12])
    output[ind_ac,:,:] = 0
    output[output<1.2] = 0
    a = np.sum(output,0)
    a = np.reshape(a,[length,batch_size])

    return a


def thresholding(count_patch,processed_patch,mario):
    if mario and (np.where(count_patch!=0))[0]!=[]:
        
        pt = (np.where(count_patch==np.max(count_patch)))[0]
    else:
        pt = (np.where(count_patch!=0))[0]
    #
    ptt = np.zeros([5,])
    ptt[0:len(pt)] = pt[0:5]
    ind = True
    if len(pt) == 0:
        ind = False
    
    if len(pt) <5 and len(pt)>0:
        for i in range(5-len(pt)):
            ptt[-(i+1)] = pt[-1]
    
        
    ptt = ptt.astype(np.int64)        
    #ptt = np.concatenate([ptt])       
    
    
#    batch = (np.flip(processed_patch[ptt,:,:,:],-1)/255).astype(np.float32) #color channel for different api
#    batch = batch[:,2:30,2:30,:]  
#    batch_concat = np.concatenate([batch,batch_anchor],0)
    return ptt,ind,len(pt)



def scores_locations(img,A,B,C,ind_ac,processed_patch,processed_location,mario = False):
#    start = time.time()
    test_image = img
    if mario:
        ppp = []
        for i in range(len(processed_patch)):
            ppp.append(cv2.flip(processed_patch[i],1))
        ppp = np.array(ppp)
    #processed_patch,processed_location = Gamma(test_image)
#    scores_patch = np.zeros([8,])#rank
    #count_patch = np.zeros([8,])#rank
    
    
    
    for i in range(1):#rank
        a = DPCN_score(processed_patch,150,60,144,int(12*12),A,B,C,ind_ac)
        scores_patch = np.sum(a,1)
        count_patch = (np.sum(a,1)>0).astype(np.int)
        
    if mario:
        for i in range(1):#rank
            a_f = DPCN_score(ppp,150,60,144,int(12*12),A,B,C,ind_ac)
            scores_patch_f = np.sum(a_f,1)
            count_patch_f = (np.sum(a,1)>0).astype(np.int)
            
        a = np.maximum(a,a_f)
        scores_patch = np.maximum(scores_patch,scores_patch_f)
        count_patch = np.maximum(count_patch,count_patch_f)

    ptt,ind,num_object = thresholding(scores_patch,processed_patch,mario)
#    stop = time.time()
#    print("Salience Map Generation: ", stop - start, " seconds")
    return (processed_patch[ptt,:])[0:num_object,:],(processed_location[ptt,:])[0:num_object,:],num_object,ind,a


    


def labeling(batch_concat,batch_size_c,net,device,num_object):
    oringinal_data = IICDataset(ip = batch_concat,transform = 1)
    
            
    feed_x = torch.ones((batch_size_c,3,28,28), dtype=torch.float)
    for s in range(batch_size_c):
        sample = oringinal_data[s]   
        feed_x[s,:,:,:] = sample['image']
    feed_x.to(device)
    latent_x,_ =net(feed_x)
    lab_a = (latent_x.cpu().detach().numpy())
    
    return lab_a[0:num_object]



def TK(location_p,location):
    #_,location,_ = Gamma(frame)
    
    location_r = np.zeros(location_p.shape)
    for i in range(np.size(location_p,0)):
        dis_list = []
        for j in range(len(location)):
            dis = (location[j][0]-location_p[i][0])**2+(location[j][1]-location_p[i][1])**2
            dis_list.append(dis)
        dis_list = np.array(dis_list)
        index = np.where(dis_list == np.min(dis_list))[0][0]
        location_r[i,:] = location[index]
    
    return location_r


def Merge(processed_location3):
    processed_location3_1 = np.zeros(processed_location3.shape)
    biaoji = np.arange(len(processed_location3))
    for i in range(len(processed_location3)):
        for j in range(len(processed_location3)-1-i):
            if np.abs(processed_location3[i,1] - processed_location3[j,1])<5:
                processed_location3[j,0] =min(processed_location3[j,0],processed_location3[i,0])
                biaoji[i] = 100
    
    return processed_location3[np.where(biaoji!=100)]


def truth_matching(img_gray,template):

#template = img[177:189,114:143,:]
    # print("GT")
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    
    
    threshold = 0.7
    return np.where( res >= threshold)

# to accurately detect mario vs blocks
def truth_matching_mario(img_gray,template):

#template = img[177:189,114:143,:]
    # print("GT")
    w, h = template.shape[::-1]
    
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    
    
    threshold = 0.75
    return np.where( res >= threshold)

if __name__ == "__main__":
    pass