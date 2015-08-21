import random
import os
import sys
import math
import numpy as np
import copy




dataName="D://python2.7.6//MachineLearning//rotated0-9_CNN2//dataVec.txt" #n x 16x16 matric
labelfile= "D://python2.7.6//MachineLearning//rotated0-9_CNN2//dataLabel.txt" #1 x n matric
 
outPath="D://python2.7.6//MachineLearning//rotated0-9_CNN2//para"     

global classDic,labelList,dataList #11 2000
 
global epoch;epoch=100
global alpha;alpha=1.0
alpha0=0
swind=2#subsample window
sarea=swind**2#4

xd=16#change 32
xdd=xd*xd
numv=6
numbbb=6
vdim=3#5
vdd=vdim**2#25

c1dim=xd-vdim+1#28
c1dd=c1dim**2

s2dim=c1dim/swind#14
s2dd=s2dim**2

numw=12 
numbb=16 
wdim1=3######5 
wdim2=3 ########5
wdim3=numv#6
wddd=wdim1*wdim2*wdim3#5x5x6
c3dim=s2dim-wdim1+1#10
c3dd=c3dim**2

s4dim=c3dim/swind#5
s4dd=s4dim**2

numc=10
numb=10


nbatch=1
 


######################

def loadData():
    global dataMat,yMat,classList,labelList,dataList,dataArr
    classDic={};labelList=[];dataList=[]
    ########## all label  list
    content=open(labelfile,'r')
    line=content.readline().strip(' ')
    line=line.split(' ')
    for label in line:
        labelList.append(int(label))
    print '1',len(labelList)
    
    ##########
    obs=[]
    content=open(dataName,'r')
    line=content.readline().strip('\n').strip(' ')
    line=line.split(' ')
    #print line,len(line)
    while len(line)>1:
        
        obs=[float(n) for n in line if len(n)>1]
        #print 'o',obs,len(obs)
        
        line=content.readline().strip('\n').strip(' ');line=line.split(' ')
         
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d obs loaded'%len(dataList),len(set(labelList)),'kinds of labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
    numx=len(dataList)
    #####
    dataMat=np.mat(dataList)
    dataArr=np.zeros((numx,xd,xd))
    for x in range(numx):
        dataArr[x,:,:]=vec2mat(dataMat[x,:],xd,xd)
     
    ########
     
     
    yMat=np.zeros((numx,numc))
    for n in range(numx):
        truey=labelList[n]
        yMat[n,truey]=1.0


def initialH():
    global dataMat,yMat,dataArr
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    nh=c1dd#28x28
    nhh=s2dd #14x14
    nhhh=c3dd #10x10
    nhhhh=s4dd #5x5
    
    n=np.shape(dataArr)[0]
    hMat=np.zeros((numv,c1dim,c1dim)) 
    hhMat=np.zeros((numv,s2dim,s2dim))
    hhhMat=np.zeros((numw,c3dim,c3dim))
    hhhhMat=np.zeros((numw,s4dim,s4dim))
    outputMat=np.mat(np.zeros((n,numc))) #nclass
    
     

def initialPara():
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat,dataArr #initial from random eps
    num=np.shape(dataArr)[0]
      
    Cmat=np.zeros((numc,numw,s4dim,s4dim))
    bmat=np.mat(np.zeros((1,numb)))
    Wmat=np.zeros((numw,wdim3,wdim1,wdim2))
    bbmat=np.mat(np.zeros((1,numbb)))
    Vmat=np.zeros((numv,vdim,vdim))
    bbbmat=np.mat(np.zeros((1,numbbb)))
    ####
    vin=1*vdim**2
    vout=numv*vdim**2
    vr=math.sqrt(6.0/(vin+vout)) 
    win=numv*wdim1**2
    wout=numw*wdim1**2
    wr=math.sqrt(6.0/(win+wout)) 
    cin=numw
    cout=numc
    cr=math.sqrt(6.0/(cin+cout)) 
    
    for i in range(numc):
        for j in range(numw):
            for k in range(s4dim):
                for d in range(s4dim):
                    Cmat[i,j,k,d]=random.uniform((-1.0)*cr,cr)
    for i in range(numw):
        for j in range(wdim3):
            for k in range(wdim1):
                for d in range(wdim2):
                    Wmat[i,j,k,d]=random.uniform((-1.0)*wr,wr)
    for i in range(numv):
        for j in range(vdim):
            for k in range(vdim):
                Vmat[i,j,k]=random.uniform((-1.0)*vr,vr)
    '''####w 6x5x5 sparse
    global wZero
    wZero={0:[3,4,5],1:[0,4,5],2:[0,1,5],3:[0,1,2],4:[1,2,3],5:[2,3,4],6:[4,5],7:[1,5],8:[0,1],9:[1,2],10:[2,3],11:[3,4],12:[2,5],13:[0,3],14:[1,4]}#,15:[]}
    for k in range(numw):#16
        if k in wZero:
            for i in wZero[k]:
                Wmat[k,i,:,:]=np.zeros((wdim1,wdim2))'''
         
        
     
    

            
def initialErr():#transfer err sensitive
    global errW,errC,errV,up1,up2,up11,up22
    global dataArr
     
    n=np.shape(dataArr)[0]
    errW=np.zeros((numw,c3dim,c3dim))
    errC=np.mat(np.zeros((1,numc)))
    errV=np.zeros((numv,c1dim,c1dim))
    up1=np.zeros((numw,s4dim,s4dim))
    up2=np.zeros((numw,c3dim,c3dim))
    up11=np.zeros((numv,s2dim,s2dim))
    up22=np.zeros((numv,c1dim,c1dim))
    
     
    
def initialGrad():
      
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    gradc=np.zeros((numc,numw,s4dim,s4dim))
    gradb=np.mat(np.zeros((1,numb)))
    gradw=np.zeros((numw,wdim3,wdim1,wdim2))
    gradbb=np.mat(np.zeros((1,numbb)))
    gradv=np.zeros((numv,vdim,vdim))
    gradbbb=np.mat(np.zeros((1,numbbb))) 
def initialInc():
    global IncC,IncW,IncV,IncB,IncBB,IncBBB
    IncC=np.zeros((numc,numw,s4dim,s4dim))
    IncB=np.mat(np.zeros((1,numb)))
    IncW=np.zeros((numw,wdim3,wdim1,wdim2))
    IncBB=np.mat(np.zeros((1,numbb)))
    IncV=np.zeros((numv,vdim,vdim))
    IncBBB=np.mat(np.zeros((1,numbbb)))
  
    

def forward(x): #xi index not xvector :1 obs
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global dataArr
    global errW,errC,errV,up1,up2,up11,up22
    global wZero
    xmat=np.mat(dataArr[x,:,:])
     
     
    ####      (32-5+1)x(32-5+1) =  28x28
    ####5x5 convolution 32x32 ->28x28  ||6 v kernel
    for j in range(numv)[:]: #6
        ###each v kernel
        kij=rot180(np.mat(Vmat[j,:,:]))#???????????????????
        feaMat=calcConv(xmat,kij,vdim,c1dim,c1dim);#print '1',feaMat#32x32mat   5x5mat -> 28x28matric
        feaMat=feaMat+bbbmat[0,j];#print '2',feaMat
        feaMat=1.0/(1.0+np.exp((-1.0)*feaMat));#print '3',feaMat
        hMat[j,:,:]=feaMat
     
    #####subsample
    for j in range(numv)[:]: #6
        ####pool with 2x2 window mean pooling
        feaMap=np.mat(hMat[j,:,:]);#print '21',feaMap
        poolMap=np.mat(np.zeros((s2dim,s2dim)))#14x14
        for hang in range(s2dim):
            for lie in range(s2dim):
                patch=feaMap[hang*swind:hang*swind+swind,lie*swind:lie*swind+swind]#subsample window 2x2
                m=patch.mean()
                poolMap[hang,lie]=m
        #####14x14
        hhMat[j,:,:]=poolMap;#print '22',hhMat[j,:,:]
    #########second conv 6x14x14  conv 16@6x5x5 ->16 @ 10x10
    for i in range(numw)[:]: #each w in 16
        hhhMat[i,:,:]=np.zeros((c3dim,c3dim))
        ######
        for j in range(numv):#6
            '''
            if i in wZero and j in wZero[i]:continue
            else:'''
            kij=rot180(np.mat(Wmat[i,j,:,:]))#???????????????????
            hhhMat[i,:,:]=hhhMat[i,:,:]+calcConv(np.mat(hhMat[j,:,:]),kij,wdim1,c3dim,c3dim)
        #print '31',hhhMat[i,:,:]
        ######bias sigmoid
        hhhMat[i,:,:]=hhhMat[i,:,:]+bbmat[0,i];#print '32',hhhMat[i,:,:]
        hhhMat[i,:,:]=1.0/(1.0+np.exp((-1.0)*hhhMat[i,:,:]));#print '33',hhhMat[i,:,:]
    ##########second subsample  16@10x10 ->16@5x5
    for i in range(numw)[:]: #16
        feaMap=np.mat(hhhMat[i,:,:])#10x10
        ####pool with 2x2 window mean pooling
        poolMap=np.mat(np.zeros((s4dim,s4dim)))
        for hang in range(s4dim):
            for lie in range(s4dim):
                patch=feaMap[hang*swind:hang*swind+swind,lie*swind:lie*swind+swind]#subsample window 2x2
                m=patch.mean()
                poolMap[hang,lie]=m
        #####5x5 
        hhhhMat[i,:,:]=poolMap;#print '41',hhhhMat[i,:,:]
    
    #######full connect
    for k in range(numc):#11class
        f=hhhhMat*Cmat[k,:,:,:]+bmat[0,k]#16x5x5  x  16x5x5==
        f=f.flatten().sum()
        outputMat[x,k]=f
    outputMat[x,:]=softmax(outputMat[x,:])
    #print '5',outputMat[x,:]
    ######
     
                
def calcGrad(x):#x index not vec
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataArr
    global errW,errC,errV,up1,up2,up11,up22  
    
    ####err c floor
    fy=outputMat[x,:]-yMat[x,:] #matric 1x10
    sgm=np.array(outputMat[x,:])*(1.0-np.array(outputMat[x,:]))
    errC=np.mat(np.array(fy)*sgm)#matric 1x10
    ######grad c b
    for k in range(numc):
        gradc[k,:,:,:]=errC[0,k]*hhhhMat #1x1  x  16x5x5
        gradb[0,k]=copy.copy(errC[0,k])#1x10 ##cannot change  at the same time
    ####################up1 up2 ->errw -> grad w bb
    ###up1
    ss=np.zeros((numw,s4dim,s4dim))
    for k in range(numc):
        ss=ss+errC[0,k]*Cmat[k,:,:,:]#16x5x5
    ###upsample up2
    for i in range(numw):# 16 kernel w
        up1d=s4dim#dim before upsample or expand 5
        up2d=c3dim#dim after expand              10
        up2[i,:,:]=upsample(up1[i,:,:],up1d,up2d) #1 x 5x5 -> 5x5->10x10-> 1 x 10x10
    #########errw
    for i in range(numw):
        sgm=np.array(hhhMat[i,:,:])*(1.0-np.array(hhhMat[i,:,:]))#1  x  10x10
        errW[i,:,:]=np.mat(sgm*np.array(up2[i,:,:]))#1  x  10x10
    ######grad w bb
    for i in range(numw):#16
        for j in range(numv):#6
            '''if i in wZero and j in wZero[i]:continue
            else: #gradw= rothh conv errw
            '''
            matx=rot180(hhMat[j,:,:])#matx=np.mat(hhMat[j,:,:])#matric ?????????????????????
            errmap=rot180(np.mat(errW[i,:,:]))
            gradw[i,j,:,:]=calcConv(matx,errmap,c3dim,wdim1,wdim2)
            #gradw[i,j,:,:]=rot180(gradw[i,j,:,:])#######????????????????????????
        ####
        gradbb[0,i]=errW[i,:,:].flatten().sum()
         
    #######################up11 up22 -> errV -> gradv bbb
    ###up11 14x14
    for j in range(numv):#6 pieces 
        up11[j,:,:]=np.zeros((s2dim,s2dim))#  14x14
        ####accumulated 16 pieces
        for i in range(numw): #16pieces
            mat10=np.mat(errW[i,:,:])#10x10
            mat18=fill0mat(mat10,c3dim,s2dim+wdim1-1) #10x10 mat->18x18mat
            kij=np.mat(Wmat[i,j,:,:])#wijmat=rot180(Wmat[i,j,:,:])#?????????????????????????
            up11[j,:,:]=up11[j,:,:]+calcConv(mat18,kij,wdim1,s2dim,s2dim) # 18x18  conv  5x5->14x14->vec
    ###upsample up22 28x28
    for j in range(numv):#6
        up11d=s2dim#14
        up22d=c1dim#28
        up22[j,:,:]=upsample(up11[j,:,:],up11d,up22d) #1  x  14x14->14x14->28x28 
    #####up22 done ->err v
    for j in range(numv):#6
        sgm=np.array(hMat[j,:,:])*(1.0-np.array(hMat[j,:,:]))#28x28
        errV[j,:,:]=np.mat(np.array(up22[j,:,:])*sgm)
    ###grad  v bbb
    for j in range(numv):
        xmat=rot180(np.mat(dataArr[x,:,:]))#xmat=np.mat(dataArr[x,:,:])#?????????????????
        errmap=rot180(np.mat(errV[j,:,:]))
        gradv[j,:,:]=calcConv(xmat,errmap,c1dim,vdim,vdim)
        #gradv[j,:,:]=rot180(gradv[j,:,:])#??????????????????????????
        gradbbb[0,j]=errV[j,:,:].flatten().sum()

def divideNormalizeGrad():
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    #############divide num of obs in batch
    gradc=gradc/float(nbatch)
    gradw=gradw/float(nbatch)
    gradv=gradv/float(nbatch)
    gradb=gradb/float(nbatch)
    gradbb=gradbb/float(nbatch)
    gradbbb=gradbbb/float(nbatch)
    ##############gradient normalize
    for k in range(numc):
        gradc[k,:,:,:]=normalize(gradc[k,:,:,:],'vector')#length=1
    for k in range(numw):
        gradw[k,:,:,:]=normalize(gradw[k,:,:,:],'vector')
    for k in range(numv):
        gradv[k,:,:]=normalize(gradv[k,:,:],'vector')

def calcInc():
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global IncC,IncW,IncV,IncB,IncBB,IncBBB
     
    IncC=IncC*alpha0+(-1.0)*alpha*gradc
    IncW=IncW*alpha0+(-1.0)*alpha*gradw
    IncV=IncV*alpha0+(-1.0)*alpha*gradv
    IncB=IncB*alpha0+(-1.0)*alpha*gradb
    IncBB=IncBB*alpha0+(-1.0)*alpha*gradbb
    IncBBB=IncBBB*alpha0+(-1.0)*alpha*gradbbb
    '''
    IncC=(-1.0)*alpha*gradc
    IncW=(-1.0)*alpha*gradw
    IncV=(-1.0)*alpha*gradv
    IncB=(-1.0)*alpha*gradb
    IncBB=(-1.0)*alpha*gradbb
    IncBBB=(-1.0)*alpha*gradbbb
    '''
    
    
def updatePara():
    global IncC,IncW,IncV,IncB,IncBB,IncBBB
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
     
    Cmat=Cmat+IncC
    Wmat=Wmat+IncW
    Vmat=Vmat+IncV
    bmat=bmat+IncB
    bbmat=bbmat+IncBB
    bbbmat=bbbmat+IncBBB
    #####
    '''####w 6x5x5 sparse
    
    wZero={0:[3,4,5],1:[0,4,5],2:[0,1,5],3:[0,1,2],4:[1,2,3],5:[2,3,4],6:[4,5],7:[1,5],8:[0,1],9:[1,2],10:[2,3],11:[3,4],12:[2,5],13:[0,3],14:[1,4]}#,15:[]}
    for k in range(numw):#16
        if k in wZero:
            for i in wZero[k]:
                Wmat[k,i,:,:]=np.zeros((wdim1,wdim2))'''
    
    
def calcLoss(obs):#based on 100 fixed sample ,no based on obs after ff bp  
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataArr,labelList # fk is calculated with old para
    num=np.shape(dataArr)[0]
    loss=0.0
    diff=np.mat(outputMat[obs,:]-yMat[obs,:])
    ss=diff*diff.T;ss=ss[0,0];loss=ss;loss=loss*0.5
    '''
    for n in range(num)[:]:
        diff=np.mat(outputMat[n,:]-yMat[n,:])#1x10 mat
        ss=diff*diff.T;ss=ss[0,0] 
        loss+=ss
     
    loss=loss*0.5/float(num)'''
    #print 'least square loss',loss
    #####err
    err=0.0
    for i in range(num)[:]:
        truey=labelList[i]
        maxL=-10;maxProb=-10
        for k in range(numc):
            if maxProb==-10 or maxProb<outputMat[i,k]:
                maxProb=outputMat[i,k]
                maxL=k#update leizhu
        #print truey,maxlabel
        if truey!=maxL:err+=1.0
    errRate=err/float(num)
    return loss,errRate
    
def gradCheckerC(x):#x index not xvec
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataMat
    
    eps=0.0001
    for c in range(numc):
        for dim in range(numw*s4dd):#16x5x5
            ###################calc loss postive
            #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
            Cmat[c,dim]=Cmat[c,dim]+eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx1=outputMat[x,:]-yMat[x,:]
            lossx1=lossx1*lossx1.T
            lossx1=lossx1[0,0]*0.5
            ##################calc loss negative
            #####theta vec negative=theta vec - [1,0,0,0,..]
            Cmat[c,dim]=Cmat[c,dim]-2.0*eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx2=outputMat[x,:]-yMat[x,:]
            lossx2=lossx2*lossx2.T
            lossx2=lossx2[0,0]*0.5
            ######difference between loss
            loss12=(lossx1-lossx2)/eps*2.0
            #######check: compare numgrad with derivative grad
            if abs(loss12-gradc[c,dim])>0.001:
                print c,dim,'gradc1-gradc2',loss12-gradc[c,dim]


def gradCheckerW(x):#x index not xvec
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataMat
    
    eps=1.0
    for w in range(numw):
        for dim in range(wddd):#6x5x5
            ###################calc loss postive
            #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
            Wmat[w,dim]=Wmat[w,dim]+eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx1=outputMat[x,:]-yMat[x,:]
            lossx1=lossx1*lossx1.T
            lossx1=lossx1[0,0]*0.5
            ##################calc loss negative
            #####theta vec negative=theta vec - [1,0,0,0,..]
            Wmat[w,dim]=Wmat[w,dim]-2.0*eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx2=outputMat[x,:]-yMat[x,:]
            lossx2=lossx2*lossx2.T
            lossx2=lossx2[0,0]*0.5
            ######difference between loss
            loss12=(lossx1-lossx2)/eps*2.0
            #######check: compare numgrad with derivative grad
            if abs(loss12-gradw[w,dim])>0.001:
                print w,dim,'gradc1-gradc2',loss12-gradw[w,dim]  

def gradCheckerV(x):#x index not xvec
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataMat
    
    eps=1.0
    for v in range(numv):
        for dim in range(vdim*vdim):#6x5x5
            ###################calc loss postive
            #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
            Vmat[v,dim]=Vmat[v,dim]+eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx1=outputMat[x,:]-yMat[x,:]
            lossx1=lossx1*lossx1.T
            lossx1=lossx1[0,0]*0.5
            ##################calc loss negative
            #####theta vec negative=theta vec - [1,0,0,0,..]
            Vmat[v,dim]=Vmat[v,dim]-2.0*eps
            forward(x)#to get h outputf
            ###calc loss  
            lossx2=outputMat[x,:]-yMat[x,:]
            lossx2=lossx2*lossx2.T
            lossx2=lossx2[0,0]*0.5
            ######difference between loss
            loss12=(lossx1-lossx2)/eps*2.0
            #######check: compare numgrad with derivative grad
            if abs(loss12-gradv[v,dim])>0.001:
                print v,dim,'gradc1-gradc2',loss12-gradv[v,dim]            
            
def gradCheckerB(x):#x index not xvec
    global gradc,gradw,gradv,gradb,gradbb,gradbbb
    global hMat,hhMat,hhhMat,hhhhMat,outputMat
    global Cmat,Wmat,Vmat,bmat,bbmat,bbbmat
    global outputMat,yMat,dataMat
    
    eps=1.0
    for b in range(numbbb): 
        ###################calc loss postive
        #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
        bbbmat[0,b]=bbbmat[0,b]+eps
        forward(x)#to get h outputf
        ###calc loss  
        lossx1=outputMat[x,:]-yMat[x,:]
        lossx1=lossx1*lossx1.T
        lossx1=lossx1[0,0]*0.5
        ##################calc loss negative
        #####theta vec negative=theta vec - [1,0,0,0,..]
        bbbmat[0,b]=bbbmat[0,b]-2.0*eps
        forward(x)#to get h outputf
        ###calc loss  
        lossx2=outputMat[x,:]-yMat[x,:]
        lossx2=lossx2*lossx2.T
        lossx2=lossx2[0,0]*0.5
        ######difference between loss
        loss12=(lossx1-lossx2)/eps*2.0
        #######check: compare numgrad with derivative grad
        print 'gradc1-gradc2',loss12-gradbbb[0,b]
 

##########################################support function
    
def vec2mat(vec,nhang,nlie):# vec->6 x  5x5
    m1,n1=np.shape(vec)
    szdiff=m1*n1-nhang*nlie
    if szdiff!=0:
        print 'this vec cannot transfer into mat'
         
    ############
    Mat=np.mat(np.zeros((nhang,nlie)))
    for m in range(nhang):
        for n in range(nlie):
            pos=m*nlie+n
            Mat[m,n]=vec[0,pos]
    return Mat
    
    
def shuffleObs(): #shuffle index of obs
    global dataMat
    num,dim=np.shape(dataMat) #1394 piece of obs
    order=range(num)[:]  #0-100  for loss calc,101...for train obs by obs ///not work. must use whole set to train
    random.shuffle(order)
    #####
     
    
    return order
    
    

def softmax(outputMat): #1x10 vec
    vec=np.exp(outputMat)  #1x10  #wh+b
    ss=vec.sum(1);ss=ss[0,0]
    outputMat=vec/(ss+0.000001)
    return outputMat
    
def normalize(cub,opt):
    if opt=='vector': #in order to mode or length ||vec||=1
        mode=cub*cub
        mode=mode.flatten().sum()
        mode=math.sqrt(mode)
        cub=cub/(mode+0.0001)
    if opt not in ['vector','prob']:
        print 'only vector or prob'
    return cub 

def calcConv(xmat,kernelmat,kerneldim,convdim1,convdim2): #not add bias not sigmoid
    ###      
    convmat=np.mat(np.zeros((convdim1,convdim2)))
    ######
    for i in range(convdim1):
        for j in range(convdim2):
            patch=xmat[i:i+kerneldim,j:j+kerneldim]
            s1=patch.A*kernelmat.A          #only matric could :s1.sum(0).sum(1)[0,0]
            s1=s1.sum(1).sum(0)         #if array must     :s1.sum(1).sum(0)
            convmat[i,j]=s1
    return convmat
            
def upsample(mat,up1d,up2d):
    up2mat=np.zeros((up2d,up2d))
    ######calc up2 :upsample: expand and divide 2x2 pooling windon
    for hang in range(up1d):
        for lie in range(up1d):
            m=mat[hang,lie]/float(sarea)
            mat2x2=np.mat(np.zeros((swind,swind)))+m#2x2 window filed with mean/4
            up2mat[swind*hang:swind*hang+swind,swind*lie:swind*lie+swind]=mat2x2
    return up2mat

 
def fill0mat(matsmall,dsmall,dlarge): #10x10mat->18x18mat
    matlarge=np.mat(np.zeros((dlarge,dlarge)))
    margin=(dlarge-dsmall)/2 #(18-10)/2==4
    matlarge[margin:dlarge-margin,margin:dlarge-margin]=matsmall
    return matlarge

 
                
def rot180(wijmat):
    m,n=np.shape(wijmat)#5 x 5
    wijmat2=np.mat(np.zeros((m,n)))
    for i in range(m):
        for j in range(n):
            wijmat2[i,j]=wijmat[m-1-i,n-1-j]
    return wijmat2
  

    
    

###################main
loadData()
initialH()
initialPara()
initialErr()
initialGrad()
initialInc()
####
'''
forward(1)
calcGrad(1)
gradCheckerC(1)
gradCheckerW(1)
gradCheckerV(1)
'''
print '2 cnn , momentum update para after 10 ep,no batch, rot as matlab tool '

#####
#####train
for ep in range(epoch)[:]:
    orderList=shuffleObs()
     
    #alpha/=2.0
    if ep>10:alpha0=0.5
    if ep>20:alpha0=0.9
    
    for obs in orderList[:]: #obs=x index not vec
        initialGrad()  
        forward(obs) #compute h output based on one obs
        calcGrad(obs)##accumulate grad over obs 
        #######
        divideNormalizeGrad()
        calcInc()
        updatePara()#momentum method to update para:w=w+inc;inc=incOld x alpha - step x grad
        loss,err=calcLoss(obs)#loss calc with outputf computed with old para based on obs to calc loss
    print  'epoch %d loss %f err %f'%(ep,loss,err)
    if err<0.01:break

'''
#####################output
ctemp=np.zeros((numc,numw*s4dd))
for i in range(numc):
    ctemp[i,:]=Cmat[i,:,:,:].flatten()
Cmat=ctemp
##
wtemp=np.zeros((numw,wddd))
for i in range(numw):
    wtemp[i,:]=Wmat[i,:,:,:].flatten()
Wmat=wtemp
###
vtemp=np.zeros((numv,vdd))
for i in range(numv):
    vtemp[i,:]=Vmat[i,:,:].flatten()
Vmat=vtemp


#####output para w m n c ,b
#global Cmat,Wmat,bmat,bbmat

 
outfile1 = "C.txt"
outfile2 = "W.txt"
outfile22="V.txt"

outfile3 = "B.txt"
outfile4 = "BB.txt"
outfile44="BBB.txt"
 

outPutfile=open(outPath+'/'+outfile1,'w')
n,m=np.shape(Cmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Cmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
##
outPutfile=open(outPath+'/'+outfile2,'w')
n,m=np.shape(Wmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Wmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
###
outPutfile=open(outPath+'/'+outfile22,'w')
n,m=np.shape(Vmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Vmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
#####

outPutfile=open(outPath+'/'+outfile3,'w')
n,m=np.shape(bmat)

for j in range(m):
    outPutfile.write(str(bmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
## 
outPutfile=open(outPath+'/'+outfile4,'w')
n,m=np.shape(bbmat)

for j in range(m):
    outPutfile.write(str(bbmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
####
outPutfile=open(outPath+'/'+outfile44,'w')
n,m=np.shape(bbbmat)

for j in range(m):
    outPutfile.write(str(bbbmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()

'''
 

 

 
 

 
    
         
    
    
    







'''
np.ones((2,3,3))
array([[[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]],

       [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]])
#####
a3[0,0:2,0:2]=a3[0,0:2,0:2]+1
 
array([[[ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 1.,  1.,  1.]],

       [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]])
        '''
    
