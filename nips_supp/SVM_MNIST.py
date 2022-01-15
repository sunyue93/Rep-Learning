import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cvxpy as cp

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X=train_X.astype(float)
test_X=test_X.astype(float)
for i in range(len(train_X)):
    train_X[i]-=np.mean(train_X[i])
    train_X[i]/=np.std(train_X[i])
for i in range(len(test_X)):
    test_X[i]-=np.mean(test_X[i])
    test_X[i]/=np.std(test_X[i])

trX=train_X.reshape([60000,784])
tsX=test_X.reshape([10000,784])

i1=0
i2=2
MAX=10
NUM=int(MAX*(MAX-1)/2)
ctr=0
BT=np.zeros((NUM,784))
BTC=np.zeros((NUM,784))
COV=np.zeros((NUM,784,784))
batch_sz = 800
for i1 in np.arange(MAX):
    for i2 in np.arange(i1+1,MAX):
        print(ctr)
        ly1=train_y==i1
        ly2=train_y==i2
        lt1=test_y==i1
        lt2=test_y==i2
        tr1=trX[ly1]+0.
        tr2=trX[ly2]+0.
        ts1=tsX[lt1]+0.
        ts2=tsX[lt2]+0.
        y=-np.ones((np.sum(ly1)+np.sum(ly2>0)))
        y[:np.sum(ly1)]=1
        yt=-np.ones((np.sum(lt1)+np.sum(lt2>0)))
        yt[:np.sum(lt1)]=1

        ts12=np.concatenate([ts1,ts2])
        tr12=np.concatenate([tr1,tr2])

        batch = np.random.choice(tr12.shape[0],batch_sz)
        print("batch")
        print(batch[:10])
        tr_b = tr12[batch,:]
        y_b = y[batch]

        w = cp.Variable(784)

        objective = cp.Minimize(cp.sum_squares(w))
        constraints = [1 <= cp.multiply(y_b, cp.matmul(tr_b,w))]
        #objective = cp.Minimize(cp.sum_squares(w) + 0.1 * cp.sum(cp.pos(1 - cp.multiply(y_b, cp.matmul(tr_b,w)))))
        #constraints = []
        prob = cp.Problem(objective, constraints)

        #result = prob.solve()
        result = prob.solve(solver=cp.CVXOPT,abstol = 1e-4)
        #result = prob.solve(solver=cp.CBC,maximumSeconds = 10)
        bt = w.value
        #print(cp.matmul(tr_b,w.value).shape)
        #print(y_b.shape)
        #print(cp.multiply(y_b, cp.matmul(tr_b,w.value)).value)
        #ytest = y*cp.matmul(tr12,w)
        #print(ytest.shape)
        #bt = np.mean(ts1, axis=0) - np.mean(ts2, axis=0)

        #bt=npl.pinv(tr12).dot(y)
        COV12=ts12.T.dot(ts12)/len(ts12)
        COV[ctr]=COV12
        V,EIG,_=npl.svd(COV12)
        SQ=V.dot(np.diag(np.sqrt(EIG)).dot(V.T))
        BT[ctr]=bt
        btc=SQ.dot(bt)
        BTC[ctr]=btc
        ctr+=1
    np.save('BT',BT)
    np.save('COV',COV)

for i in range(NUM):
    BT[i,:]/=npl.norm(BT[i,:])
    BTC[i,:]/=npl.norm(BTC[i,:])

COV_BT=BTC.T.dot(BTC)/NUM
COV_F=np.sum(COV,0)/NUM
EIG_BT=npl.eig(COV_BT)[0]
EIG_F=npl.eig(COV_F)[0]

V,EIG_BT,_=npl.svd(COV_BT)
SQ=V.dot(np.diag(np.sqrt(EIG_BT)).dot(V.T))
PROD=SQ.dot(COV_F).dot(SQ.T)
ID=np.identity(784)
PRODID=SQ.dot(ID).dot(SQ.T)
print('Identity alignment',np.sum(np.real(npl.eig(PRODID)[0]))/npl.norm(EIG_BT)/np.sqrt(784))
print('Canonical alignment',np.sum(np.real(npl.eig(PROD)[0]))/npl.norm(EIG_F)/npl.norm(EIG_BT))

COV_BT=BT.T.dot(BT)/NUM
COV_F=np.sum(COV,0)/NUM
EIG_BT=npl.eig(COV_BT)[0]
EIG_F=npl.eig(COV_F)[0]

V,EIG_BT,_=npl.svd(COV_BT)
SQ=V.dot(np.diag(np.sqrt(EIG_BT)).dot(V.T))
PROD=SQ.dot(COV_F).dot(SQ.T)
ID=np.identity(784)
PRODID=SQ.dot(ID).dot(SQ.T)
print('Identity alignment',np.sum(np.real(npl.eig(PRODID)[0]))/npl.norm(EIG_BT)/np.sqrt(784))
print('Beta alignment',np.sum(np.real(npl.eig(PROD)[0]))/npl.norm(EIG_F)/npl.norm(EIG_BT))
