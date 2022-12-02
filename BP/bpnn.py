import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split #训练集，测试集划分函数
import torch
import torch.nn.functional as Fun
import pandas as pd
 
#设置超参数
l=0.02 #学习率
epochs=400 #训练轮数
n_feature=5 #输入特征
n_hidden=[15,15,15,15] #隐层节点数
n_output=1 #输出(富营养化指数)
 
#1.准备数据
data = pd.read_excel("D:/data2015-2018浙江省实测富营养化参数数据.xlsx", sheet_name = 0)
data = data[['lon','lat','水深','溶解氧','盐度','磷酸盐','WJN','COD']]
data['富营养化指数']=list(map(lambda COD,DIN,DIP: COD*DIN*DIP*1000.0/4.5, data['COD'], data['WJN'],data['磷酸盐']))
data_positive = data[data.select_dtypes(include=[np.number]).ge(0).all(1)] #删除所有带负值的行
#过滤及其异常值
abnormal_up = data_positive['富营养化指数'].mean() + 5* data_positive['富营养化指数'].std()
data_positive = data_positive[data_positive['富营养化指数']<abnormal_up]

#设置训练集数据80%，测试集20%
x_train0,x_test0,y_train,y_test=train_test_split(data_positive[['lon','lat','盐度','水深','溶解氧']].to_numpy(),data_positive[['富营养化指数']].to_numpy(),test_size=0.2,random_state=22)
#归一化(也就是所说的min-max标准化)通过调用sklearn库的标准化函数
min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train0)
x_test = min_max_scaler.fit_transform(x_test0)

#将数据类型转换为tensor方便pytorch使用
x_train=torch.FloatTensor(x_train)
y_train=torch.FloatTensor(y_train)
x_test=torch.FloatTensor(x_test)
y_test=torch.FloatTensor(y_test)
 
#2.定义BP神经网络
class BPNetModel(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(BPNetModel, self).__init__()
        self.hiddden1=torch.nn.Linear(n_feature,n_hidden[0])#定义隐层网络
        self.hiddden2=torch.nn.Linear(n_hidden[0],n_hidden[1])
        self.hiddden3=torch.nn.Linear(n_hidden[1],n_hidden[2])
        self.hiddden4=torch.nn.Linear(n_hidden[2],n_hidden[3])
        self.out=torch.nn.Linear(n_hidden[3],n_output)#定义输出层网络
    def forward(self,x):
        x=Fun.relu(self.hiddden1(x)) #隐层激活函数采用relu()函数
        x=Fun.relu(self.hiddden2(x))
        x=Fun.relu(self.hiddden3(x))
        x=Fun.relu(self.hiddden4(x))
        out=self.out(x) #Fun.softmax(self.out(x),dim=1) #输出层采用softmax函数
        return out
#3.定义优化器和损失函数
net=BPNetModel(n_feature=n_feature,n_hidden=n_hidden,n_output=n_output) #调用网络
optimizer=torch.optim.Adam(net.parameters(),lr=l) #使用Adam优化器，并设置学习率
loss_fun=torch.nn.MSELoss() #使用MSE损失函数
 
#4.训练数据
loss_steps=np.zeros(epochs) #构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
accuracy_steps=np.zeros(epochs)
 
for epoch in range(epochs):
    y_pred=net(x_train) #前向传播
    loss=loss_fun(y_pred,y_train)#预测值和真实值对比
    optimizer.zero_grad() #梯度清零
    loss.backward() #反向传播
    optimizer.step() #更新梯度
    loss_steps[epoch]=loss.item()#保存loss
    running_loss = loss.item()
    print(f"第{epoch}次训练, loss={running_loss}".format(epoch,running_loss))
    with torch.no_grad(): #下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        y_pred=net(x_test)
        correct= r2_score(y_test,y_pred)
        accuracy_steps[epoch]=correct
        print("R² = ", accuracy_steps[epoch])

#5.绘制损失函数和精度
fig_name="EI_BPNet"
fontsize=10
fig,(ax1,ax2)=plt.subplots(2,figsize=(5,4),sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy",fontsize=fontsize)
ax1.set_title(fig_name,fontsize="xx-large")
ax2.plot(loss_steps)
ax2.set_ylabel("train loss",fontsize=fontsize)
ax2.set_xlabel("epochs",fontsize=fontsize)
plt.tight_layout()
#plt.savefig(fig_name+'.png')
plt.show()

plt.plot(list(range(0,len(y_test))), y_test, color = 'blue',alpha=1.0)
plt.plot(list(range(0,len(y_pred))), y_pred, color = 'red',alpha=0.7)
plt.show()