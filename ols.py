import torch
import matplotlib as plt
import numpy as np
import torch.nn as nn
import pandas as pd

data = pd.read_excel("C://Users//86134//Desktop//SRTP//data2015-2018浙江省实测富营养化参数数据.xlsx", sheet_name=0)
data = data[['叶绿素a', '氨', '总有机碳', 'WJN', '溶解氧', '盐度', '磷酸盐', '总氮', '总磷', 'COD']]
data['富营养化指数'] = list(
    map(lambda COD, DIN, DIP: round(COD * DIN * DIP * 1000.0 / 4.5, 6), data['COD'], data['总氮'], data['总磷']))
data_positive = data[data.select_dtypes(include=[np.number]).ge(0).all(1)]
x = []
x1 = list(data_positive["总氮"])
x2 = list(data_positive["总有机碳"])
x3 = list(data_positive["总磷"])
x4 = list(data_positive["溶解氧"])

y = list(data_positive["富营养化指数"])
for i in range(len(x1)):
    x.append(torch.Tensor([x1[i], x2[i], x3[i],x4[i]]))
x = torch.tensor([it.cpu().detach().numpy() for it in x])
y = torch.Tensor(y)
y = torch.unsqueeze(y, dim=1)

# 用WJN,磷酸盐,COD作为自变量计算出的Loss非常大，在学习率为0.1的情况下SDG跑137000次都还有125754
# 用盐度、水深、溶解氧作为自变量计算出的loss也非常大，在学习率0.001的情况下SDG跑194000次还有150618
# 总氮、总磷、总有机碳作为自变量计算出的Loss相对小一点，在学习率0.01的情况下SDG跑71700次稳定在60448
# 总氮、总有机碳、总磷、溶解氧作为自变量计算的loss在0.01学习率下，SDG跑20000次稳定在54670
#


class LinearRegression(nn.Module):
    def __init__(self, inSize, outSize):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(inSize, outSize)  # 输入输出维度

    def forward(self, x):
        predict = self.linear(x)
        return predict


learnRate = 1e-2
outDim = 1
inDim = 4
thisModel = LinearRegression(inDim, outDim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(thisModel.parameters(), lr=learnRate)
time = 200
epoch = 0
while True:
    y_predict = thisModel(x)
    loss = criterion(y_predict, y)
    if epoch % 100 == 0:
        print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if loss.item() < 10:
        break
# print("y = {}*x + {}".format(thisModel.linear.weight.item(), thisModel.linear.bias.item()))
