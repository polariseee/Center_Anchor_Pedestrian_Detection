import torch
from sklearn import preprocessing


a = torch.Tensor([[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],
                  [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0,17.0,18.0]],
                  [[19.0,20.0,21.0],[22.0,23.0,24.0],[25.0,26.0,27.0]]])
print(a)
print(a.size())
# b = preprocessing.normalize(a, norm='l2', axis=0)
# print(b)

norm = torch.norm(a, dim=0)
output = torch.div(a, norm)
print(norm)
print(output)