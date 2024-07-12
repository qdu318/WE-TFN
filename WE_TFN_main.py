import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 读取 h5 文件并返回数据
def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        date = f['date'][:]
        data = f['data'][:]
    return date, data

# 数据归一化
def normalize_data(data):
    scaler = MinMaxScaler()  # 使用 MinMaxScaler 进行归一化
    data_shape = data.shape
    data = data.reshape(-1, data_shape[-1])
    data = scaler.fit_transform(data)
    return data.reshape(data_shape), scaler
class FeatureComponent(nn.Module):
    def __init__(self, input_dim):
        super(FeatureComponent, self).__init__()
        self.linear = nn.Linear(input_dim, 16)   #由1024 变为 16

    def forward(self, water_demand):
        batch_size, seq_length, input_dim = water_demand.shape   # 7220 2 1024
        water_demand = water_demand.view(batch_size * seq_length, input_dim)  #14440,1024
        features = F.relu(self.linear(water_demand))  #14440, 16
        features = features.view(batch_size, seq_length, 16)  # 7220 2 16
        return features

class TDCL(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes=[2, 4, 8]):
        super(TDCL, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_channels, output_channels, ks, padding=ks//2) for ks in kernel_sizes])
        self.pool = nn.AvgPool1d(kernel_size=2, stride=1)

    def forward(self, x):
        conv_outs = [self.pool(conv(x)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=2)
        out = torch.tanh(out)    # 7220,16,6
        return out

class AttentionalPoolingMechanism(nn.Module):  # 7220, 16, 6     7220 2 16
    def __init__(self, input_dim):
        super(AttentionalPoolingMechanism, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, features):
        batch_size, seq_length, input_dim = features.shape   #7220 2 16
        features = features.view(batch_size * seq_length, input_dim)  #14440,16
        value = self.fc(features)  #14440,1
        value = value.view(batch_size, seq_length, 1)# 7220,2,1
        attention_weights = F.softmax(value, dim=1)
        attention_weights = attention_weights.permute(0, 2, 1)  # 7220,1,2
        x = x.permute(0, 2, 1)  #7220,6,16
        x=x.reshape(7220, 2, 48)
        weighted_features = torch.bmm(attention_weights, x)
        weighted_features = weighted_features.permute(0, 2, 1)  # (7220,48,1)
        return weighted_features


class WE_TFN(nn.Module):
    def __init__(self, input_dim, conv_channels=16, kernel_sizes=[2, 4, 8]):  #input_dim 1024
        super(WE_TFN, self).__init__()
        self.feature_component = FeatureComponent(input_dim)
        self.tdcl = TDCL(16, conv_channels, kernel_sizes)
        self.apm = AttentionalPoolingMechanism(conv_channels)
        self.fc = nn.Linear(24, input_dim)  # 输出维度调整为输入维度

    def forward(self, water_demand):
        features = self.feature_component(water_demand)
        features = features.permute(0, 2, 1)  # 7220 16 2
        temporal_features = self.tdcl(features)  # 7220, 16, 6
        context_features = self.apm(temporal_features, features.permute(0, 2, 1)) # (7220,48,1)
        context_features = context_features.view(features.size(0), 2, 24)  # 调整维度以匹配时间步
        output = self.fc(context_features)  # 调整全连接层的输出维度
        return output

# 读取数据文件
file_path = 'E:\PythonProject\WE-TFN\data\BJ16_M32x32_T30_InOut.h5'
date, data = read_h5_file(file_path)
batch_size, time_steps, height, width = data.shape
data = data.reshape(batch_size, time_steps, -1)  # (batch_size, time_steps, height * width)
print(data.shape)   #7220,2,1024
# 创建目标数据
# 这里假设目标是预测下一时间步的水需求，因此 target 是 data 的右移一个时间步
target = np.roll(data, shift=-1, axis=1)  # 右移一个时间步    7220,2,1024
target[:, -1, :] = 0  # 最后一个时间步没有下一步数据，填充为0

# 将数据转换为张量
water_demand = torch.tensor(data, dtype=torch.float32) #7220,2,1024
target = torch.tensor(target, dtype=torch.float32)   #7220,2,1024

input_dim = water_demand.shape[2]  #1024
model = WE_TFN(input_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 前向传播
output = model(water_demand)
print(output.shape)
print(target.shape)
# 计算损失
loss = criterion(output, target[:, :-1, :])
# 计算 RMSE
rmse = torch.sqrt(loss)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Output shape: {output.shape}')
print(f'Loss: {loss.item()}')
print(f'RMSE: {rmse.item()}')
