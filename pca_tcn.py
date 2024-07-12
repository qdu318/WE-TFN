import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_data_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = np.array(f['data'])
    return data

file_path = 'E:\PythonProject\TCN-master\BJ16_M32x32_T30_InOut.h5'
data = load_data_from_h5(file_path)
data = data.astype(np.float32)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
x_train, x_test = data[:train_size], data[train_size:]

y_train = x_train[:, 1:]
x_train = x_train[:, :-1]
y_test = x_test[:, 1:]
x_test = x_test[:, :-1]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def apply_pca(x_train, x_test, y_train, y_test, n_components):
    n_samples, seq_length, height, width = x_train.shape
    x_train_flat = x_train.view(n_samples, seq_length * height * width).numpy()
    x_test_flat = x_test.view(x_test.shape[0], seq_length * height * width).numpy()
    y_train_flat = y_train.view(n_samples, seq_length * height * width).numpy()
    y_test_flat = y_test.view(x_test.shape[0], seq_length * height * width).numpy()

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_flat)
    x_test_pca = pca.transform(x_test_flat)
    y_train_pca = pca.fit_transform(y_train_flat)
    y_test_pca = pca.transform(y_test_flat)

    x_train_pca = torch.tensor(x_train_pca, dtype=torch.float32).view(n_samples, n_components, -1)
    x_test_pca = torch.tensor(x_test_pca, dtype=torch.float32).view(x_test.shape[0], n_components, -1)
    y_train_pca = torch.tensor(y_train_pca, dtype=torch.float32).view(n_samples, n_components, -1)
    y_test_pca = torch.tensor(y_test_pca, dtype=torch.float32).view(x_test.shape[0], n_components, -1)

    return x_train_pca, x_test_pca, y_train_pca, y_test_pca

n_components = 30
x_train_pca, x_test_pca, y_train_pca, y_test_pca = apply_pca(x_train, x_test, y_train, y_test, n_components)
print('train_pca.shape和test_pca.shape:', x_train_pca.shape, x_test_pca.shape)
print('train_label_pca.shape和test_label_pca.shape:', y_train_pca.shape, y_test_pca.shape)

num_epochs = 100
learning_rate = 0.001
batch_size = 32
num_channels = [25, 25, 25, 30]  # 最后一层的通道数调整为30，以匹配目标标签的通道数
kernel_size = 2
dropout = 0.2

train_dataset = TensorDataset(x_train_pca, y_train_pca)
test_dataset = TensorDataset(x_test_pca, y_test_pca)
print('train_data.shape和test_data.shape:', len(train_dataset), len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('train_loader.shape和test_loader.shape:', len(train_loader), len(test_loader))

model = TemporalConvNet(num_inputs=n_components, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.view(outputs.size(0), -1))
        actuals.append(targets.view(targets.size(0), -1))

predictions = torch.cat(predictions).numpy()
actuals = torch.cat(actuals).numpy()

mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print(f'Test RMSE: {rmse}')
