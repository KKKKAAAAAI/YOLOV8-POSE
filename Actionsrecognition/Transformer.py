import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 定义Transformer编码器模型类
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.positional_encoding = PositionalEncoding(input_size, d_model, dropout_rate)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, num_heads, dropout_rate) for _ in range(num_layers)])
        self.output_fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 输入x的形状：(batch_size, seq_len, input_size)
        x = self.positional_encoding(x)
        for i in range(self.num_layers):
            x, _ = self.transformer_layers[i](x)
        # 取序列中最后一个时间步的输出作为模型的输出
        output = self.output_fc(x[:, -1, :])
        return output

# 定义位置编码器模块类
class PositionalEncoding(nn.Module):
    def __init__(self, input_size, d_model, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # 计算位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)

        # 添加到编码器参数中
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # 输入x的形状：(batch_size, seq_len, input_size)
        x = x + self.pos_enc[:x.size(1), :]
        return self.dropout(x)

# 定义Transformer层模块类
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # 输入x的形状：(batch_size, seq_len, d_model)
        residual = x
        # 多头自注意力层
        x = self.layer_norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = self.dropout1(x)
        x += residual
        # 前馈神经网络层
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x += residual
        return x, _

# 定义多头自注意力层模块类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.scale = math.sqrt(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def split_heads(self, x):
        # 输入x的形状：(batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        # 输入query的形状：(batch_size, seq_len, d_model)
        # 输入key的形状：(batch_size, seq_len, d_model)
        # 输入value的形状：(batch_size, seq_len, d_model)
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 计算注意力矩阵
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)

        # 输出形状：(batch_size, seq_len, d_model)
        output = self.output_projection(context)
        return output, attention_probs

# 定义前馈神经网络层模块类
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # 输入x的形状：(batch_size, seq_len, d_model)
        x = nn.ReLU(self.linear1(x))
        x = self.dropout(x)
        # 输出形状：(batch_size, seq_len, d_model)
        x = self.linear2(x)
        return x

# 定义数据集类
class SkeletonDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.labels = self.data[:, -1]
        self.data = self.data[:, :-1]
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).type(torch.float32)
        label = torch.tensor(self.labels[index]).type(torch.long)
        return data, label

# 定义训练函数
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc

# 定义测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_acc += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc

if __name__ == '__main__':
    # 设定超参数
    input_size = 18
    d_model = 512
    num_layers = 6
    num_heads = 8
    dropout_rate = 0.2
    num_epochs = 10
    batch_size = 256
    learning_rate = 0.001

    # 读入数据
    train_loader = DataLoader(SkeletonDataset('train_data.npy'), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SkeletonDataset('test_data.npy'), batch_size=batch_size, shuffle=False)

    # 定义设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerEncoder(input_size, d_model, num_layers, num_heads, dropout_rate).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练并测试模型
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        print('Epoch {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, train_loss, train_acc, test_loss, test_acc))