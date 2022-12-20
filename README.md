# custom_pytorch

Custom Framework for Training Neural Network.
Based on numpy, the interface of the framework replicates PyTorch.

Analogs of the following methods were implemented:
1. Layers:
- torch.nn.Linear
- torch.nn.BatchNormalization
- torch.nn.Dropout
- torch.nn.Sequential
2. Activations:
- torch.nn.ReLU
- torch.nn.Sigmoid
- torch.nn.Softmax
- torch.nn.LogSoftmax
3. Criterions:
- torch.nn.MSELoss
- torch.nn.CrossEntropyLoss
4. Optimizers:
- torch.optim.SGD
- torch.optim.Adam
5. DataLoader:
- torch.utils.data.DataLoader


