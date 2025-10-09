import marimo

__generated_with = "0.16.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Daily Dose of Data Science

    This notebook accompanies the code for our model compression blog.

    Read the full blog here: [Machine Learning Model Compression: A Critical Step Towards Efficient Deep Learning](https://www.dailydoseofds.com/model-compression-a-critical-step-towards-efficient-machine-learning)

    Author: Avi Chawla
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports""")
    return


@app.cell
def _():
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

    from time import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    return (
        DataLoader,
        F,
        nn,
        np,
        optim,
        pd,
        sys,
        time,
        torch,
        torchvision,
        tqdm,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load the MNIST dataset""")
    return


@app.cell
def _(DataLoader, torchvision, transforms):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return testloader, trainloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Knowledge Distillation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Teacher Model""")
    return


@app.cell
def _(F, nn):
    class TeacherNet(nn.Module):
        def __init__(self):
            super(TeacherNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 5)
            self.pool = nn.MaxPool2d(5, 5)
            self.fc1 = nn.Linear(32 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (TeacherNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluation function""")
    return


@app.cell
def _(testloader, torch):
    def evaluate(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _data in testloader:
                _inputs, _labels = _data
                _outputs = model(_inputs)
                _, predicted = torch.max(_outputs.data, 1)
                total = total + _labels.size(0)
                correct = correct + (predicted == _labels).sum().item()
        return correct / total
    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize and train the teacher model""")
    return


@app.cell
def _(TeacherNet, nn, optim):
    teacher_model = TeacherNet()
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    teacher_criterion = nn.CrossEntropyLoss()
    return teacher_criterion, teacher_model, teacher_optimizer


@app.cell
def _(
    evaluate,
    teacher_criterion,
    teacher_model,
    teacher_optimizer,
    trainloader,
):
    for _epoch in range(5):
        teacher_model.train()
        _running_loss = 0.0
        for _data in trainloader:
            _inputs, _labels = _data
            teacher_optimizer.zero_grad()
            _outputs = teacher_model(_inputs)
            _loss = teacher_criterion(_outputs, _labels)
            _loss.backward()
            teacher_optimizer.step()
            _running_loss = _running_loss + _loss.item()
        teacher_accuracy = evaluate(teacher_model)
        print(f'Epoch {_epoch + 1}, Loss: {_running_loss / len(trainloader)}, Accuracy: {teacher_accuracy * 100:.2f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Student Model""")
    return


@app.cell
def _(F, nn):
    class StudentNet(nn.Module):
        def __init__(self):
            super(StudentNet, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (StudentNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize and train the teacher model""")
    return


@app.cell
def _(StudentNet, optim):
    student_model = StudentNet()
    student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    return student_model, student_optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Loss function (KL Divergence)""")
    return


@app.cell
def _(F):
    def knowledge_distillation_loss(student_logits, teacher_logits):
        p_teacher = F.softmax(teacher_logits, dim=1)
        p_student = F.log_softmax(student_logits, dim=1)
        _loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
        return _loss
    return (knowledge_distillation_loss,)


@app.cell
def _(
    evaluate,
    knowledge_distillation_loss,
    student_model,
    student_optimizer,
    teacher_model,
    testloader,
    trainloader,
):
    for _epoch in range(5):
        student_model.train()
        _running_loss = 0.0
        for _data in trainloader:
            _inputs, _labels = _data
            student_optimizer.zero_grad()
            student_logits = student_model(_inputs)
            teacher_logits = teacher_model(_inputs).detach()
            _loss = knowledge_distillation_loss(student_logits, teacher_logits)
            _loss.backward()
            student_optimizer.step()
            _running_loss = _running_loss + _loss.item()
        student_accuracy = evaluate(student_model)
        print(f'Epoch {_epoch + 1}, Loss: {_running_loss / len(testloader)}, Accuracy: {student_accuracy * 100:.2f}%')
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit evaluate(teacher_model)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit evaluate(student_model) # student model runs faster
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Zero-Pruning""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model""")
    return


@app.cell
def _(nn, torch):
    # Define a simple neural network for MNIST classification
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    return (SimpleNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluation function""")
    return


@app.cell
def _(testloader, torch):
    def evaluate_1(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _data in testloader:
                _inputs, _labels = _data
                _outputs = model(_inputs)
                _, predicted = torch.max(_outputs.data, 1)
                total = total + _labels.size(0)
                correct = correct + (predicted == _labels).sum().item()
        return correct / total
    return (evaluate_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize and train the neural network""")
    return


@app.cell
def _(SimpleNet, evaluate_1, nn, optim, trainloader):
    net = SimpleNet()
    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.Adam(net.parameters(), lr=0.001)
    for _epoch in range(5):
        _running_loss = 0.0
        net.train()
        for _data in trainloader:
            _inputs, _labels = _data
            _optimizer.zero_grad()
            _outputs = net(_inputs.view(-1, 28 * 28))
            _loss = _criterion(_outputs, _labels)
            _loss.backward()
            _optimizer.step()
        _running_loss = _running_loss + _loss.item()
        _accuracy = evaluate_1(net)
        print(f'Epoch {_epoch + 1}, Loss: {_running_loss / len(trainloader)}, Accuracy: {_accuracy * 100:.2f}%')
    return (net,)


@app.cell
def _(net, torch):
    # Save model
    torch.save(net, "net.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define the pruning threshold (λ) and apply Pruning""")
    return


@app.cell
def _(evaluate_1, net, np, torch, tqdm):
    _thresholds = np.linspace(0, 0.1, 11)
    results = []
    total_params = np.sum([_param.numel() for _name, _param in net.named_parameters() if 'weight' in _name])
    for _threshold in tqdm(_thresholds):
        for _name, _param in net.named_parameters():
            if 'weight' in _name:
                _param.data[torch.abs(_param.data) < _threshold] = 0
        zero_params = np.sum([torch.sum(_param == 0).item() for _name, _param in net.named_parameters() if 'weight' in _name])
        _accuracy = evaluate_1(net)
        results.append([_threshold, _accuracy, total_params, zero_params])
    return (results,)


@app.cell
def _(pd, results):
    results_1 = pd.DataFrame(results, columns=['Threshold', 'Accuracy', 'Original Params', 'Zero Params'])
    results_1['Zero percentage'] = 100 * results_1['Zero Params'] / results_1['Original Params']
    results_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Select best threshold and apply pruning""")
    return


@app.cell
def _(torch):
    _threshold = 0.03
    net_1 = torch.load('net.pt')
    for _name, _param in net_1.named_parameters():
        if 'weight' in _name:
            _param.data[torch.abs(_param.data) < _threshold] = 0
    return (net_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Represent as sparse matrix""")
    return


@app.cell
def _(net_1):
    import scipy.sparse as sp
    sparse_weights = []
    for _name, _param in net_1.named_parameters():
        if 'weight' in _name:
            np_weight = _param.data.cpu().numpy()
            sparse_weights.append(sp.csr_matrix(np_weight))
    return (sparse_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Size before pruning""")
    return


@app.cell
def _(net_1):
    _total_size = 0
    for _name, _param in net_1.named_parameters():
        if 'weight' in _name:
            tensor = _param.data
            _total_size = _total_size + tensor.element_size() * tensor.numel()
    tensor_size_mb = _total_size / 1024 ** 2
    tensor_size_mb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Size after pruning""")
    return


@app.cell
def _(sparse_weights):
    _total_size = 0
    for w in sparse_weights:
        _total_size = _total_size + w.data.nbytes
    csr_size_mb = _total_size / 1024 ** 2
    csr_size_mb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Activation pruning""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model""")
    return


@app.cell
def _(SimpleNet, nn, torch):
    class SimpleNet_1(nn.Module):

        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x1 = torch.relu(self.fc1(x))
            x2 = torch.relu(self.fc2(x1))
            x3 = torch.relu(self.fc3(x2))
            x4 = self.fc4(x3)
            return (x1, x2, x3, x4)
    return (SimpleNet_1,)


@app.cell
def _(testloader, torch):
    def evaluate_2(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _data in testloader:
                _inputs, _labels = _data
                _outputs = model(_inputs)[-1]
                _, predicted = torch.max(_outputs.data, 1)
                total = total + _labels.size(0)
                correct = correct + (predicted == _labels).sum().item()
        return correct / total
    return (evaluate_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize and train the neural network""")
    return


@app.cell
def _(SimpleNet_1, evaluate_2, nn, optim, trainloader):
    net_2 = SimpleNet_1()
    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.Adam(net_2.parameters(), lr=0.001)
    for _epoch in range(5):
        net_2.train()
        _running_loss = 0.0
        for _data in trainloader:
            _inputs, _labels = _data
            _optimizer.zero_grad()
            _outputs = net_2(_inputs)
            _loss = _criterion(_outputs[-1], _labels)
            _loss.backward()
            _optimizer.step()
            _running_loss = _running_loss + _loss.item()
        _accuracy = evaluate_2(net_2)
        print(f'Epoch {_epoch + 1}, Loss: {_running_loss / len(trainloader)}, Accuracy: {_accuracy * 100:.2f}%')
    return (net_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluate the average activations of neurons on the training data""")
    return


@app.cell
def _(net_2, torch, trainloader):
    net_2.eval()
    all_activations = [torch.zeros(512), torch.zeros(256), torch.zeros(128)]
    data_size = len(trainloader.dataset.targets)
    with torch.no_grad():
        for _data in trainloader:
            _inputs, _ = _data
            activations_fc1 = torch.relu(net_2.fc1(_inputs.view(-1, 28 * 28)))
            activations_fc2 = torch.relu(net_2.fc2(activations_fc1))
            activations_fc3 = torch.relu(net_2.fc3(activations_fc2))
            all_activations[0] = all_activations[0] + torch.sum(activations_fc1, dim=0)
            all_activations[1] = all_activations[1] + torch.sum(activations_fc2, dim=0)
            all_activations[2] = all_activations[2] + torch.sum(activations_fc3, dim=0)
    for idx, activations in enumerate(all_activations):
        all_activations[idx] = activations / data_size
    return (all_activations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Apply activation pruning across thresholds""")
    return


@app.cell
def _(SimpleNet_1, all_activations, evaluate_2, net_2, nn, np, time, tqdm):
    _thresholds = np.linspace(0, 1, 11)
    results_2 = []
    _original_total_params = sum((p.numel() for p in net_2.parameters()))
    for _threshold in tqdm(_thresholds):
        new_net = SimpleNet_1()
        new_net.fc1.weight = net_2.fc1.weight
        new_net.fc2.weight = net_2.fc2.weight
        new_net.fc3.weight = net_2.fc3.weight
        new_net.fc4.weight = net_2.fc4.weight
        new_net.fc1.bias = net_2.fc1.bias
        new_net.fc2.bias = net_2.fc2.bias
        new_net.fc3.bias = net_2.fc3.bias
        new_net.fc4.bias = net_2.fc4.bias
        new_net.fc1.weight = nn.Parameter(new_net.fc1.weight[all_activations[0] >= _threshold])
        new_net.fc2.weight = nn.Parameter(new_net.fc2.weight[:, all_activations[0] >= _threshold])
        new_net.fc2.weight = nn.Parameter(new_net.fc2.weight[all_activations[1] >= _threshold])
        new_net.fc3.weight = nn.Parameter(new_net.fc3.weight[:, all_activations[1] >= _threshold])
        new_net.fc3.weight = nn.Parameter(new_net.fc3.weight[all_activations[2] >= _threshold])
        new_net.fc4.weight = nn.Parameter(new_net.fc4.weight[:, all_activations[2] >= _threshold])
        new_net.fc1.bias = nn.Parameter(new_net.fc1.bias[all_activations[0] >= _threshold])
        new_net.fc2.bias = nn.Parameter(new_net.fc2.bias[all_activations[1] >= _threshold])
        new_net.fc3.bias = nn.Parameter(new_net.fc3.bias[all_activations[2] >= _threshold])
        _accuracies = 0
        _total_time = 0
        for _ in range(7):
            _start = time()
            _accuracies = _accuracies + evaluate_2(new_net)
            _total_time = _total_time + (time() - _start)
        _new_total_params = sum((p.numel() for p in new_net.parameters()))
        results_2.append([_threshold, 100 * _accuracies / 7, _original_total_params, _new_total_params, _total_time / 7])
    return (results_2,)


@app.cell
def _(pd, results_2):
    results_3 = pd.DataFrame(results_2, columns=['Threshold', 'Accuracy', 'Original Params', 'New Params', 'Inference Time'])
    results_3['Size Reduction'] = 1 - results_3['New Params'] / results_3['Original Params']
    results_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Low-rank Factorization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model""")
    return


@app.cell
def _(SimpleNet_1, nn, torch):
    class SimpleNet_2(nn.Module):

        def __init__(self):
            super(SimpleNet_1, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x1 = torch.relu(self.fc1(x))
            x2 = torch.relu(self.fc2(x1))
            x3 = torch.relu(self.fc3(x2))
            x4 = self.fc4(x3)
            return x4
    return (SimpleNet_2,)


@app.cell
def _(testloader, torch):
    def evaluate_3(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _data in testloader:
                _inputs, _labels = _data
                _outputs = model(_inputs)
                _, predicted = torch.max(_outputs.data, 1)
                total = total + _labels.size(0)
                correct = correct + (predicted == _labels).sum().item()
        return correct / total
    return (evaluate_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Initialize and train the neural network""")
    return


@app.cell
def _(SimpleNet_2, evaluate_3, nn, optim, trainloader):
    net_3 = SimpleNet_2()
    _criterion = nn.CrossEntropyLoss()
    _optimizer = optim.Adam(net_3.parameters(), lr=0.001)
    for _epoch in range(5):
        net_3.train()
        _running_loss = 0.0
        for _data in trainloader:
            _inputs, _labels = _data
            _optimizer.zero_grad()
            _outputs = net_3(_inputs)
            _loss = _criterion(_outputs, _labels)
            _loss.backward()
            _optimizer.step()
            _running_loss = _running_loss + _loss.item()
        _accuracy = evaluate_3(net_3)
        print(f'Epoch {_epoch + 1}, Loss: {_running_loss / len(trainloader)}, Accuracy: {_accuracy * 100:.2f}%')
    return (net_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Function to determine minimum matrix operations

    Taken from https://www.geeksforgeeks.org/python-program-for-matrix-chain-multiplication-dp-8/
    """
    )
    return


@app.cell
def _(sys):
    def MatrixChainOrder(p, i, j):

        if i == j:
            return 0

        _min = sys.maxsize

        for k in range(i, j):

            count = (MatrixChainOrder(p, i, k)
                 + MatrixChainOrder(p, k + 1, j)
                       + p[i-1] * p[k] * p[j])

            if count < _min:
                _min = count;

        return _min;
    return (MatrixChainOrder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Apply LRF across thresholds""")
    return


@app.cell
def _(MatrixChainOrder, evaluate_3, net_3, nn, time, torch, tqdm):
    rank_values = [128, 100, 90, 80, 60, 50, 40, 30, 20, 10, 5, 2, 1]
    results_4 = []
    _original_total_params = 128 * 256
    batch_size = 32
    U, S, V = torch.svd(net_3.fc3.weight)
    for rank in tqdm(rank_values):
        U_low_rank = U[:, :rank]
        S_low_rank = torch.diag(S[:rank])
        V_low_rank = V[:, :rank]
        factorized_weight_matrix = torch.mm(U_low_rank, torch.mm(S_low_rank, V_low_rank.t()))
        net_3.fc3.weight = nn.Parameter(factorized_weight_matrix)
        weight_list = [batch_size, 256, rank, rank, 128]
        if rank == 128:
            total_operations = batch_size * 256 * 128
        else:
            total_operations = MatrixChainOrder(weight_list, 1, 4)
        _accuracies = 0
        _total_time = 0
        for _ in range(7):
            _start = time()
            _accuracies = _accuracies + evaluate_3(net_3)
            _total_time = _total_time + (time() - _start)
        _new_total_params = 128 * rank + rank ** 2 + rank * 256
        results_4.append([rank, 100 * _accuracies / 7, _original_total_params, _new_total_params, total_operations, _total_time / 7])
    return (results_4,)


@app.cell
def _(pd, results_4):
    results_5 = pd.DataFrame(results_4, columns=['Threshold', 'Accuracy', 'Original Params', 'New Params', 'Operations', 'Inference Time'])
    results_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Quantization""")
    return


@app.cell
def _(net_3, torch):
    quantized_model = torch.quantization.quantize_dynamic(net_3, {torch.nn.Linear}, dtype=torch.qint8)
    return (quantized_model,)


@app.cell
def _(evaluate_3, quantized_model):
    evaluate_3(quantized_model)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
