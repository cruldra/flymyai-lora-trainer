import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 模型压缩：高效机器学习的关键步骤

    四种减少模型占用空间和推理时间的关键方法。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 动机

    训练机器学习（ML）模型经常被追求更高准确率的无情追求所驱动。

    许多人创建越来越复杂的深度学习模型，这些模型在"性能方面"无疑表现得非常出色。

    然而，复杂性严重影响了它们的现实世界实用性。

    ![模型复杂性与实用性](https://www.dailydoseofds.com/content/images/2023/09/image-133.png)

    **多年来，模型开发的主要目标一直是实现最佳性能指标。**

    这种做法不幸的是也被许多基于排行榜的竞赛所推广。虽然没有错，但在我看来，这掩盖了关注解决方案现实世界适用性的重要性。

    然而，重要的是要注意，当涉及到在生产（或面向用户）系统中部署这些模型时，焦点从原始准确性转向效率、速度和资源消耗等考虑因素。

    因此，通常当我们将任何模型部署到生产环境时，发布到生产环境的特定模型并不仅仅基于性能来确定。

    ![部署考虑因素](https://www.dailydoseofds.com/content/images/2023/09/image-134.png)

    相反，我们必须考虑几个与ML无关的运营指标。

    **它们是什么？让我们了解一下！**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 典型运营指标

    当模型部署到生产环境时，必须满足某些要求。

    通常，这些"要求"在模型的原型设计阶段不被考虑。

    例如，可以合理假设面向用户的模型可能必须处理来自模型集成的产品/服务的大量请求。

    当然，我们永远不能要求用户等待，比如说，一分钟让模型运行并生成预测。

    因此，除了"模型性能"之外，我们还希望优化几个其他运营指标：

    ### 1）推理延迟

    这是模型处理单个输入并生成预测所需的时间。

    它测量向模型发送请求和接收响应之间的延迟。

    ![推理延迟](https://www.dailydoseofds.com/content/images/2023/09/image-135.png)

    努力实现低推理延迟对于所有实时或交互式应用程序都至关重要，因为用户期望快速响应。

    正如您可能猜到的，高延迟将导致糟糕的用户体验，并且不适合许多应用程序，如：

    - 聊天机器人
    - 实时语音转文本转录
    - 游戏等等

    ### 2）吞吐量

    吞吐量是模型在给定时间段内可以处理的推理请求数量。

    它估计模型同时处理多个请求的能力。

    ![吞吐量](https://www.dailydoseofds.com/content/images/2023/09/image-136.png)

    再次，正如您可能猜到的，高吞吐量对于具有大量传入请求的应用程序至关重要。

    这些包括电子商务网站、推荐系统、社交媒体平台等。高吞吐量确保模型可以同时为许多用户提供服务，而不会出现显著延迟。

    ### 3）模型大小

    这指的是模型在加载用于推理目的时占用的内存量。

    它量化了存储模型进行预测或生成实时输出所需的所有参数、配置和相关数据所需的内存占用。

    ![模型大小](https://www.dailydoseofds.com/content/images/2023/09/image-138.png)

    模型大小的重要性在资源受限环境中部署模型时变得特别明显。

    许多生产环境，如移动设备、边缘设备或物联网设备，具有有限的内存容量。

    显然可以猜测，在这种情况下，模型的大小将直接影响它是否可以部署。

    ![资源受限环境](https://www.dailydoseofds.com/content/images/2023/09/image-139.png)

    大型模型可能无法适应可用内存，使它们对这些资源受限的设置不切实际。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 现实案例

    这是一个著名的故事。

    2006年，Netflix推出了"Netflix奖"，这是一个机器学习竞赛，鼓励ML工程师构建最佳算法来预测用户对电影的评分。

    **大奖是100万美元。**

    竞赛结束后，Netflix在2009年向一个开发团队颁发了100万美元奖金，因为他们的算法将公司推荐引擎的准确性提高了**10%**。

    这是很多的！

    然而，Netflix从未使用过该解决方案，因为它过于复杂。

    Netflix是这样说的：

    > 获胜改进的准确性增加似乎不足以证明将它们带入生产环境所需的工程努力是合理的。

    开发模型的复杂性和资源需求使其对现实世界部署不切实际。Netflix面临几个挑战：

    1. **可扩展性：** 该模型不容易扩展以处理Netflix平台上的大量用户和电影。它需要大量的计算资源来为他们拥有的数百万用户提供实时推荐。
    2. **维护：** 在生产环境中管理和更新如此复杂的模型将是一个后勤噩梦。对模型的频繁更新和更改将难以实施和维护。
    3. **延迟：** 集成模型的推理延迟远非流媒体服务的理想选择。用户期望近乎即时的推荐，但模型的复杂性使实现低延迟变得困难。

    > 您可以在这里阅读更多关于这个故事的信息：[Netflix奖故事](https://www.wired.com/2012/04/netflix-prize-costs/?ref=dailydoseofds.com)。

    因此，Netflix从未将获胜解决方案集成到其生产推荐系统中。相反，他们继续使用现有算法的简化版本，这对实时推荐更实用。

    Netflix奖的这个现实案例提醒我们，我们必须努力在模型复杂性和实际效用之间取得微妙的平衡。

    ---

    **虽然高度复杂的模型可能在研究和竞赛环境中表现出色，但由于可扩展性、维护和延迟问题，它们可能不适合现实世界的部署。**

    在实践中，更简单、更高效的模型通常是在生产环境中提供无缝用户体验的更好选择。

    让我问你这个。以下两个模型中，你更愿意将哪个集成到面向用户的产品中？

    ![模型选择](https://www.dailydoseofds.com/content/images/2023/09/image-141.png)

    我强烈偏好模型B。

    如果你理解这一点，你就与在生产中保持简单的想法产生共鸣。

    幸运的是，有各种技术可以帮助我们减少模型的大小，从而提高模型推理的速度。

    这些技术被称为**模型压缩**方法。

    使用这些技术，您可以减少原始模型的延迟和大小。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 模型压缩

    顾名思义，模型压缩是一组用于减少模型大小和计算复杂性同时保持或甚至改善其性能的技术。

    ![模型压缩](https://www.dailydoseofds.com/content/images/2023/09/image-142.png)

    它们旨在使模型更小——这就是为什么名称是"**模型压缩**"。

    通常，预期较小的模型将：

    - 具有较低的推理延迟，因为较小的模型可以提供更快的预测，使它们非常适合实时或低延迟应用程序。
    - 由于其减少的计算需求而易于扩展。
    - 具有较小的内存占用。

    在本文中，我们将看看四种帮助我们实现这一目标的技术：

    ![模型压缩技术](https://www.dailydoseofds.com/content/images/2023/09/image-143.png)

    1. 知识蒸馏
    2. 剪枝
    3. 低秩分解
    4. 量化

    正如我们很快将看到的，这些技术试图在模型大小和准确性之间取得平衡，使在面向用户的产品中部署模型相对更容易。

    让我们逐一了解它们！
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 知识蒸馏

    这是减少模型大小最常见、有效、可靠和我最喜欢的技术之一。

    本质上，知识蒸馏涉及训练一个较小、更简单的模型（称为"学生"模型）来模仿较大、更复杂模型（称为"教师"模型）的行为。

    该术语可以分解如下：

    - **知识：** 指机器学习模型在训练期间获得的理解、洞察或信息。这种"知识"通常可以由模型的参数、学习的模式和其进行预测的能力来表示。

    ![神经网络的知识](https://www.dailydoseofds.com/content/images/2023/09/image-67.png)

    - **蒸馏：** 在这种情况下，蒸馏意味着将知识从一个模型转移或浓缩到另一个模型。它涉及训练学生模型来模仿教师模型的行为，有效地转移教师的知识。

    ![知识蒸馏过程](https://www.dailydoseofds.com/content/images/2023/09/image-68.png)

    这是一个两步过程：

    - 像通常那样训练大模型。这被称为"教师"模型。
    - 训练一个较小的模型，旨在模仿较大模型的行为。这也被称为"学生"模型。

    ![教师学生模型](https://www.dailydoseofds.com/content/images/2023/09/image-144.png)

    知识蒸馏的主要目标是将知识或学习的洞察从教师转移到学生模型。

    这允许学生模型以更少的参数和降低的计算复杂性实现可比较的性能。

    该技术也很直观。

    当然，将其与学术环境中的现实世界师生场景进行比较，学生模型可能永远不会表现得像教师模型那样好。

    但通过持续的训练，我们可以创建一个**几乎**与较大模型一样好的较小模型。

    这回到了我们上面讨论的目标：

    > 在模型大小和准确性之间取得平衡，使在面向用户的产品中部署模型相对更容易。

    以这种方式开发的模型的经典例子是[DistillBERT](https://huggingface.co/docs/transformers/model_doc/distilbert?ref=dailydoseofds.com)。它是[BERT](https://arxiv.org/abs/1810.04805?ref=dailydoseofds.com)的学生模型。

    DistillBERT比BERT小约40%，这在大小上是巨大的差异。

    尽管如此，它保留了BERT约97%的自然语言理解（NLU）能力。

    **更重要的是，DistillBERT在推理中大约快60%。**

    这是我在Transformer模型的一项研究中亲身体验和验证的：

    ![研究结果](https://www.dailydoseofds.com/content/images/2023/09/Screenshot-2023-09-13-at-12.40.42-PM.png)

    如上所示，在其中一个研究数据集（SensEval-2）上，BERT达到了76.81的最佳准确率。使用DistilBERT，准确率为75.64。

    在另一个任务（SensEval-3）上，BERT达到了80.96的最佳准确率。使用DistilBERT，准确率为80.23。

    当然，DistilBERT不如BERT好。然而，性能差异很小。

    考虑到运行时性能优势，在生产环境中使用DistilBERT而不是BERT更有意义。

    💡 如果您有兴趣了解更多关于我的研究，可以在这里阅读：[Transformer在词义消歧上的比较研究](https://arxiv.org/pdf/2111.15417.pdf?ref=dailydoseofds.com)。

    知识蒸馏的最大缺点之一是必须首先训练一个较大的教师模型来训练学生模型。

    然而，在资源受限的环境中，训练大型教师模型可能不可行。

    假设我们至少在开发环境中不受资源限制，知识蒸馏最常见的技术之一是**基于响应的知识蒸馏**。

    顾名思义，在**基于响应的知识蒸馏**中，重点是匹配教师模型和学生模型的输出响应（预测）。

    ![基于响应的知识蒸馏](https://www.dailydoseofds.com/content/images/2023/09/image-149.png)

    谈到分类用例，这种技术将类预测的**概率分布**从教师转移到学生。

    它涉及训练学生不仅产生准确的预测，而且模仿教师模型的软预测（概率分数）。

    由于我们试图模仿**教师模型的类预测的概率分布**，损失函数的理想候选者是KL散度。

    我们在之前关于t-SNE的文章中详细讨论了这一点。

    然而，这里有一个快速回顾：
    [Formulating and Implementing the t-SNE Algorithm From Scratch](https://www.dailydoseofds.com/formulating-and-implementing-the-t-sne-algorithm-from-scratch/)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### KL散度

    KL散度背后的核心思想是评估当一个分布用于近似另一个分布时丢失了多少信息。

    因此，丢失的信息越多，KL散度越大。结果，差异性越大。

    两个概率分布P(x)和Q(x)之间的KL散度计算如下：

    ![KL散度公式](https://www.dailydoseofds.com/content/images/2023/08/image-531.png)

    KL散度公式可以这样理解：

    KL散度DKL(P||Q)测量当使用分布Q来近似P时丢失了多少信息。

    想象一下，如果概率分布P和Q对于所有x都相同，即P(x) = Q(x)，那么：

    ![KL散度为0的情况](https://www.dailydoseofds.com/content/images/2023/08/image-532.png)

    简化后，我们得到：

    ![简化结果](https://www.dailydoseofds.com/content/images/2023/08/image-533.png)

    这正是我们在基于响应的知识蒸馏中打算实现的。

    简单地说，我们希望学生模型的类预测的概率分布与教师模型的类预测的概率分布相同。

    - 首先，我们可以像通常那样训练教师模型。
    - 接下来，我们可以指示学生模型模仿教师模型的类预测的概率分布。

    让我们看看如何使用PyTorch实际使用基于响应的知识蒸馏。

    更具体地说，我们将在MNIST数据集上训练一个稍微复杂的神经网络。然后，我们将使用基于响应的知识蒸馏技术构建一个更简单的神经网络。
    """
    )
    return


@app.cell
def _():
    # 导入必要的库
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    print("导入完成")
    return DataLoader, F, datasets, nn, optim, torch, transforms


@app.cell
def _(DataLoader, datasets, transforms):
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    return test_loader, train_loader


@app.cell
def _(nn, torch):
    # 定义教师模型（CNN）
    class TeacherModel(nn.Module):
        def __init__(self):
            super(TeacherModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    print("教师模型定义完成")
    return (TeacherModel,)


@app.cell
def _(TeacherModel, nn, optim, torch):
    # 初始化教师模型
    teacher_model = TeacherModel()
    teacher_criterion = nn.CrossEntropyLoss()
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)

    print(f"使用设备: {device}")
    print("教师模型初始化完成")
    return device, teacher_criterion, teacher_model, teacher_optimizer


@app.cell
def _(
    device,
    teacher_criterion,
    teacher_model,
    teacher_optimizer,
    train_loader,
):
    # 训练教师模型
    def train_teacher_model(model, train_loader, criterion, optimizer, epochs=5):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f'训练轮次: {epoch+1}/{epochs}, 批次: {batch_idx}, 损失: {loss.item():.6f}')

            print(f'轮次 {epoch+1} 平均损失: {running_loss/len(train_loader):.6f}')

    # 训练教师模型
    print("开始训练教师模型...")
    train_teacher_model(teacher_model, train_loader, teacher_criterion, teacher_optimizer)
    print("教师模型训练完成")
    return


@app.cell(hide_code=True)
def _(device, teacher_model, test_loader, torch):
    # 评估教师模型
    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    teacher_accuracy = evaluate_model(teacher_model, test_loader)
    print(f"教师模型准确率: {teacher_accuracy:.2f}%")
    return evaluate_model, teacher_accuracy


@app.cell
def _(nn):
    # 定义学生模型（简单的前馈网络）
    class StudentModel(nn.Module):
        def __init__(self):
            super(StudentModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(-1, 28*28)  # 展平输入
            x = nn.functional.relu(self.fc1(x))
            x = self.dropout(x)
            x = nn.functional.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    print("学生模型定义完成")
    return (StudentModel,)


@app.cell
def _(F):
    # 定义KL散度损失函数
    def kl_divergence_loss(student_logits, teacher_logits, temperature=3.0):
        """
        计算学生模型和教师模型输出之间的KL散度

        Args:
            student_logits: 学生模型的输出
            teacher_logits: 教师模型的输出
            temperature: 温度参数，用于软化概率分布
        """
        # 使用温度参数软化logits
        student_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

        # 计算KL散度
        kl_div = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

        return kl_div * (temperature ** 2)

    print("KL散度损失函数定义完成")
    return (kl_divergence_loss,)


@app.cell
def _(StudentModel, device, optim):
    # 初始化学生模型
    student_model = StudentModel()
    student_model.to(device)
    student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    print("学生模型初始化完成")
    return student_model, student_optimizer


@app.cell
def _(
    device,
    kl_divergence_loss,
    student_model,
    student_optimizer,
    teacher_model,
    torch,
    train_loader,
):
    # 训练学生模型（知识蒸馏）
    def train_student_model(student, teacher, train_loader, optimizer, epochs=5):
        student.train()
        teacher.eval()  # 教师模型设为评估模式

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                # 获取学生和教师的输出
                student_output = student(data)
                with torch.no_grad():
                    teacher_output = teacher(data)

                # 计算KL散度损失
                loss = kl_divergence_loss(student_output, teacher_output)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f'蒸馏训练轮次: {epoch+1}/{epochs}, 批次: {batch_idx}, 损失: {loss.item():.6f}')

            print(f'蒸馏轮次 {epoch+1} 平均损失: {running_loss/len(train_loader):.6f}')

    # 训练学生模型
    print("开始知识蒸馏训练...")
    train_student_model(student_model, teacher_model, train_loader, student_optimizer)
    print("学生模型训练完成")
    return


@app.cell
def _(evaluate_model, student_model, test_loader):
    # 评估学生模型
    student_accuracy = evaluate_model(student_model, test_loader)
    print(f"学生模型准确率: {student_accuracy:.2f}%")
    return (student_accuracy,)


@app.cell(hide_code=True)
def _(mo, student_accuracy, teacher_accuracy):
    mo.md(
        f"""
    ### 性能比较 — 教师和学生模型

    总结一下，教师模型是基于CNN的神经网络架构。然而，学生模型是一个简单的前馈神经网络。

    **性能对比结果：**

    - **教师模型准确率：** {teacher_accuracy:.2f}%
    - **学生模型准确率：** {student_accuracy:.2f}%

    当然，学生模型的性能不如教师模型，这是预期的。然而，考虑到它只由简单的前馈层组成，它仍然非常接近。

    这就是基于响应的知识蒸馏的工作原理。

    学生模型大约快35%。

    这就是基于响应的知识蒸馏的工作原理。

    总结：

    - 首先，我们像通常那样训练大型（教师）模型。
    - 接下来，我们训练一个较小的（学生）模型，目标是模仿教师模型的类预测的概率分布。

    ![基于响应的知识蒸馏总结](https://www.dailydoseofds.com/content/images/2023/09/image-148.png)

    在上面的演示中，我们看了分类用例的基于响应的知识蒸馏——**其中响应是类概率**。

    然而，我们也可以将基于响应的知识蒸馏用于回归模型。在这种情况下，我们可以使用基于回归的损失函数，例如均方误差（MSE），而不是KL散度损失函数。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 剪枝

    剪枝通常用于基于树的模型，其中它涉及移除分支（或节点）以简化模型。

    ![决策树剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-69.png)

    当然，删除节点会导致模型准确性的下降。

    因此，在决策树的情况下，核心思想是迭代地删除子树，删除后导致：

    - 分类成本的最小增加
    - 复杂性（或节点）的最大减少

    ![剪枝子树](https://www.dailydoseofds.com/content/images/2023/09/image-66.png)

    在上面的图像中，两个子树都导致相同的成本增加。然而，删除具有更多节点的子树以减少计算复杂性更有意义。

    同样的想法也可以转化为神经网络。

    正如您可能猜到的，神经网络中的剪枝涉及识别和消除对模型整体性能贡献最小的特定连接或神经元。

    ![神经网络剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-150.png)

    删除整个层是另一个选择。但我们很少实践它，因为它可能导致权重矩阵不对齐。

    更重要的是，量化特定层对最终输出的贡献并不容易。

    因此，与其删除训练网络的层，不如重新定义架构而不包含我们打算删除的层可能是更好的方法。

    **通过剪枝，目标是创建一个更紧凑的神经网络，同时保留尽可能多的预测能力。**

    这主要通过两种方式完成：

    - **神经元剪枝：**

    ![神经元剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-72.png)

    - 想法是从网络中消除整个**节点**。
    - 结果，表示层的矩阵变小。
    - 这导致更快的推理和更低的内存使用。

    - **权重剪枝：**

    ![权重剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-73.png)

    - 这涉及从网络中消除**边**。
    - 权重剪枝可以被认为是在矩阵中放置零来表示被移除的边。
    - 然而，在这种情况下，矩阵的大小保持不受影响。
    - 因此，矩阵的大小保持相同，但它们变得稀疏。
    - 虽然消除边可能不一定导致更快的推理，但它确实有助于优化内存使用。这是因为稀疏矩阵通常比密集矩阵占用更少的空间。

    通过从网络中删除不重要的权重（或节点），可以期待几个改进：

    - 更好的泛化
    - 改进的推理速度
    - 减少的模型大小

    然而，大问题是：**我们应该剪枝哪些权重和神经元？**

    有几种流行的方法来在神经网络中实现剪枝。让我们讨论它们。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 零剪枝

    这是剪枝神经网络最简单的方法之一，无论是在理解还是实现方面。

    这里，核心思想是基于边权重的大小来移除边。移除边意味着将其权重设置为零。

    ![零剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-75.png)

    如果边权重接近零，这意味着它们可能对神经元的激活贡献很低。因此，我们可以将这些边权重设置为零。

    实现方面，首先，我们像通常那样训练神经网络。

    接下来，我们可以为边权重定义一个阈值（λ），比如λ = 0.02。

    最后，我们将所有低于指定阈值的权重归零，如下所示：
    """
    )
    return


@app.cell(hide_code=True)
def _(torch):
    # 零剪枝实现
    def zero_pruning(model, threshold=0.02):
        """
        对模型进行零剪枝

        Args:
            model: 要剪枝的模型
            threshold: 剪枝阈值
        """
        total_params = 0
        pruned_params = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    total_params += param.numel()
                    # 找到小于阈值的权重
                    mask = torch.abs(param) < threshold
                    pruned_params += mask.sum().item()
                    # 将小于阈值的权重设为零
                    param[mask] = 0.0

        pruning_ratio = pruned_params / total_params * 100
        print(f"剪枝比例: {pruning_ratio:.2f}%")
        print(f"总参数: {total_params}, 剪枝参数: {pruned_params}")

        return pruning_ratio

    print("零剪枝函数定义完成")
    return (zero_pruning,)


@app.cell(hide_code=True)
def _(
    StudentModel,
    device,
    evaluate_model,
    student_model,
    test_loader,
    zero_pruning,
):
    # 创建一个新的学生模型副本进行剪枝测试
    pruned_model = StudentModel()
    pruned_model.load_state_dict(student_model.state_dict())
    pruned_model.to(device)

    # 评估剪枝前的性能
    accuracy_before = evaluate_model(pruned_model, test_loader)
    print(f"剪枝前准确率: {accuracy_before:.2f}%")

    # 进行零剪枝
    pruning_ratio = zero_pruning(pruned_model, threshold=0.02)

    # 评估剪枝后的性能
    accuracy_after = evaluate_model(pruned_model, test_loader)
    print(f"剪枝后准确率: {accuracy_after:.2f}%")
    print(f"准确率变化: {accuracy_after - accuracy_before:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 激活剪枝

    与仅依赖基于阈值的零剪枝不同，神经元激活也可以用作剪枝的标准。

    本质上，当通过网络运行数据集时，我们可以计算和分析各个神经元激活的某些统计信息。

    ![计算神经元激活统计](https://www.dailydoseofds.com/content/images/2023/09/image-152.png)

    在这里，我们可能观察到一些神经元总是输出接近零的值。

    结果，它们很可能被移除，因为它们对模型的输出影响很小。

    ![相似的平均激活](https://www.dailydoseofds.com/content/images/2023/09/image-153.png)

    这也很直观——如果一个神经元很少具有高激活值，那么可以合理假设它没有对模型的输出做出贡献。

    以下是我们需要遵循的步骤来实现激活剪枝：

    - 像通常那样训练神经网络。
    - 训练后，再次通过网络运行训练数据（不反向传播梯度）。
    - 计算隐藏层中每个神经元在整个数据上的平均激活。
    - 定义剪枝阈值λ。
    - 标记激活低于剪枝阈值λ的神经元。
    - 从训练网络的权重矩阵中移除标记的神经元。
    - 完成！
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 冗余剪枝

    有时，如果层中的两个神经元具有非常相似的激活，这可能意味着它们正在捕获相同的特征。

    根据这种直觉，我们可以移除其中一个神经元并保持相同的功能。

    ![冗余剪枝](https://www.dailydoseofds.com/content/images/2023/09/image-154.png)

    虽然这听起来很直观，但这实现起来有点复杂。

    此外，冗余剪枝有时可能是剪枝神经网络的**有点不可靠**的方法。你能回答为什么吗？这是给你的练习。

    尽管如此，我认为了解文献中提出的常见技术仍然是有用的。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 低秩分解

    在其核心，低秩分解旨在使用较低秩的矩阵来近似神经网络的权重矩阵。

    本质上，想法是将复杂的权重矩阵表示为两个或更多简单矩阵的乘积。

    ![矩阵分解为更小的矩阵](https://www.dailydoseofds.com/content/images/2023/09/image-155.png)

    如果我们理解这些个别术语，想法会变得更清楚：

    - **低秩：**
        - 在线性代数中，矩阵的"秩"指的是该矩阵中线性无关行（或列）的最大数量。
        - 因此，低秩矩阵具有较少的线性无关行或列。
        - 这意味着它可以近似为较少数量的基向量的组合，帮助我们减少维度。
    - **分解**
        - 在数学中，"分解"意味着将复杂的数学对象（如矩阵）表达为更简单数学对象的乘积。

    因此，**低秩分解**意味着将给定的权重矩阵分解为两个或更多较低维度矩阵的乘积。

    这些较低维度的矩阵通常被称为"因子矩阵"或"基矩阵"。

    让我们了解它是如何工作的。

    ### 步骤1）执行矩阵分解

    在神经网络中，每一层都有一个权重矩阵。

    我们可以将这些原始权重矩阵分解为低秩近似。

    有许多不同的矩阵分解方法可用，如奇异值分解（SVD）、非负矩阵分解（NMF）或截断SVD。

    ### 步骤2）指定秩

    在矩阵分解中，您通常必须为低秩近似选择秩`k`。

    它确定奇异值的数量（对于SVD）或因子的数量（对于分解方法）。

    ![指定秩](https://www.dailydoseofds.com/content/images/2023/09/image-156.png)

    秩`k`的选择直接与模型大小减少和信息保存之间的权衡相关。

    ### 步骤3）重构权重矩阵

    一旦您获得了低秩矩阵，您可以使用它们来转换输入，而不是原始权重矩阵。

    ![重构权重矩阵](https://www.dailydoseofds.com/content/images/2023/09/image-157.png)

    这样做的好处是它减少了神经网络的计算复杂性，同时保留了训练期间学习的重要特征。

    通过用它们的低秩近似替换原始权重矩阵，我们可以有效地减少模型中的参数数量，从而减少其大小。

    **让我们看看这个实际操作！**

    我们将使用奇异值分解（SVD）将权重矩阵分解为低秩矩阵。
    """
    )
    return


@app.cell
def _(torch):
    # 低秩分解实现
    def low_rank_factorization(weight_matrix, rank):
        """
        使用SVD对权重矩阵进行低秩分解

        Args:
            weight_matrix: 要分解的权重矩阵
            rank: 目标秩

        Returns:
            U, S, V: 分解后的矩阵
        """
        # 执行SVD分解
        U, S, V = torch.svd(weight_matrix)

        # 截断到指定的秩
        U_low_rank = U[:, :rank]
        S_low_rank = torch.diag(S[:rank])
        V_low_rank = V[:, :rank]

        return U_low_rank, S_low_rank, V_low_rank

    def compute_operations(original_shape, rank):
        """
        计算原始矩阵和分解矩阵的操作数

        Args:
            original_shape: 原始矩阵形状 (m, n)
            rank: 分解后的秩

        Returns:
            original_ops, factorized_ops: 操作数
        """
        m, n = original_shape
        batch_size = 32  # 假设批次大小

        # 原始矩阵操作数: batch_size * m * n
        original_ops = batch_size * m * n

        # 分解矩阵操作数: batch_size * (m * rank + rank * rank + rank * n)
        factorized_ops = batch_size * (m * rank + rank * rank + rank * n)

        return original_ops, factorized_ops

    print("低秩分解函数定义完成")
    return compute_operations, low_rank_factorization


@app.cell
def _(compute_operations, low_rank_factorization, student_model):
    # 演示低秩分解
    # 获取学生模型的第一个线性层权重
    fc1_weight = student_model.fc1.weight.data
    print(f"原始权重矩阵形状: {fc1_weight.shape}")

    # 选择不同的秩进行分解
    ranks = [32, 64, 96]

    for rank in ranks:
        # 执行低秩分解
        U, S, V = low_rank_factorization(fc1_weight, rank)

        print(f"\n秩 {rank} 的分解结果:")
        print(f"U 形状: {U.shape}")
        print(f"S 形状: {S.shape}")
        print(f"V 形状: {V.shape}")

        # 计算参数数量
        original_params = fc1_weight.numel()
        factorized_params = U.numel() + S.numel() + V.numel()

        print(f"原始参数数量: {original_params}")
        print(f"分解后参数数量: {factorized_params}")
        print(f"参数减少: {(1 - factorized_params/original_params)*100:.2f}%")

        # 计算操作数
        original_ops, factorized_ops = compute_operations(fc1_weight.shape, rank)
        print(f"原始操作数: {original_ops}")
        print(f"分解后操作数: {factorized_ops}")
        print(f"操作减少: {(1 - factorized_ops/original_ops)*100:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 量化

    最后，我们将讨论另一种直观且有用的技术来减少模型的大小。

    ### 动机

    通常，神经网络的参数（层权重）使用32位浮点数表示。这很有用，因为它提供了高精度水平。

    此外，由于参数通常不受任何特定值范围的约束，所有深度学习框架默认为参数分配最大的数据类型。

    但使用最大的数据类型也意味着消耗更多内存。

    正如您可能猜到的，量化涉及使用较低位表示，如16位、8位、4位，甚至1位来表示参数。

    ![量化](https://www.dailydoseofds.com/content/images/2023/09/image-158.png)

    这导致存储模型参数所需的内存量显著减少。

    例如，考虑您的模型有超过一百万个参数，每个都用32位浮点数表示。

    如果可能，用8位数字表示它们可以导致内存使用的显著减少（约75%），同时仍然允许表示大范围的值。

    当然，量化在模型大小和精度之间引入了权衡。

    虽然减少参数的位宽使模型更小，但它也导致精度损失。

    这意味着模型的预测变得比原始的全精度模型更近似。

    因此，在考虑部署量化时，仔细评估模型大小/推理速度和准确性之间的权衡是重要的。

    尽管有这种权衡，量化在模型大小是关键约束的场景中特别有用，如边缘设备、移动应用程序或智能手机等专用硬件。

    量化实际使用的最常见方式之一称为**训练后量化**。

    ### 训练后量化

    顾名思义，这涉及使用较低精度的浮点数来表示训练模型的权重。

    所以首先，我们像通常那样训练模型。

    接下来，我们改变模型参数的数据类型。

    这是通过在代表性数据集上评估模型的权重和激活并确定表示它们的最合适精度来完成的。
    """
    )
    return


@app.cell
def _(StudentModel, device, student_model, torch):
    # 训练后量化实现
    def post_training_quantization(model):
        """
        对模型进行训练后量化

        Args:
            model: 要量化的模型

        Returns:
            quantized_model: 量化后的模型
        """
        # 设置量化配置
        model.eval()

        # 使用PyTorch的动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # 量化线性层
            dtype=torch.qint8   # 使用8位整数
        )

        return quantized_model

    def get_model_size(model):
        """
        计算模型大小（以MB为单位）

        Args:
            model: 模型

        Returns:
            size_mb: 模型大小（MB）
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    # 创建一个新的模型副本进行量化
    quantization_model = StudentModel()
    quantization_model.load_state_dict(student_model.state_dict())
    quantization_model.to(device)

    # 计算原始模型大小
    original_size = get_model_size(quantization_model)
    print(f"原始模型大小: {original_size:.2f} MB")

    # 执行量化
    quantized_model = post_training_quantization(quantization_model)

    # 计算量化后模型大小
    quantized_size = get_model_size(quantized_model)
    print(f"量化后模型大小: {quantized_size:.2f} MB")
    print(f"大小减少: {(1 - quantized_size/original_size)*100:.2f}%")
    return quantization_model, quantized_model


@app.cell
def _(evaluate_model, quantization_model, quantized_model, test_loader, torch):
    # 评估量化模型性能
    def evaluate_quantized_model(model, test_loader):
        """
        评估量化模型
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                # 量化模型在CPU上运行
                data, target = data.cpu(), target.cpu()
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    # 评估量化模型
    quantized_accuracy = evaluate_quantized_model(quantized_model, test_loader)
    print(f"量化模型准确率: {quantized_accuracy:.2f}%")

    # 与原始模型比较
    original_accuracy = evaluate_model(quantization_model, test_loader)
    print(f"原始模型准确率: {original_accuracy:.2f}%")
    print(f"准确率变化: {quantized_accuracy - original_accuracy:.2f}%")
    return original_accuracy, quantized_accuracy


@app.cell
def _(mo, original_accuracy, quantized_accuracy):
    mo.md(
        f"""
    ### 量化结果分析

    如上所示：

    - 两个模型都达到了相同的测试准确率（原始模型：{original_accuracy:.2f}%，量化模型：{quantized_accuracy:.2f}%）。
    - 然而，量化模型更小，因为它使用较小的数据类型来表示参数。

    💡 **重要提示**

    虽然在这种情况下，准确率是相同的。然而，这并不意味着量化模型总是会表现得像未量化（或正常）模型一样好。重要的是要在一系列数据类型上评估量化模型，并主观地决定最佳模型，就像我们之前在零剪枝、激活剪枝和低秩分解中所做的那样。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 结论

    至此，我们结束了这篇文章。

    在这篇文章中，我们讨论了几种模型压缩技术，这些技术对以下方面很有用：

    1. 加速模型推理阶段
    2. 减少模型的大小

    上述两个参数都是生产环境中部署的模型的关键要求。

    特别是，我们讨论了：

    - **知识蒸馏**：训练一个大模型（教师）并将其知识转移到一个较小的模型（学生）。
    - **剪枝**：从网络中移除不相关的边和节点。
        - 零剪枝
        - 激活剪枝
        - 冗余剪枝
    - **低秩分解**：将权重矩阵分解为较小的"低秩"矩阵。
    - **量化**：使用较低位表示来存储参数以减少模型的内存使用，而不是默认的大表示。

    我们还讨论了它们在Python中的实现。

    您可以在这里找到本文的代码：

    [模型压缩文章代码](https://www.dailydoseofds.com/content/files/2023/09/Model-Compression-Article-2.ipynb)

    了解这些技术对于在资源有限的设备上部署ML模型也非常有帮助——例如智能手机和Amazon Alexa。

    ### 关键要点

    1. **模型压缩是生产部署的必需品**：在现实世界应用中，模型的效率往往比纯粹的准确性更重要。

    2. **权衡是不可避免的**：所有压缩技术都涉及在模型大小/速度和准确性之间的权衡。关键是找到适合您特定用例的平衡点。

    3. **组合技术可能更有效**：您可以组合多种压缩技术（例如，知识蒸馏 + 剪枝 + 量化）以获得更好的结果。

    4. **评估是关键**：始终在您的特定数据集和用例上评估压缩模型，因为结果可能因应用而异。

    5. **硬件考虑**：不同的压缩技术在不同的硬件上可能有不同的效果。例如，量化在某些专用硬件上可能特别有效。

    通过掌握这些模型压缩技术，您将能够构建既高性能又高效的机器学习系统，适合在各种环境中部署。
    """
    )
    return


if __name__ == "__main__":
    app.run()
