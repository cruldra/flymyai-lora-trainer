import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 完整的MLOps蓝图：机器学习系统生命周期

    ## 回顾

    在深入MLOps和LLMOps速成课程的第2部分之前，让我们简要回顾一下在课程的前一部分中涵盖的内容。

    在第1部分中，我们探索了MLOps的背景和基础知识。我们首先了解了MLOps的确切含义。然后我们意识到模型开发实际上只是一个更大旅程中的很小一部分。

    ![MLOps概览](https://www.dailydoseofds.com/content/images/2025/07/image-111.png)

    接下来，我们讨论了为什么MLOps很重要，以及它如何帮助解决生产环境中ML的一些长期存在的问题。

    ![MLOps重要性](https://www.dailydoseofds.com/content/images/2025/07/image-112.png)

    我们还探索了ML系统/MLOps与传统软件系统之间的差异。

    ![ML vs 传统软件](https://www.dailydoseofds.com/content/images/2025/07/image-113.png)

    然后我们将注意力转向讨论生产ML中的一些术语和系统级关注点。在那里，我们检查了与延迟和吞吐量、数据和概念漂移、反馈循环等相关的一些常见关注点。

    ![系统级关注点](https://www.dailydoseofds.com/content/images/2025/07/image-114.png)

    最后，我们快速查看了机器学习系统生命周期及其各个阶段。

    ![ML系统生命周期](https://www.dailydoseofds.com/content/images/2025/07/image-115.png)

    在第1部分结束时，我们清楚地认识到ML不仅仅是以模型为中心的练习，而是一个系统工程学科，其中可重现性、自动化和监控是一等公民。

    如果你还没有学习第1部分，我们强烈建议先复习它，因为它建立了理解我们即将涵盖的材料所必需的概念基础。

    你可以在下面找到它：

    [The Full MLOps Blueprint: Background and Foundations for ML in Production](https://www.dailydoseofds.com/mlops-crash-course-part-1/)

    在本章中，我们将更深入地探索ML系统生命周期，重点关注每个单独阶段的关键细节。

    之后，我们将通过一个最小演示来演练，该演示提供了ML系统的快速模拟，帮助你实际连接到我们已经涵盖的一些理论概念的基础知识以及我们将在这一部分进一步探索的概念。

    一如既往，每个概念都将通过清晰的示例和演练来解释，以培养扎实的理解。

    让我们开始吧！

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 机器学习系统生命周期（续）

    在我们的上一部分中，我们概述了机器学习系统生命周期。我们看到了ML系统生命中的不同阶段：范围界定 → 数据 → 建模 → 部署和监控。

    ![ML系统生命周期概览](https://www.dailydoseofds.com/content/images/2025/07/image-116.png)

    让我们通过进一步分解其中一些组件来加深我们的理解，重点关注ML管道的核心技术元素：

    ### 数据管道

    在ML中，数据的质量和管理通常比特定的建模算法更重要。

    在生产ML系统中，你需要一个强大的数据管道来可靠地将数据输入模型训练，并最终输入模型推理（服务）。

    数据管道的关键方面包括：

    #### 数据摄取

    将来自各种来源的原始数据导入你的系统/开发环境。

    ![数据摄取](https://www.dailydoseofds.com/content/images/2025/07/image-129-1.png)

    摄取可以批量完成（定期导入转储或运行每日作业）或通过流式处理（实时处理传入事件）。

    #### 数据存储

    一旦数据被摄取，它需要被存储（通常以原始形式和处理形式）。

    ![数据存储](https://www.dailydoseofds.com/content/images/2025/07/image-130.png)

    常见的存储解决方案包括数据湖（例如，云对象存储如AWS S3、GCP Cloud Storage或本地HDFS）、关系数据库或专门的存储。

    👉

    本地HDFS意味着在你自己的基础设施上运行Hadoop分布式文件系统（HDFS）。

    如果你想深入了解HDFS，我们在PySpark深度探讨中涵盖了它：

    [Don't Stop at Pandas and Sklearn! Get Started with Spark DataFrames and Big Data ML using PySpark](https://www.dailydoseofds.com/dont-stop-at-pandas-and-sklearn-get-started-with-spark-dataframes-and-big-data-ml-using-pyspark/)

    许多生产ML团队维护一个特征存储，它充当预计算特征的集中数据库，用于模型训练，也可用于在线推理。

    特征存储确保离线训练数据和在线服务数据之间的一致性。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 数据处理（ETL）

    原始数据通常需要大量处理才能对建模有用。这包括连接多个数据源、清理、标准化特征、编码分类变量、创建新特征等。

    这通常在ETL（提取-转换-加载）管道中完成。可以使用Apache Spark等工具，甚至是带有Pandas的普通Python脚本，具体取决于规模。

    ![数据处理ETL](https://www.dailydoseofds.com/content/images/2025/07/image-131.png)

    这一步的输出通常是一个准备好进行模型训练的策划数据集。

    #### 数据标注和注释

    如果ML问题是监督学习并需要标签（真实情况），你需要一个过程来为你的数据获取标签。在某些情况下，标签是自然收集的，在其他情况下，你可能需要人工注释。

    ![数据标注](https://www.dailydoseofds.com/content/images/2025/08/image-3.png)

    生产系统可能包括一个标注管道，使用内部团队或众包来持续标注新数据。

    #### 数据版本控制和元数据

    跟踪哪些数据用于训练哪个模型是至关重要的。

    数据可能随时间变化（追加新记录、应用更正等），所以简单地说"在数据集X上训练"可能不够；我们需要知道数据集X的哪个版本用于训练模型，以便进行可重现性、审计和模型比较等目的。

    许多团队记录其管道的元数据：数据提取的时间戳、文件的校验和、记录数等。

    ![数据版本控制](https://www.dailydoseofds.com/content/images/2025/07/image-132.png)

    工具可以帮助明确管理数据集版本，例如，DVC允许你对数据进行版本控制。我们稍后会更多地讨论数据版本控制。

    在我们转到ML系统生命周期的下一阶段之前，需要做一个重要区分：离线数据管道（用于训练）和在线数据管道（用于实时提供特征）之间的区别。

    👉

    在线提供特征（或在线服务数据）意味着收集/计算机器学习模型为特定用户或请求实时进行预测所需的输入数据（特征）。

    离线管道可以更重，因为它们不面向用户；我们可能花费数小时处理大批量数据来创建训练集。在线管道需要轻量级和低延迟，例如，为单个用户请求即时计算特征。

    确保离线和在线管道一致（不产生矛盾结果）是一个已知的挑战。如果它们分歧，模型在生产中的行为可能与训练中不同，导致训练/服务偏差。

    使用共享特征存储或通过模拟在线计算来派生训练数据是保持它们同步的常见解决方案。

    总的来说，数据管道是生产中ML的基础。如果输入的数据不可靠，输出的模型预测也不会可靠。

    正如Chip Huyen强调的，由于ML模型从数据中学习，"开发ML模型始于工程数据"。实际上，数据质量问题是生产ML中比算法问题更频繁的失败原因。这就是为什么MLOps如此关注数据监控、验证和版本控制。

    接下来，让我们看看训练和实验过程。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 模型训练和实验

    一旦你的数据准备好了，下一个阶段就是模型训练。在生产环境中，这个阶段通常仍然离线进行（在研究或批处理环境中），而不是直接在实时生产系统中。

    然而，在生产场景中管理训练的方式比临时实验更严格。这个阶段的关键考虑因素包括：

    #### 实验跟踪

    在开发过程中，数据科学家会尝试多种方法，如不同的模型架构、特征、超参数设置等。跟踪这些实验至关重要，以便结果可重现和可比较。

    ![实验跟踪](https://www.dailydoseofds.com/content/images/2025/07/image-133.png)

    这涉及记录以下内容：

    - 哪个代码版本产生了模型
    - 使用了哪个数据子集
    - 设置了什么超参数
    - 评估指标（准确性、损失等）

    👉

    好处是你可以轻松回答诸如"哪次训练运行给了我们那个0.85 AUC模型，它的参数是什么？"或"添加特征X实际上提高了准确性还是没有？"这样的问题。

    #### 选择获胜者和模型验证

    通过实验，你可能最终得到一个在验证指标上表现最好的候选模型。

    ![模型选择](https://www.dailydoseofds.com/content/images/2025/08/image-4-1.png)

    在将此模型推送到生产之前，它通常会经过更严格的评估。

    这可能意味着：

    - 在实验期间未见过的新保留测试集上进行测试，以获得性能的无偏估计。
    - 它还可能涉及特定领域的评估：例如，让临床医生审查医疗AI的预测样本，或在边缘案例上运行模型。
    - 在某些组织中，这一步包括对伦理和偏见问题的审查。

    👉

    在项目开始时建立基线指标和接受标准是一个好做法（例如，"我们需要比当前系统至少提高5%"）。这样，你就知道实验模型何时"足够好"可以发布。

    #### 训练管道自动化

    在生产MLOps中，你经常将训练过程自动化为管道，使其可重复并最终可调度。

    这意味着脚本化整个序列：获取最新数据 → 预处理 → 训练模型 → 评估指标 → （可选）如果好的话将模型推送到注册表。特定工具或带有CI系统的自定义脚本可以编排这个过程。

    💡

    模型注册表是存储和管理机器学习模型不同版本的中央存储库。它允许团队跟踪模型的整个生命周期，从开发和测试到部署。当模型被"推送到注册表"时，它通常会记录其版本、性能指标和训练数据集的元数据。

    即使初始部署是手动的，准备好管道意味着你可以轻松重新触发训练（例如，"每周用最新数据训练新模型"）。如果你计划进行持续重新训练以保持模型最新，自动化训练变得特别重要。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 资源管理

    训练现代ML模型（特别是深度学习）可能是计算密集型的。在生产设置中，你可能利用带有GPU/TPU的云虚拟机或分布式计算集群来更快地训练。

    MLOps涉及使这种资源使用高效（使用正确的硬件，如果可能的话并行化作业，并且可能对非常大的数据集/模型使用分布式训练）。

    虽然训练基础设施的详细研究目前超出了我们的范围，但要意识到"生产中的ML"的一部分是确保你可以在合理的时间内训练所需复杂性的模型。

    例如，使用云服务在需要时启动GPU实例，或使用竞价实例来节省成本等。许多团队使用Docker镜像或类似工具来封装训练环境，使其在开发机器和训练服务器之间可移植和一致。

    #### 超参数调优

    训练的一个子集是调优超参数（如学习率、正则化强度和网络架构选择）。在生产管道或甚至正常项目设置中，这可以通过执行网格搜索、随机搜索或[贝叶斯优化](https://www.dailydoseofds.com/bayesian-optimization-for-hyperparameter-tuning/)的作业自动化。

    这些调优作业的结果被反馈到实验跟踪系统中。重要的是不要对验证集过度调优（以避免以微妙的方式过拟合），所以有时团队会在调优后对真正的盲测试进行最终评估。

    #### 训练中的协作和可重现性

    通常，多个人可能会在模型代码上工作。使用代码版本控制（Git）是必不可少的，这样实验就与特定的代码提交相关联。

    ![协作和版本控制](https://www.dailydoseofds.com/content/images/2025/07/image-134-1.png)

    一些团队甚至为不同的模型想法使用特性分支。环境（库版本等）应该被管理（使用`requirements.txt`/`pip`、Conda或Docker作为更重型的解决方案）。这确保如果其他人试图重新运行你的训练代码，他们可以得到相同的结果。

    总之，生产环境中的训练和实验阶段是关于严格性和跟踪的。

    它仍然是创造性和探索驱动的，但你正在奠定基础（通过跟踪和自动化），以确保获胜的模型可以可靠地移动到下一阶段。

    在这个阶段结束时，你理想情况下有：一个训练好的模型工件（例如，`.pkl`、`SavedModel`、`ONNX`或其他支持的文件），伴随的元数据（训练代码版本、数据版本、指标），以及这个模型准备提供价值的信心，但需要在实时系统中进行进一步测试。

    接下来，让我们看看部署阶段。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 模型部署和推理

    部署是橡胶遇到路面的阶段。模型从训练环境中取出并集成到生产系统中，以便它可以开始为最终用户或其他系统提供预测。

    ![模型部署](https://www.dailydoseofds.com/content/images/2025/08/image.png)

    模型部署有多种模式，包括在线实时服务、离线批处理、边缘部署等，但我们将涵盖常见方面和考虑因素：

    #### 打包模型

    首先，训练好的模型工件需要以可以在生产中加载和执行的形式打包。

    如果你的训练环境和生产环境相似（例如，都是Python），你可能直接序列化模型（使用Python的pickle、`joblib`或框架的保存格式，如Keras的`.h5`、PyTorch的`.pt`等）。

    但如果环境不同（比如，在Python中训练但在C++中推理以提高效率），你可能将模型导出为标准化格式，如`ONNX`，它可以在其他语言中加载。

    ![模型打包](https://www.dailydoseofds.com/content/images/2025/07/image-135-1.png)

    无论如何，部署的一部分是产生一个将被部署的版本化模型工件。将其存储在模型注册表或工件存储中是一个好做法，这是我们稍后将讨论的主题。

    👉

    模型注册表基本上是一个感兴趣的模型数据库，包含版本和元数据。

    请注意，只有少数模型你会想要在集中存储库中注册以供以后使用。我们不注册每个记录的模型。

    ![模型注册表](https://www.dailydoseofds.com/content/images/2025/07/image-136.png)

    #### 作为服务部署（在线推理）

    实时应用程序最常见的方法是将模型部署为API后面的微服务。例如，Python中的Flask或FastAPI应用程序，在启动时加载模型并公开`/predict`端点。

    此服务接收特征输入（可能作为JSON）并实时返回模型预测（如分类标签或分数）。你通常会使用Docker容器化此服务以保持一致性，然后在服务器或Kubernetes集群上运行它。

    ![在线推理服务](https://www.dailydoseofds.com/content/images/2025/08/image.png)

    许多公司使用RESTful API或有时使用gRPC。服务应该是可扩展的（你可能运行多个副本来处理高负载）并在负载均衡器后面。在MLOps环境中，部署模型可以像通过DevOps管道部署任何其他微服务一样简单，除了确保包含和加载模型文件的额外步骤。

    现代云ML平台（如Amazon SageMaker和Azure ML）可以自动化其中的大部分：你给它们一个模型工件，它们为你启动一个可扩展的端点。

    无论如何，延迟和吞吐量在这里是关键：如果你的模型为面向用户的功能提供动力，你需要确保推理得到优化（如果需要，使用适当的硬件或优化，如模型量化）以满足延迟预算。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 批量推理

    不是每个ML部署都是实时API。许多用例对批量预测是可以接受的。

    例如，假设你有一个为营销活动细分客户的模型。你可能每周对所有客户记录运行一次此模型，并将结果存储在数据库中。这就是批量推理。

    ![批量推理](https://www.dailydoseofds.com/content/images/2025/08/Screen-Recording-2025-07-26-at-4.11.14---PM-2-2.gif)

    它可以实现为加载模型、处理大型数据集并写入输出的计划作业。批量作业可以根据规模在大数据框架上运行。它们通常没有严格的延迟要求，但你必须担心吞吐量（如何高效处理数百万条记录）。

    批量的一个优势是你可以使用分布式计算的全部功能，而且你不必保持服务24/7运行。你只在需要时运行它。批量推理的代码可能与训练代码位于同一存储库中（例如，一个接受输入数据集并输出结果的脚本`predict.py`）。

    #### 边缘和移动部署

    在某些情况下，模型不是部署到服务器，而是部署到最终用户设备（手机、物联网设备等）。

    ![边缘部署](https://www.dailydoseofds.com/content/images/2025/07/image-138.png)

    这引入了限制，如有限的计算、功耗，以及除非用户更新应用程序否则无法直接更新。

    ![移动部署限制](https://www.dailydoseofds.com/content/images/2025/07/image-139.png)

    压缩模型（剪枝、量化）和使用专门运行时等技术变得相关。

    ![模型压缩技术](https://www.dailydoseofds.com/content/images/2025/07/image-140.png)

    例如，语音助手模型可能在智能音箱上运行。因此，它必须小而高效。虽然边缘部署本身是一个大话题，但值得提及作为部署风格之一。

    #### 与更大系统的集成

    通常，部署模型不仅仅是关于模型代码本身，还关于与现有系统的集成。例如，如果你将欺诈检测模型部署为服务，交易处理系统必须调用此服务，并决定如果模型标记某些内容该怎么办。

    这可能涉及模型之外的一些应用程序逻辑。可能还需要后备或手动覆盖，例如，如果模型服务宕机，你的系统可能默认为安全行为，或者如果模型不是很有信心，可能升级到人工审查。

    这些集成考虑对生产就绪性很重要，因为它们有助于可靠性。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 金丝雀发布和A/B测试

    在许多情况下，你不会部署新模型并立即将100%的流量发送给它。特别是如果模型变化很大，团队使用金丝雀部署等策略。

    在金丝雀部署中，我们将一小部分流量发送到新模型，将新行为和输出/性能与旧模型进行比较。如果看起来不错（没有错误，更好的指标），逐渐增加流量。

    ![金丝雀部署](https://substackcdn.com/image/fetch/$s_!QNGP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff0ad2e03-ad3d-4a7c-9d0f-1cfc31fb1224_3392x904.png)

    另一种方法是A/B测试，其中一部分用户从新模型获得预测，其他用户从控制组（旧模型）获得预测，你比较业务指标。

    ![A/B测试](https://substackcdn.com/image/fetch/$s_!yHzQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbc361e6f-bc8b-4673-b3b8-ada70f1a27b7_3368x704.png)

    例如，如果是推荐系统，新模型是否真的增加了点击或转换？如果你有基础设施，这可以在线实时完成，或者如果你以受控方式部署，可以通过分析日志离线完成。

    这些技术确保新模型实际上是改进，并且在完全推出之前不会造成负面副作用。

    从MLOps角度来看，支持A/B测试意味着你的系统可能在生产中同时运行两个版本的模型一段时间。这需要跟踪版本和路由逻辑，这再次突出了版本控制和可观察性的需要。

    金丝雀部署和A/B测试之间的关键差异：

    - 金丝雀部署专注于发布期间的风险降低和稳定性，而A/B测试专注于优化和功能有效性。
    - 金丝雀部署监控运营指标（错误、性能），而A/B测试专注于用户行为指标（转换、参与度）。
    - 金丝雀部署使用小的代表性组来识别问题，而A/B测试需要统计上显著和平衡的组来确保有效比较。

    虽然不同，这些策略可以结合使用。金丝雀部署可能先于A/B测试，确保新版本在实验设置中使用之前是稳定的。

    #### 扩展和可靠性

    一旦部署，你的模型服务必须扩展以处理负载。这可能意味着水平扩展（运行N个副本）。

    使用像Kubernetes这样的容器编排是常见的：你可以定义一个维持所需吞吐量的自动扩展部署。微服务的可靠性实践在这里适用：

    - 健康检查：服务应该报告模型是否已加载并正常运行
    - 日志记录：请求和响应，可能采样一些用于分析
    - 警报：如果错误率或延迟激增，应通知值班工程师

    有时模型有特定的故障模式，例如，如果模型加载失败或由于上游数据问题开始返回奇怪的输出。所以监控需要在系统级别和应用程序级别（下一节将详细介绍监控）。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 模型注册表和CI/CD集成

    在结构良好的MLOps设置中，当模型被训练和批准（来自前一阶段）时，它会在模型注册表中注册一个版本号（例如，"FraudModel v1.3"）。

    ![模型注册表集成](https://www.dailydoseofds.com/content/images/2025/07/image-141.png)

    然后部署可以从此注册表中拉取模型。ML部署过程可以与CI/CD管道集成：例如，一旦测试通过并且模型被批准，CI管道可以自动构建包含模型的新Docker镜像并将其部署到暂存环境。

    几个可用的工具可以用来获取特定的模型版本，甚至跟踪哪个版本在暂存与生产中。这有助于治理，你总是可以知道哪个确切的模型二进制文件在生产中运行以及什么数据/代码产生了它。

    我们将在下一节中更多地讨论模型注册表，但这里相关的是部署模型理想情况下应该像部署代码一样可追溯和可重复。

    总之，部署将静态模型工件转变为与世界交互的实时服务（或作业）。它需要扎实的工程：

    - 容器化
    - API设计
    - 可扩展性
    - 云基础设施（通常）

    部署的成功标准通常是延迟、吞吐量和可靠性（正常运行时间、错误处理）。仅仅模型在实验室中99%准确是不够的；如果它在生产中需要10秒响应或经常崩溃，用户体验将受到影响。

    因此，MLOps的很大一部分是在模型质量与工程约束之间取得平衡（有时甚至为了速度或稳定性而简化模型）。

    ### 监控和可观察性

    部署后，ML模型不会被单独留下；它需要持续监控。

    ![监控和可观察性](https://www.dailydoseofds.com/content/images/2025/08/image-7.png)

    监控ML系统涉及你在正常软件服务中监控的所有内容，加上一些ML特定的角度。

    让我们深入了解：

    - 运营监控
    - 漂移监控
    - 模型性能监控

    #### 运营监控

    这是传统方面。确保服务正常运行并响应。关键指标包括：

    - 延迟（响应时间）
    - 吞吐量（每秒请求数）

    ![延迟和吞吐量监控](https://www.dailydoseofds.com/content/images/2025/08/image-6.png)

    - 错误率（HTTP 5xx错误等）
    - 资源使用（CPU、内存、GPU利用率如果适用）

    ![资源监控](https://www.dailydoseofds.com/content/images/2025/08/image-5.png)

    如果使用云自动扩展，你可能监控负载并相应扩展。

    ML API的运营监控类似于任何Web服务。可以使用Prometheus/Grafana、CloudWatch等工具来收集这些指标。

    应该为异常设置警报（例如，延迟激增或服务宕机）。此外，由于ML模型可能在专门硬件（GPU）上运行，跟踪这些资源以避免OOM崩溃很重要。

    从MLOps角度来看，你还想记录对模型的调用，不仅用于调试，还用于分析。

    通常，每个请求的模型输入特征和输出都会被记录（适当采样和隐私考虑），以便你稍后可以分析它们。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### 漂移监控

    ML生产中的独特挑战之一是确保输入数据保持在模型知道或可以令人满意地泛化的范围内。数据漂移意味着输入数据的分布随时间变化。

    ![数据漂移](https://www.dailydoseofds.com/content/images/2025/08/image-8.png)

    例如，图像分类模型在白天图像上训练，但现在它获得更多夜间图像；因此，像素分布发生变化。

    概念漂移意味着输入和输出之间的关系发生变化。例如，垃圾邮件过滤器模型的垃圾邮件定义可能会随着垃圾邮件发送者采用新技术而改变，因此过去用于指示垃圾邮件的特征可能不再适用。

    ![概念漂移](https://www.dailydoseofds.com/content/images/2025/08/image-9.png)

    监控漂移涉及对传入数据进行统计检查。你可能跟踪数值特征的均值或标准差，或类别的频率，并与训练中看到的进行比较。

    显著偏差可能表明漂移。有高级指标（人口稳定性指数、KL散度等）可以量化漂移。

    #### 模型性能监控

    最终，我们关心模型是否仍然很好地执行其任务。挑战在于我们通常不会立即知道生产中每个预测的"正确答案"。例如，如果模型预测交易是"欺诈"或"非欺诈"，我们可能直到稍后（如果有的话）才知道它是否正确。

    然而，我们可以做一些代理指标和定期检查：

    - 代理指标：
        - 对于分类，你可以监控模型的置信度分数分布。如果模型突然非常不确定（所有预测的置信度较低）或过于确定（总是给出极端概率），可能有问题。

    ![置信度分布监控](https://www.dailydoseofds.com/content/images/2025/08/image-10-1.png)

    - 你还可以监控正预测的比率。如果历史上你的欺诈模型预测15%的交易为欺诈，现在它标记55%，这是一个红旗；要么世界发生了变化，要么模型有问题。

    ![预测比率监控](https://www.dailydoseofds.com/content/images/2025/08/image-11-1.png)

    - 对于回归，监控输出范围。比如如果你预测价格，突然你有负价格或与以前不同的极大值，这是一个问题。

    ![回归输出监控](https://www.dailydoseofds.com/content/images/2025/08/image-12-1.png)

    - 真实情况反馈：
        - 在某些系统中，你确实会得到最终的真实情况。
        - 例如，在推荐系统中，用户点击可以作为推荐是否相关的真实情况。在信用评分模型中，你最终会看到用户是否违约（尽管有很长的延迟）。
        - 每当稍后有真实结果可用时，你应该闭合循环并测量模型在最近数据上的实际准确性。
        - 这可以通过定期回测来完成。例如，取过去预测及其结果的样本，计算准确性、精确度/召回率等指标，并随时间跟踪这些指标。这些指标的下降表明性能退化，可能由于漂移或模型陈旧。

    ![回测分析](https://www.dailydoseofds.com/content/images/2025/08/image-14-1.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - 监控中的影子模型：
        - 一些高级设置部署影子模型，即与旧模型并行运行的新版本，不影响决策，只是为了比较输出。这可以用来监控如果你更新模型会发生什么。

    ![影子模型概念](https://www.dailydoseofds.com/content/images/2025/08/image-21.png)

    - 例如，你可能有一个影子中的新模型版本，它也获得实时流量输入，你将其预测与当前模型进行比较。

    ![影子模型比较](https://www.dailydoseofds.com/content/images/2025/08/image-15-1-1.png)

    - 这可以突出显示它们不同意的情况，并允许分析新模型是否会做得更好或更差。

    - 业务指标：
        - 最终，如果ML模型是产品的核心，其性能将显示在业务KPI中。
        - 例如，如果推荐模型变差，你可能会看到参与度下降（点击率或观看时间）。这些指标通常是嘈杂的，受许多因素影响，但跟踪模型应该驱动的指标很重要。

    ![业务指标监控](https://www.dailydoseofds.com/content/images/2025/08/image-16-1.png)

    - MLOps涉及与产品所有者合作定义生产中成功的样子（例如，"我们期望这个模型将转换提高X%"），然后看看这是否成立。

    👉

    Booking.com与数百个模型的经验教导说，模型性能与业务性能不同。你必须测量两者。准确性稍低的模型如果更符合真实目标或用户体验，实际上可能在业务方面表现更好。

    #### 警报和响应

    有了监控，你需要当出现问题时该怎么办的策略。例如，如果数据漂移超过阈值，你是否自动重新训练模型（如果可能）？或者你是否警告工程师进行调查？

    如果模型的准确性（通过延迟的真实情况测量）低于阈值，你是否回滚到以前的模型版本？

    ![警报和响应策略](https://www.dailydoseofds.com/content/images/2025/08/image-17-1.png)

    一些系统实施自动回滚。例如，如果新模型的指标比旧模型差5%，将流量恢复到旧模型。

    然而，许多组织仍然依赖手动干预，因为如果监控指标嘈杂，自动决策可能很棘手。无论如何，运营团队（或MLOps团队）应该有问题的运行手册。

    例如，如果模型开始发送太多警报，考虑用新数据重新训练或检查数据管道问题；如果服务延迟退化，也许模型太慢，因此可能扩展或优化代码等。

    ![运行手册示例](https://www.dailydoseofds.com/content/images/2025/08/image-18-1.png)

    最后，为了说明监控为什么如此关键，想象一个高风险领域如医疗诊断中的ML模型。

    如果来自诊所的数据发生变化（也许新的传感器设备引入了稍微不同的信号特征），模型的预测悄悄变得不那么准确，如果不被检测到，你可能会有严重后果。

    监控理想情况下会捕捉到新数据分布不同，最近的预测结果与医生的评估不匹配，促使模型更新或调查。

    ![医疗AI监控](https://www.dailydoseofds.com/content/images/2025/08/image-20.png)

    即使在不太关键的领域如广告点击预测中，未能监控意味着你可能几个月都在提供次优模型，损失大量收入，因为没有人注意到准确性随时间的缓慢下降。

    简而言之，监控"闭合循环"在ML生命周期中。它将来自实时系统的信息反馈到开发过程。它确保你的模型继续做你期望的事情，并提供何时应该刷新或改进它的信号。

    MLOps超越部署建立长期监督。

    正如一个指南所说，一旦你部署了模型，"旅程才刚刚开始"。期望在这里花费大量精力，因为监控和维护模型是生产ML中大部分持续工作所在。

    随着ML系统生命周期的详细覆盖，让我们通过一个简单而具体的模拟来演练，将玩具模型从Jupyter笔记本带到API服务。

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 从训练到API的实践项目

    为了具体化这些想法，让我们通过一个简化的部署ML模型的实践示例来演练。

    想象我们有一个Jupyter笔记本/Python脚本，在经典的鸢尾花数据集上训练了一个基本模型，一个scikit-learn分类器（从花瓣/萼片测量预测花种）。

    离线时，我们获得了良好的准确性。现在我们想让这个模型作为Web服务可用，以便应用程序可以用新的花朵测量调用它并获得预测的种类。

    我们将使用FastAPI创建一个简单的模型推理API。这个练习旨在模拟一个最小的ML系统。从数据科学原型到微服务的场景。

    ![项目概览](https://www.dailydoseofds.com/content/images/2025/08/image-1.png)

    我们将遵循的步骤：

    - 训练并序列化模型（这将在笔记本或脚本中完成）。
    - 编写一个FastAPI应用程序，加载模型并定义预测端点。
    - 运行API服务器并用样本输入测试它。
    - 用Docker容器化应用程序（用于可重现性考虑，也因为在真实场景中，需要时可以部署到某个平台）。

    ### 项目设置

    我们使用的代码和项目设置作为zip文件附在下面。你可以简单地提取它并运行`uv sync`命令开始。建议按照README文件中的说明开始。

    ![项目设置](https://www.dailydoseofds.com/content/images/2025/07/image-118.png)

    下载下面的zip文件：

    [MLOps-FastAPI-I.zip](https://www.dailydoseofds.com/content/files/2025/07/MLOps-FastAPI-I-1.zip)

    有关依赖项的详细信息，你可以检查`pyproject.toml`或`requirements.txt`文件。

    ### 训练并保存模型

    现在，让我们定义一个非常基本的训练脚本，训练模型并保存它：

    ```python
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    # 加载鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    # 保存模型
    joblib.dump(model, 'iris_model.pkl')
    print("模型已保存为 iris_model.pkl")
    ```

    我们使用了一个有50棵树的随机森林。然后我们序列化训练好的模型。

    这段代码作为单独的训练管道运行，不在部署的应用程序中。

    在实践中，你还会对这个模型进行版本控制（例如，"v1"），也许将指标记录到实验跟踪器。但现在，我们有一个准备部署的模型文件。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 创建用于推理的FastAPI应用程序

    我们将编写一个应用程序，在启动时加载`iris_model.pkl`并公开一个`/predict`端点。

    端点将接受特征输入（萼片和花瓣测量）并返回预测的种类。

    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    import joblib
    import numpy as np
    from sklearn.datasets import load_iris

    # 加载训练好的模型
    model = joblib.load('iris_model.pkl')

    # 加载鸢尾花数据集以获取类别名称
    iris = load_iris()

    # 创建FastAPI应用程序
    app = FastAPI(title="鸢尾花分类API", version="1.0.0")

    # 定义输入数据模型
    class IrisFeatures(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float

    # 定义输出数据模型
    class PredictionResponse(BaseModel):
        species: str
        confidence: float

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "模型服务正在运行"}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_species(features: IrisFeatures):
        # 将输入转换为numpy数组
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])

        # 进行预测
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # 获取种类名称和置信度
        species_name = iris.target_names[prediction]
        confidence = round(float(probabilities[prediction]), 3)

        return PredictionResponse(
            species=species_name,
            confidence=confidence
        )
    ```

    关于这段代码的几个注意事项：

    - 我们定义了一个`IrisFeatures` Pydantic模型来验证传入的JSON。这意味着请求应该发送一个带有指定键（`sepal_length`、`sepal_width`等）的JSON，它将自动解析为该模型。
    - 我们在端点函数外部加载模型，所以它在应用程序启动时加载一次（这避免了每次请求重新加载，这会很慢）。
    - 我们提供了一个简单的`/health` GET端点。在生产中，负载均衡器或编排器可能调用这个来检查我们的服务是否活着。
    - `/predict`端点接受特征，运行`model.predict`和`model.predict_proba`，然后返回种类名称和置信度分数。

    ![API架构](https://www.dailydoseofds.com/content/images/2025/08/image-2.png)

    - 我们使用`iris.target_names`，这是我们从数据集获得的。

    👉

    在真实代码中，我们需要保存这些映射或简单地输出数字类别并让客户端解释它，但为了演示，我们假设我们在这个范围内有映射。

    - 我们为了整洁而四舍五入置信度。响应中的所有数据都是JSON可序列化的（字符串和浮点数）。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 运行和测试API

    通常，你会使用服务器运行这个，使用Uvicorn（FastAPI的ASGI服务器）：

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 5000 --reload
    ```

    这将启动服务器（`--reload`在开发中很有用；它在代码更改时自动重启）。

    一旦你运行这个，你会得到类似下面显示的视图：

    ![服务器启动](https://www.dailydoseofds.com/content/images/2025/07/image-119.png)

    要测试API，我们可以定义一个简单的脚本：

    ```python
    import requests
    import time

    # API端点
    url = "http://localhost:5000/predict"

    # 测试数据
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # 记录开始时间
    start_time = time.time()

    # 发送POST请求
    response = requests.post(url, json=test_data)

    # 记录结束时间
    end_time = time.time()

    # 打印结果
    if response.status_code == 200:
        result = response.json()
        print(f"预测种类: {result['species']}")
        print(f"置信度: {result['confidence']}")
        print(f"响应时间: {(end_time - start_time) * 1000:.2f} ms")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
    ```

    运行这个脚本，我们得到类似以下的结果：

    ![测试结果](https://www.dailydoseofds.com/content/images/2025/07/image-120.png)

    这表明我们的模型非常确定这朵花是setosa种类，这是有道理的，考虑到输入。我们还得到了获得响应消耗的时间。

    FastAPI还在`http://localhost:5000/docs`提供可浏览的文档UI，你可以在那里与API交互（感谢从Pydantic模式自动生成）。

    ![Swagger UI](https://www.dailydoseofds.com/content/images/2025/07/image-121-1.png)

    这里我们有端点和我们服务的模式详细信息。在模式部分，我们可以检查在`IrisFeatures`下提供的输入详细信息：

    ![模式详细信息](https://www.dailydoseofds.com/content/images/2025/07/image-122.png)

    现在让我们看看如何通过这个界面与我们的API交互。选择`/health`端点并点击"Try it out"：

    ![健康检查测试](https://www.dailydoseofds.com/content/images/2025/07/image-123-1.png)

    之后点击"Execute"按钮：

    ![执行健康检查](https://www.dailydoseofds.com/content/images/2025/07/image-124-1.png)

    一旦你按照上述顺序操作，你会观察到类似下面显示的视图：

    ![健康检查结果](https://www.dailydoseofds.com/content/images/2025/07/image-125-1.png)

    同样我们可以测试我们的`/predict`端点。

    ![预测端点测试](https://www.dailydoseofds.com/content/images/2025/07/image-126.png)

    输入值：`{"sepal_length": 1.8, "sepal_width": 8.9, "petal_length": 6.9, "petal_width": 7}`

    如果我们使用上述值集，这是我们得到的：

    ![预测结果](https://www.dailydoseofds.com/content/images/2025/07/image-127-1.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    为了进一步发展，我们可以用Dockerfile容器化我们的设置。例如：

    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY app.py iris_model.pkl requirements.txt ./
    RUN pip install -r requirements.txt
    EXPOSE 80
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
    ```

    在Dockerfile中：

    - 第1行：使用最小的Python 3.11基础镜像。
    - 第2行：将容器内的工作目录设置为`/app`。
    - 第3行：将应用程序代码（`app.py`）、模型文件（`.pkl`）和依赖项列表（`requirements.txt`）复制到容器中。
    - 第4行：安装`requirements.txt`中列出的Python包。
    - 第5行：指定应用程序将在容器内的端口80上监听。
    - 第6行：使用Uvicorn在所有接口的端口80上启动FastAPI应用程序。

    接下来，我们可以用以下命令构建镜像：

    ```bash
    docker build -t iris-api .
    ```

    并用以下命令运行它：

    ```bash
    docker run -p 5000:80 iris-api
    ```

    这里用`-p 5000:80`，我们将容器的端口80映射到localhost的端口5000。

    使用服务的过程与之前相同。你可以使用`test.py`文件、`curl`命令或`http://localhost:5000/docs`的Swagger UI。

    总的来说，我们的简单API现在有效地是一个用于ML推理的微服务。当然，在真实场景中，我们会将其部署到某个地方进行访问：

    - 然后也许在云服务或Kubernetes集群上运行该容器。我们还会为模型路径等内容合并环境变量或配置。
    - 我们还会在predict函数内包含日志记录（例如，记录输入和输出或至少记录请求发生）用于监控。也许实现一些更多的输入验证（例如，没有负长度）。

    尽管如此，这个例子涉及几个生产考虑：

    - 我们序列化了模型（反映可重现性）
    - 我们分离了训练和服务
    - 我们构建了一个具有明确定义合同的API（输入模式）
    - 我们可以通过运行它的多个容器轻松扩展这个

    如果流量增加，我们将其放在负载均衡器后面并扩展。如果我们重新训练一个更好的模型，我们更新`iris_model.pkl`（确保在需要时也更新代码）并重新部署。

    在完整的MLOps设置中，我们会有一个CI/CD管道，这样当新模型被注册时（比如鸢尾花模型v2），部署管道会拾取它并部署这个服务的新版本，也许首先到暂存进行测试，然后到生产。

    通过这个实践之旅，我们模拟了将训练好的模型转变为API端点作为微服务。

    这是行业中发生的简化版本：数据科学家通常从笔记本/训练脚本开始，但工程师或ML工程师会拿训练好的模型并将其包装在应用程序中（或使用自动化平台来做）。

    通过以简单的方式做这件事，你可以欣赏所有涉及的部分：

    - 你需要模型工件
    - 加载它的方法
    - API框架
    - 输入/输出的数据模型
    - 依赖项的环境设置等

    每一个都对应我们之前讨论的一些关键概念（服务框架、数据验证、通过Docker的环境可重现性等）。

    在未来的部分中，当我们通过关键理论要点和方法进展时，我们将看看更复杂系统的实践模拟。现在，上述示例提供了足够的实践基础。

    ---
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
                                    ## 结论

    通过这次详细探索，我们已经超越了MLOps的什么和为什么，进入了MLOps系统的如何。准确地说，机器学习系统实际上是如何构建、训练、部署和监控的。

    我们首先深入挖掘ML系统生命周期，不仅仅作为一系列学术阶段，而是作为需要持续关注、自动化和验证的现实世界工程工作流程。

    我们探索了每个阶段应有的理论严谨性，从构建强大的数据管道到管理训练实验，从将模型部署为可扩展的API到通过系统监控维护性能。

    这些阶段中的每一个都代表了从临时实验到结构化、生产就绪思维的根本转变。

    这一部分的最大收获之一是数据的重要性。一次又一次，行业经验表明，模型性能只有从中学习的数据那么好。

    我们讨论了生产级ML系统不仅需要数据集，还需要精心设计的数据管道，具有可靠的摄取、存储、处理（ETL）、标注和版本控制机制。数据一致性中的小缺陷可能在部署后级联成预测中的大规模退化。

    因此，投资数据基础设施不仅仅是后端关注；它是任何ML系统的支柱。

    在训练和实验阶段，我们强调了从随意的笔记本修补到有组织的实验的转变，具有强大的可重现性保证。

    我们还强调了金丝雀发布和A/B测试等部署策略。这些机制允许团队安全地验证改进，及早检测回归，并对模型推出做出数据驱动的决策。

    最后，我们以一个经常被低估的关键阶段结束生命周期：监控和可观察性。在这里，DevOps的传统指标遇到了新的ML特定挑战，如数据漂移、概念漂移和模型性能随时间退化。

    我们介绍了处理这些的方法，从日志记录和警报，到跟踪代理指标，通过真实情况闭合反馈循环，甚至部署影子模型与生产模型进行比较。

    因为在MLOps中，部署不是旅程的结束，而是一套全新且经常不确定的挑战的开始。

    为了将这些概念锚定在现实中，我们以一个实际模拟结束：在鸢尾花数据集上将scikit-learn模型部署为FastAPI服务。这个简单但说明性的示例带我们经历了从模型训练到序列化，从编写推理API到用Docker容器化整个系统的旅程。

    它反映了现实中发生的事情，将在笔记本中构建的模型转换为活的、可扩展的微服务。

    虽然这个练习可能是轻量级的，但它涉及的核心生产原则——可重现性、输入验证、API标准化、环境一致性——是普遍适用的。

    正是这种简单性和结构的结合使得这样的模拟对于第一次进入MLOps的学习者有价值。

    随着我们继续这个系列，未来的章节将更仔细地看看：

    - 版本控制和可重现性的实践工具
    - ML系统的CI/CD管道
    - 来自行业的真实案例研究
    - 性能监控
    - LLMOps的特殊考虑
    - 端到端项目

    我们还将介绍支持MLOps和LLMOps的不断发展的工具和服务堆栈。但正如我们在整个过程中重申的，工具来来去去。保持不变的是基础原则：为可重现性而构建，为自动化而设计，为监控而计划，始终为可靠性而工程。

    这就是为什么，正如在前一部分也强调的，这门课程严重倾向于理论。实现细节可能因用例、规模和行业而异。但如果你深入理解底层系统设计和生命周期原则，你将有能力导航任何MLOps堆栈或适应任何LLMOps场景。

    所以，当我们前进时，期望看到理论、方法和轻量级模拟的持续结合，弥合实验和生产之间的差距。目标是帮助你培养成熟的、以系统为中心的思维方式，将机器学习不视为独立工件，而是更广泛软件生态系统的活的一部分。
    """
    )
    return


if __name__ == "__main__":
    app.run()
