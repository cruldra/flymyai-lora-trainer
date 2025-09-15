import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r"""
        # 自编码器 — 深度学习要点 #1

        ![特色图片](https://hackernoon.imgix.net/hn-images/1*8ixTe1VHLsmKB3AquWdxpQ.png?w=1200)

        **作者：** [Julien Despois](https://hackernoon.com/u/juliendespois)  
        **职位：** 深度学习工程师  
        **发布时间：** 2017年2月7日  
        **来源：** [HackerNoon](https://hackernoon.com/)

        **特色内容：** 数据压缩、图像重建和分割（附实例！）

        ---

        在"**深度学习要点**"系列中，我们将**不会**像在[**A.I. Odyssey**](https://medium.com/@juliendespois/talk-to-you-computer-with-you-eyes-and-deep-learning-a-i-odyssey-part-2-7d3405ab8be1)中那样看到如何使用深度学习端到端地解决复杂问题。相反，我们将研究不同的技术，以及一些**示例和应用**。

        > **如果您喜欢人工智能，请务必[订阅新闻通讯](http://eepurl.com/cATXvT)以接收文章更新和更多内容！**
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 引言

        ### 什么是自编码器？

        神经网络有各种形状和大小，通常以其**输入**和**输出数据类型**为特征。例如，图像分类器是用*卷积神经网络*构建的。它们以**图像**作为输入，输出**类别的概率分布**。

        *自编码器（AE）*是一类神经网络，其**输入与输出相同***。它们通过将输入压缩为*潜在空间表示*，然后从这种表示重建输出来工作。

        *我们将看到使用输入的修改版本如何更加有趣*

        ![简单自编码器架构](https://hackernoon.imgix.net/hn-images/1*-5D-CBTusUnmsbA6VYdY3A.png?w=1200)

        **简单自编码器架构** — 输入被压缩然后重建

        ### 卷积自编码器

        自编码器的一个非常流行的用途是将它们应用于图像。**技巧**是用*卷积*层替换*全连接*层。这些层与池化层一起，将输入从**宽而薄**（比如100 x 100像素，3个通道 — RGB）转换为**窄而厚**。这有助于网络从图像中提取**视觉特征**，从而获得更准确的潜在空间表示。重建过程使用*上采样*和卷积。

        生成的网络称为*卷积自编码器*（*CAE*）。

        ![卷积自编码器架构](https://hackernoon.imgix.net/hn-images/1*8ixTe1VHLsmKB3AquWdxpQ.png?w=1200)

        **卷积自编码器架构** — 它将宽而薄的输入空间映射到窄而厚的潜在空间

        ### 重建质量

        输入图像的*重建*通常是**模糊的**和**质量较低的**。这是*压缩*的结果，在此过程中我们**丢失了一些信息**。

        ![CAE训练重建输入](https://hackernoon.imgix.net/hn-images/1*fmlPZYhXSAe8FAISFC_fKg.png?w=1200)

        **CAE被训练来重建其输入**

        ![重建图像模糊](https://hackernoon.imgix.net/hn-images/1*kiDjGCJaKbkl6lS-BuL4NA.png?w=1200)

        **重建的图像是模糊的**
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## CAE的使用

        ### 示例1：超基础图像重建

        卷积自编码器可用于重建。例如，它们可以学习[从图片中去除噪声](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)，或重建缺失的部分。

        为此，我们不使用相同的图像作为输入和输出，而是使用**噪声版本作为输入**和**干净版本作为输出**。通过这个过程，网络学会填补图像中的空白。

        让我们看看CAE能做什么来**替换眼部图像的一部分**。*假设有一个十字准线，我们想要移除它*。我们可以手动创建数据集，这非常方便。

        ![CAE训练移除十字准线](https://hackernoon.imgix.net/hn-images/1*q-RoBbB9Bbaqt73540qfAw.png?w=1200)

        **CAE被训练来移除十字准线**

        ![重建输入没有十字准线](https://hackernoon.imgix.net/hn-images/1*PX5e64QQw-RZOmG9EWWsVA.png?w=1200)

        **尽管模糊，重建的输入没有剩余的十字准线**

        > *现在我们的自编码器已经训练好了，我们可以用它来移除我们**从未见过**的眼部图片上的十字准线！*

        ### 示例2：超基础图像着色

        在这个例子中，CAE将学习从圆形和正方形的图像*映射*到相同的图像，但**圆形**着色为**红色**，**正方形**着色为**蓝色**。

        ![CAE训练着色图像](https://hackernoon.imgix.net/hn-images/1*qygLsStGgy1zFf1EtDfIhw.png?w=1200)

        **CAE被训练来着色图像**

        ![重建着色结果](https://hackernoon.imgix.net/hn-images/1*PM7vnXw4gOw-61r5ccPEgA.png?w=1200)

        **尽管重建是模糊的，颜色大部分是正确的**

        CAE在*着色*图像的**正确**部分方面做得**相当好**。它已经理解了*圆形*是*红色*的，*正方形*是*蓝色*的。**紫色**来自蓝色和红色的混合，网络在圆形和正方形之间*犹豫*的地方。

        > *现在我们的自编码器已经训练好了，我们可以用它来着色我们**从未见过的**图片！*
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 高级应用

        上面的例子只是*概念验证*，展示了卷积自编码器能做什么。

        更令人兴奋的应用包括[完整图像着色](http://richzhang.github.io/colorization/)、[潜在空间聚类](http://proceedings.mlr.press/v48/xieb16.pdf)，或[生成更高分辨率的图像](https://arxiv.org/pdf/1501.00092v3.pdf)。后者是通过使用低分辨率作为输入和高分辨率作为输出来获得的。

        ![彩色图像着色](https://hackernoon.imgix.net/hn-images/1*a1Ffag3qwW3fMqVlub1emA.png?w=1200)

        **彩色图像着色** — 作者：Richard Zhang, Phillip Isola, Alexei A. Efros

        ![神经增强](https://hackernoon.imgix.net/hn-images/1*Bc5mDzppsH14ZZszz8YqCw.png?w=1200)

        **神经增强** — 作者：[Alexjc](https://github.com/alexjc/neural-enhance)

        ## 结论

        在这篇文章中，我们看到了如何使用*自编码器神经网络*来压缩、重建和清理数据。获得图像作为输出是非常令人兴奋的事情，**非常有趣**。

        **注意：** *有一个修改版本的AE叫做* **变分自编码器**，*用于图像生成，但我把它留到以后。*

        > **如果您喜欢人工智能，请务必[订阅新闻通讯](http://eepurl.com/cATXvT)以接收文章更新和更多内容！**

        ### 代码资源

        您可以在这里使用代码：

        🔗 **GitHub仓库：** [despoisj/ConvolutionalAutoencoder](https://github.com/despoisj/ConvolutionalAutoencoder)  
        **描述：** Keras/Tensorflow中卷积自编码器应用的快速示例

        ### 关键要点总结

        #### 🧠 **自编码器基础**
        - **定义**：输入与输出相同的神经网络
        - **工作原理**：压缩输入到潜在空间，然后重建
        - **特点**：学习数据的压缩表示

        #### 🖼️ **卷积自编码器（CAE）**
        - **架构**：使用卷积层替代全连接层
        - **优势**：更好地处理图像数据
        - **过程**：宽薄输入 → 窄厚潜在空间 → 重建输出

        #### 🎯 **应用场景**
        - **图像去噪**：移除图像中的噪声
        - **图像修复**：填补缺失部分
        - **图像着色**：为黑白图像添加颜色
        - **超分辨率**：提高图像分辨率
        - **数据压缩**：高效的数据表示

        #### ⚠️ **局限性**
        - **重建质量**：通常模糊且质量较低
        - **信息丢失**：压缩过程中不可避免的信息损失
        - **计算复杂度**：需要大量计算资源

        ### 未来展望

        自编码器是深度学习中的重要工具，为更高级的技术如变分自编码器（VAE）和生成对抗网络（GAN）奠定了基础。

        ---

        **原文链接：** [Autoencoders — Deep Learning bits #1](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694)

        感谢阅读这篇文章，敬请期待更多内容！
        """
    )
    return


if __name__ == "__main__":
    app.run()
