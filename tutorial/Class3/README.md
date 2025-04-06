## from FNN to CNN

基于MNIST数据集实现了卷积神经网络手写数字识别

**数据集来源**：torch内置可下载

**用法**：在命令行/终端输入

```
    pip install torch torchvision torchaudio (无 CUDA support)
	(如果你的显卡支持CUDA，请访问https://developer.nvidia.com/cuda-toolkit获得对应版本的命令 )
	pip install matplotlib numpy
	python <程序名>
```

**程序结构**：

```
CNN-model.py
│── Net (卷积神经网络)
│   ├── __init__()  (初始化 CNN 结构)
│   ├── forward()  (前向传播)
│── Tester (测试模型)
│   ├── __init__()  (初始化测试器)
│   ├── evaluate()  (计算模型在测试集上的准确率)
│   ├── show_predictions()  (可视化预测结果,可调参)
│── Model (训练模型)
│   ├── __init__()  (初始化训练器)
│   ├── train()  (训练模型)
│   ├── plot_loss()  (绘制训练损失曲线)
│── __main__ (主程序入口)
│   ├── model = Model()  (创建模型实例)
│   ├── model.train()  (执行训练)
│   ├── model.plot_loss()  (绘制损失曲线)
│   ├── model.tester.show_predictions()  (展示预测结果)
```

**注：**

- CNN与FNN的对比可以参考 `fixed_FNN-model.py` 对比 `CNN-model.py`
- 在 `fixed_FNN-model.py` 的注释中展示了用 `torch.nn.sequential` 直接传入模型的写法
- 更多细节请阅读注释

**Acknowledgement: QJJ 同学，纠正了Linear()后接ReLU()的错误和Softmax()在交叉熵中已有运算的重复 **
