python IGD/sample2.py \
  --config RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --dataset imagenet \
  --imagenet_dir /data/wlf/datasets/imagenette \
  --spec nette \
  --gamma 50 \
  --k 5.0 \
  --num-sampling_steps 50 \
  --high 400 \
  --low 100 \
  --save-dir ./results/igd_time_window \
  --num-samples 10


在这种情况下，准确率：60%


请仔细研究我的项目，这是对于IGD的改进。任务目标是数据集蒸馏，
IGD的做法是让生成样本产生的梯度约等于真实样本产生的梯度。或者说让生成样本对模型的更新效果与真实数据一致。
这里就涉及到了对x求二阶导，x又是经过复杂的decoder后得到的结果，梯度传递可能出现问题。
我做出了一些改进尝试缓解这个问题：
sample_mp:
  原始的IGD采样
sample:
  首先我将latent space替换成了dinov2，这个语义空间中同一类别的样本更加聚集，我期望“在这个空间中采样的样本含有更多的语义信息”。但是目前似乎并没有效果
sample2
  其次我在这个latent space中训练了一个代理模型，用它来指导latent code的方向，这样就不需要将外部的梯度传进来了
  但是我遇到的问题：z倾向于前往类别中心，代理模型的梯度也只是简单的指向了类别中心，这样采样出来的数据就缺失了代表性。
  这个的效果不好
sample3
  基于sample2的问题，我添加了一个boundary loss，并将引导loss改成了像难样本移动，期望生成样本向决策边界移动。但是似乎并没有效果。
现在，让我们回到问题的本质，我希望能够找到一种方法，既能够让生成样本对模型的更新效果与真实数据一致，又不涉及二阶导
sample4
  请编写两个脚本，基于原始的IGD，实现以下思路：
  第一个脚本：
  冻结encoder(VAE)和下游分类器(ResNet)，构建一个浅层神经网络M(z),输入z，让M(z)预测 ResNet提取到的特征。
  ResNet加载官方权重然后在我的数据集上微调几个epoch就好。
  VAE的加载预训练权重，参考IGD的加载方式。训练完保存权重
  第二个脚本：
  采样时不再计算$\nabla_z \mathcal{L}(\text{ResNet}(D(z)))$，目标是让生成的$z$经过$M(z)$后，产生的特征与真实数据的ResNet 特征一致，
  $\nabla_z \mathcal{L}_{CE}(\text{Classifier\_Head}(M(z)), y)$。
  采样参考sample_mp

  即我这里对IGD的修改只有一处：获取指导梯度的方式
  

原始IGD 需要计算$\nabla_z (\text{ResNet}(D(z)))$，其中经过Decoder$D(z)$的反向传播（尤其是二阶导）非常不稳定且昂贵。如果有一个$M(z)$能直接预测$z$解码后的ResNet 特征，即$M(z) \approx \text{ResNet}(D(z))$。优化$z$让$M(z)$匹配目标，在数学期望上等价于优化$z$让$\text{ResNet}(x)$匹配目标。

原始的IGD 梯度传递路径是：$$\text{Gradient} = \frac{\partial \mathcal{L}}{\partial \text{Feat}} \times \underbrace{\frac{\partial \text{ResNet}}{\partial \text{Pixel}} \times \frac{\partial \text{Decoder}}{\partial z}}_{\text{不稳定且昂贵}}$$你提出的$M(z)$方案将路径简化为：$$\text{Gradient} \approx \frac{\partial \mathcal{L}}{\partial \text{Feat}} \times \underbrace{\frac{\partial M}{\partial z}}_{\text{稳定且高效}}$$只要你的$M(z)$在训练集上拟合得足够好（MSE Loss 足够低），$\frac{\partial M}{\partial z}$就会成为$\frac{\partial \text{ResNet}}{\partial \text{Pixel}} \times \frac{\partial \text{Decoder}}{\partial z}$的一个平滑且稳定的近似。这在深度学习中被称为“代理梯度（Surrogate Gradient）”或“蒸馏代理”。

训练M
```shell
python IGD/train_m.py   --data_path /data/wlf/datasets/imagenet/train --dataset imagenet --depth 10   --resnet_ckpt ./checkpoints_proxy/resnet10_finetuned.pth   --save_dir ./checkpoints_proxy   --epochs 20 --spec nette --nclass 10
```


