# 本地 GPU 学习环境

可选的本地训练环境配置，环境名称为 `dl-gpu`。Python 3.11，PyTorch 2.9.1 / CUDA 12.8 wheel，torchvision 0.24.1，配套 NumPy、SciPy、Matplotlib、pandas、scikit-learn、Gymnasium classic-control、Stable Baselines3、ipykernel、TensorBoard。

网站本身不需要此环境。该 YAML 面向使用兼容 NVIDIA 显卡及驱动的 CUDA 环境；其他系统或仅使用 CPU 的读者，请按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择安装组合。

## 使用

先在本目录创建环境，再激活：

```powershell
conda env create -f environment.yml
conda activate dl-gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

运行本目录的真实 GPU 训练示例：

```powershell
python train_gpu.py --steps 200 --output-dir gpu-training-output
```

示例在 CUDA 上训练一个 16→32→2 的分类网络，合成数据包含独立训练和验证样本。保存 `result.json` 与实际训练曲线 `loss.png`。CUDA 不可用时直接报错，不静默回退到 CPU。此例用于验证环境功能，不是公开数据集成绩，也不代表复杂任务的泛化性能。

如需在 Notebook 中使用，可自行注册内核：

```powershell
python -m ipykernel install --user --name dl-gpu --display-name "Python (dl-gpu, CUDA)"
```

使用 Gymnasium 时，注意分别处理 `terminated` 与 `truncated`。

网站中的网格 Q-learning 是浏览器 CPU 教学实验；访问在线网站不会调用这台电脑的 GPU。本地 Python 环境供你自行运行训练代码。小模型与短任务不一定在 GPU 上更快。

YAML 固定了 Python 与 PyTorch 核心版本；其他依赖在创建环境时重新解析，并非完整锁定文件。运行结果生成在指定输出目录中，不属于仓库源码。

官方依据：[PyTorch 安装组合](https://pytorch.org/get-started/previous-versions/)、[Gymnasium 基础接口](https://gymnasium.farama.org/introduction/basic_usage/)。
