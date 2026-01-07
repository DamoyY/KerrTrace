# KerrTrace - Real-time Kerr Black Hole Ray Tracer

![diagram](diagram.png)

**KerrTrace** 是一个基于 Python 和 CUDA 的实时广义相对论光线追踪器（GRRT）。它利用 NVIDIA GPU 的并行计算能力，通过数值积分求解克尔时空（旋转黑洞）中的光子测地线方程，模拟出黑洞吸积盘的视觉效果。

该项目实现了物理上精确的视觉效果，包括引力透镜、多普勒频移、引力红移以及吸积盘的黑体辐射光谱。

## 特性 (Features)

*   **实时渲染**：利用 `CuPy` 和 CUDA C++ 编写的高性能内核，实现高帧率渲染。
*   **克尔度规 (Kerr Metric)**：模拟旋转黑洞，包括事件视界和光子球的非凡几何结构。
*   **Novikov-Thorne 吸积盘**：基于广义相对论薄盘模型，计算吸积盘的温度分布。
*   **物理光谱渲染**：
    *   预计算黑体辐射 LUT（查找表）。
    *   将温度映射到 CIE XYZ 色彩空间，再转换为线性 RGB。
    *   考虑了相对论效应导致的光谱频移。
*   **高质量图像处理**：
    *   ACES Tone Mapping（色调映射）以处理高动态范围亮度。
    *   SSAA（超级采样抗锯齿）。
*   **交互式漫游**：支持类似游戏的第一人称相机控制。

## 依赖环境 (Requirements)

你需要拥有一个支持 CUDA 的 NVIDIA 显卡才能运行此项目。

*   **OS**: Windows / Linux
*   **Python**
*   **CUDA Toolkit**

### Python 库
*   `cupy` (用于 CUDA 加速)
*   `pyglet` (用于窗口显示和输入处理)
*   `numpy` (用于基础数学运算)

## 安装 (Installation)

1.  **克隆仓库**
    ```bash
    git clone https://github.com/DamoyY/KerrTrace.git
    cd KerrTrace
    ```

2.  **安装 Python 依赖**
    请根据你的 CUDA 版本安装对应的 `cupy`。例如，如果你使用的是 CUDA 12.x：
    ```bash
    pip install cupy-cuda12x pyglet numpy
    ```

## 运行 (Usage)

在项目根目录下运行主程序：

```bash
python black_hole.py
```

首次运行时，程序会编译 CUDA 内核（可能需要几秒钟），并生成黑体辐射查找表。随后会弹出一个窗口显示黑洞。程序会自动将第一帧保存为 `first_frame.png`。

## 控制说明 (Controls)

| 按键 / 操作 | 功能 |
| :--- | :--- |
| **鼠标左键** | 点击窗口以锁定鼠标（进入沉浸模式） |
| **ESC** | 解锁鼠标 / 退出控制模式 |
| **鼠标移动** | 旋转视角 (Pitch/Yaw) |
| **W / S** | 前进 / 后退 |
| **A / D** | 向左 / 向右平移 |
| **Space** | 垂直上升 |
| **Ctrl** | 垂直下降 |
| **Shift** | 加速移动 |
| **鼠标滚轮** | 调整视场角 (FOV) / 变焦 |

## 物理与技术细节 (Technical Details)

### 1. 核心积分器
项目使用 **Cash-Karp 方法**（一种自适应步长的 Runge-Kutta 方法）来数值积分光子的运动方程。这确保了光线在极度弯曲的时空（如黑洞视界附近）中的轨迹精度。

### 2. 吸积盘模型
吸积盘的亮度并非简单的纹理，而是基于 **Novikov-Thorne** 模型计算出的物理温度：
*   计算最内层稳定圆轨道 (ISCO)。
*   根据径向距离计算辐射通量。
*   利用普朗克定律将温度转换为颜色。

### 3. CUDA 渲染管线
*   **Ray Marching**: 每个像素发射光线，步进求解微分方程。
*   **求交检测**: 检测光线是否落入视界（黑色）或击中吸积盘。
*   **后期处理**: 在 Kernel 内部直接完成 ACES 色调映射和 Gamma 校正。

## 许可证

本项目遵循 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议。详情请参阅 [LICENSE](LICENSE) 文件。