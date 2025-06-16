# 机器人迷宫寻路强化学习项目

本项目实现了一个基于强化学习的机器人迷宫寻路系统，使用PyBullet进行物理模拟，通过Stable Baselines 3实现强化学习算法。

## 项目结构
```bash
robots_projects/
├── config/                     # 配置文件目录
│   ├── maze_search/            # 迷宫搜索相关配置
│   │   └── __init__.py
│   ├── __init__.py
│   ├── curriculum_config.py    # 课程学习配置
│   └── default_config.py       # 默认配置
│
├── envs/                       # 环境模块目录
│   ├── mujoco/                 # MuJoCo相关环境
│   ├── pybullet/               # PyBullet相关代码
│   │   ├── __init__.py
│   │   ├── maze_builder.py     # 迷宫构建器
│   │   └── maze_search.py      # 迷宫搜索环境
│   └── __init__.py
│
├── evaluation/                 # 评估模块
│   ├── __init__.py
│   └── evaluator.py            # 模型评估器
│
├── reward_function/            # 奖励函数模块
│   ├── __init__.py
│   └── maze_search_reward.py   # 迷宫搜索的奖励函数
│
├── training/                   # 训练模块
│   ├── __init__.py
│   ├── curriculum.py           # 课程学习实现
│   └── trainer.py              # 训练器
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── logger.py               # 日志工具
│   └── visualization.py        # 可视化工具
│
├── main.py                     # 主入口脚本
└── README.md                   # 项目说明
```


## 环境要求

- Python 3.8+
- PyBullet
- Gymnasium
- Stable Baselines 3
- NumPy
- Matplotlib
- Pandas
- Seaborn

## 安装依赖

```bash
pip install -r requirements.txt
```
## 使用方法
### 测试环境
```bash
python main.py test [--episodes N] [--no-render] [--verbose]
```
### 训练模型
```bash
python main.py train [--algorithm {SAC,PPO,A2C}] [--timesteps N] [--render] [--verbose]
```
### 使用课程学习训练
```
python main.py curriculum [--render] [--resume] [--verbose]
```
### 评估模型
```
python main.py eval --model MODEL_PATH [--episodes N] [--no-render] [--verbose]
```

## 课程学习
### 本项目实现了三阶段课程学习：

- 平衡学习：机器人学习在迷宫中保持平衡
- 短距离导航：机器人学习在保持平衡的同时导航到短距离目标
- 长距离导航：机器人学习在保持平衡的同时导航到长距离目标

## 奖励函数
*奖励函数综合考虑了以下因素*：

- 距离奖励：接近目标的奖励
- 平衡奖励：保持平衡的奖励
- 碰撞惩罚：碰撞墙壁的惩罚
- 能量效率：最小化动作幅度的奖励
- 摔倒惩罚：机器人倾斜过大的惩罚
- 速度奖励：保持前进速度的奖励
- 超时惩罚：回合超时的惩罚

## 许可证
**MIT**
## 总结

以上是我根据您的项目结构图设计的模块化代码结构。这种设计有以下优点：

1. **模块化**：将代码分解为独立的模块，每个模块负责特定的功能
2. **可维护性**：每个文件都有明确的职责，便于维护和更新
3. **可扩展性**：可以轻松添加新功能或修改现有功能
4. **可读性**：代码结构清晰，易于理解
5. **可复用性**：模块化设计使代码可以在其他项目中复用

这个设计实现了您所需的所有功能，包括：
- 迷宫环境模拟
- 强化学习训练
- 课程学习
- 模型评估
- 可视化工具

您可以根据实际需求进一步调整代码细节。
