# path_config.py (放在项目根目录)
import os

# 项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 其他常用目录
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
ENVS_DIR = os.path.join(PROJECT_ROOT, 'envs')
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
# ...其他目录
