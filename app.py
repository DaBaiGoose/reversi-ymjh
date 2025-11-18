# 以下是 app.py 的完整内容

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import threading
import traceback
import sys
import subprocess
import glob
import tempfile
import shutil

# ==============================================================================
#  1. 动态加载/编译C++模块
# ==============================================================================
USE_CPP_EXTENSION = False
reversi_mcts_cpp = None
def load_cpp_extension():
    """尝试加载，如果失败则尝试编译，最终返回是否成功"""
    global USE_CPP_EXTENSION, reversi_mcts_cpp
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 尝试直接导入
    try:
        # 将当前目录加入 sys.path，确保能找到同级目录下的 .so 文件
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        import reversi_mcts_cpp
        USE_CPP_EXTENSION = True
        print("✅ 成功加载已存在的 C++ MCTS 扩展。")
        return True
    except ImportError:
        print("⚠️ 未找到可用的 C++ MCTS 扩展，准备尝试编译...")

    # 2. 尝试编译
    try:
        # 检查 pybind11
        try:
            import pybind11
        except ImportError:
            print("正在安装编译依赖 pybind11...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
            import pybind11
        
        # 确定源文件路径 (同级目录)
        cpp_source_path = os.path.join(current_dir, 'reversi_mcts.cpp')
        
        if not os.path.exists(cpp_source_path):
            print(f"❌ 编译失败：在 {current_dir} 下未找到 reversi_mcts.cpp 源文件。")
            return False

        print(f"正在编译 {cpp_source_path} ...")

        # 处理路径中的反斜杠，避免 f-string 语法错误
        cpp_source_path_for_setup = cpp_source_path.replace('\\', '/')
        
        # 清理旧的 .so 文件
        for f in glob.glob(os.path.join(current_dir, "reversi_mcts_cpp*.so")):
            try: os.remove(f)
            except: pass

        # 创建临时 setup.py
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_path = os.path.join(tmpdir, "setup.py")
            
            # 为了解决 Conda 环境下的链接问题，尝试添加系统库路径
            # 如果您在纯净环境，这行通常无害；如果在Conda环境，这能救命
            extra_link_args_str = "'-L/usr/lib/x86_64-linux-gnu'" 
            
            setup_content = f"""
import pybind11
from setuptools import setup, Extension
import os

setup(
    name='reversi_mcts_cpp',
    ext_modules=[Extension('reversi_mcts_cpp',
                          ['{cpp_source_path_for_setup}'],
                          include_dirs=[pybind11.get_include()],
                          extra_compile_args=['-std=c++11', '-O3', '-march=native', '-fPIC'],
                          # 尝试优先链接系统库，解决Conda libstdc++冲突
                          extra_link_args=[{extra_link_args_str}] if os.path.exists('/usr/lib/x86_64-linux-gnu') else [],
                          language='c++')],
    script_args=['build_ext', '--inplace'],
    zip_safe=False,
)
"""
            with open(setup_path, "w", encoding="utf-8") as f:
                f.write(setup_content)
            
            # 执行编译
            result = subprocess.run(
                [sys.executable, setup_path], 
                cwd=current_dir,  # 在当前目录下执行，生成的 .so 会在这里
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                print("❌ 编译错误信息:")
                print(result.stderr)
                return False

        # 检查编译产物
        so_files = glob.glob(os.path.join(current_dir, "reversi_mcts_cpp*.so"))
        if not so_files:
            print("❌ 编译看似成功，但未找到生成的 .so 文件。")
            return False
        
        print(f"✅ 编译成功: {os.path.basename(so_files[0])}")

        # 3. 再次尝试导入
        if "reversi_mcts_cpp" in sys.modules:
            del sys.modules["reversi_mcts_cpp"]
        
        import reversi_mcts_cpp
        USE_CPP_EXTENSION = True
        print("✅ C++ 扩展加载成功！")
        return True

    except Exception as e:
        print(f"❌ 加载/编译过程中发生异常: {e}")
        traceback.print_exc()
        return False

# 执行加载逻辑
load_cpp_extension()


# ==============================================================================
#  2. 神经网络模型
# ==============================================================================
class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=8):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        c = 128
        self.conv_input = nn.Conv2d(1, c, kernel_size=3, padding=1); self.bn_input = nn.BatchNorm2d(c)
        self.res1_conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1); self.bn1_1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1); self.bn1_2 = nn.BatchNorm2d(c)
        self.res2_conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1); self.bn2_1 = nn.BatchNorm2d(64)
        self.res2_conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1); self.bn2_2 = nn.BatchNorm2d(c)
        self.res3_conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1); self.bn3_1 = nn.BatchNorm2d(64)
        self.res3_conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1); self.bn3_2 = nn.BatchNorm2d(c)
        self.res4_conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1); self.bn4_1 = nn.BatchNorm2d(64)
        self.res4_conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1); self.bn4_2 = nn.BatchNorm2d(c)
        self.res5_conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1); self.bn5_1 = nn.BatchNorm2d(64)
        self.res5_conv2 = nn.Conv2d(64, c, kernel_size=3, padding=1); self.bn5_2 = nn.BatchNorm2d(c)
        self.policy_conv = nn.Conv2d(c, 1, kernel_size=1); self.value_gap = nn.AdaptiveAvgPool2d(1)
        self.value_fc  = nn.Linear(c, 128); self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        def residual_block(x, conv1, bn1, conv2, bn2):
            residual = x; x = F.relu(bn1(conv1(x))); x = bn2(conv2(x)); x = F.relu(x + residual); return x
        x = residual_block(x, self.res1_conv1, self.bn1_1, self.res1_conv2, self.bn1_2)
        x = residual_block(x, self.res2_conv1, self.bn2_1, self.res2_conv2, self.bn2_2)
        x = residual_block(x, self.res3_conv1, self.bn3_1, self.res3_conv2, self.bn3_2)
        x = residual_block(x, self.res4_conv1, self.bn4_1, self.res4_conv2, self.bn4_2)
        x = residual_block(x, self.res5_conv1, self.bn5_1, self.res5_conv2, self.bn5_2)
        pol = self.policy_conv(x).flatten(1); policy = F.softmax(pol, dim=1)
        val = self.value_gap(x).flatten(1); val = F.relu(self.value_fc(val)); value = torch.tanh(self.value_head(val))
        return policy, value
    
    def predict_batch(self, states_list, device):
        if not states_list: return [], []
        batch_size = len(states_list)
        states_array = np.array(states_list, dtype=np.float32).reshape(batch_size, self.board_size, self.board_size)
        states_tensor = torch.from_numpy(states_array).unsqueeze(1).to(device)
        with torch.no_grad():
            policies, values = self.forward(states_tensor)
        policies_list = [p.tolist() for p in policies.cpu()]
        values_list = values.squeeze(-1).cpu().tolist()
        return policies_list, values_list

# ==============================================================================
#  3. MCTS 实现 (双模式：C++优先，Python兜底)
# ==============================================================================

# --- Python 后备模式的节点 ---
class PyMCTSNode:
    def __init__(self, state, parent=None, prior_prob=0.0):
        self.state = state; self.parent = parent; self.prior_prob = prior_prob
        self.children = {}; self.visit_count = 0; self.total_value = 0.0

    def is_leaf(self): return len(self.children) == 0
    
    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                next_state_for_me = ReversiLogic.apply_move(self.state, action, 1)
                next_state_for_opponent = -next_state_for_me
                self.children[action] = PyMCTSNode(next_state_for_opponent, self, prob)
                
    def select_child(self, c_puct=1.5):
        best_score, best_action, best_child = -float('inf'), None, None
        for action, child in self.children.items():
            if child.visit_count == 0:
                ucb_score = c_puct * child.prior_prob * math.sqrt(self.visit_count + 1e-8)
            else:
                ucb_score = (-child.total_value / child.visit_count) + \
                            c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb_score > best_score: 
                best_score, best_action, best_child = ucb_score, action, child
        return best_action, best_child
        
    def update_recursive(self, value):
        if self.parent: self.parent.update_recursive(-value)
        self.visit_count += 1; self.total_value += value

# --- 主 MCTS 类 ---
class KataGoMCTS:
    def __init__(self, net, c_puct=1.5, num_simulations=2400):
        self.net = net; self.c_puct = c_puct; self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device); self.net.eval(); self.BOARD_SIZE = self.net.board_size

    def run(self, board_state, player):
        if USE_CPP_EXTENSION:
            return self._run_cpp(board_state, player)
        else:
            return self._run_python(board_state, player)

    def _run_cpp(self, board_state, player):
        state_flat = (board_state * player).flatten().astype(np.int32).tolist()
        
        def batch_inference_fn(states_list):
            try: return self.net.predict_batch(states_list, self.device)
            except Exception as e:
                print(f"!!! C++回调Python出错: {e}"); traceback.print_exc(); return [], []
        
        try:
            # 调用C++模块
            visit_counts_list = reversi_mcts_cpp.single_thread_run(
                self.BOARD_SIZE, self.c_puct, self.num_simulations, 16, state_flat, batch_inference_fn
            )
            return np.array(visit_counts_list)
        except Exception as e:
            print(f"!!! C++ MCTS 运行异常: {e}，正在切换到 Python 后备模式")
            return self._run_python(board_state, player)

    def _run_python(self, board_state, player):
        # 降低模拟次数以保证响应速度
        sims = 400 # Python模式下使用较少的模拟次数
        normalized_state = board_state * player
        root = PyMCTSNode(normalized_state)
        
        self._evaluate_and_expand_py(root)
        
        for _ in range(sims):
            node = root
            while not node.is_leaf():
                _, node = node.select_child(self.c_puct)
            self._evaluate_and_expand_py(node)
            
        visit_counts = np.zeros(self.BOARD_SIZE**2)
        for action, child in root.children.items(): 
            visit_counts[action] = child.visit_count
        return visit_counts

    def _evaluate_and_expand_py(self, node):
        if ReversiLogic.is_terminal(node.state):
            value = float(ReversiLogic.get_winner(node.state))
            node.update_recursive(value)
            return
            
        policy, value = self.net.predict_batch([node.state], self.device)
        policy, value = policy[0], value[0]
        
        legal_actions = ReversiLogic.get_legal_actions(node.state, 1)
        if legal_actions:
            legal_policy = np.array([policy[a] for a in legal_actions])
            if np.sum(legal_policy) > 1e-6: legal_policy /= np.sum(legal_policy)
            else: legal_policy = np.ones(len(legal_actions)) / len(legal_actions)
            node.expand(list(zip(legal_actions, legal_policy)))
            
        node.update_recursive(value)

# ==============================================================================
#  4. 无状态逻辑库 (辅助计算合法动作)
# ==============================================================================
class ReversiLogic:
    BOARD_SIZE = 8
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    @staticmethod
    def get_legal_actions(board_state, player):
        legal_actions = []
        for r in range(ReversiLogic.BOARD_SIZE):
            for c in range(ReversiLogic.BOARD_SIZE):
                if board_state[r, c] == 0:
                    if any(ReversiLogic._check_direction(board_state, r, c, dr, dc, player) for dr, dc in ReversiLogic.DIRECTIONS):
                        legal_actions.append(r * ReversiLogic.BOARD_SIZE + c)
        return legal_actions
    
    @staticmethod
    def apply_move(board_state, action, player):
        new_board = board_state.copy()
        row, col = action // ReversiLogic.BOARD_SIZE, action % ReversiLogic.BOARD_SIZE
        new_board[row, col] = player
        for dr, dc in ReversiLogic.DIRECTIONS:
            if ReversiLogic._check_direction(new_board, row, col, dr, dc, player):
                ReversiLogic._flip_direction(new_board, row, col, dr, dc, player)
        return new_board
    
    @staticmethod
    def is_terminal(board_state):
        return len(ReversiLogic.get_legal_actions(board_state, 1)) == 0 and \
               len(ReversiLogic.get_legal_actions(board_state, -1)) == 0
               
    @staticmethod
    def get_winner(board_state):
        black_count = np.sum(board_state == 1)
        white_count = np.sum(board_state == -1)
        if black_count > white_count: return 1
        if white_count > black_count: return -1
        return 0
        
    @staticmethod
    def _check_direction(board_state, r_start, c_start, dr, dc, player):
        r, c = r_start + dr, c_start + dc
        if not (0 <= r < ReversiLogic.BOARD_SIZE and 0 <= c < ReversiLogic.BOARD_SIZE and board_state[r, c] == -player):
            return False
        r += dr; c += dc
        while 0 <= r < ReversiLogic.BOARD_SIZE and 0 <= c < ReversiLogic.BOARD_SIZE:
            if board_state[r, c] == player: return True
            if board_state[r, c] == 0: return False
            r += dr; c += dc
        return False
        
    @staticmethod
    def _flip_direction(board, r_start, c_start, dr, dc, player):
        r, c = r_start + dr, c_start + dc
        to_flip = []
        while 0 <= r < ReversiLogic.BOARD_SIZE and 0 <= c < ReversiLogic.BOARD_SIZE:
            if board[r, c] == -player:
                to_flip.append((r, c))
            elif board[r, c] == player:
                for fr, fc in to_flip: board[fr, fc] = player
                return
            else: return
            r += dr; c += dc

# ==============================================================================
#  5. Flask应用设置
# ==============================================================================
app = Flask(__name__)
ai_mcts = None
is_model_loading = False
model_lock = threading.Lock()
MODEL_PATH = "./reversi_model.pth"
last_model_mtime = 0

def get_model_timestamp():
    if not os.path.exists(MODEL_PATH): return "模型文件未找到", 0
    mtime = os.path.getmtime(MODEL_PATH)
    return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'), mtime

def check_model_updated():
    global last_model_mtime
    if not os.path.exists(MODEL_PATH): return False
    return os.path.getmtime(MODEL_PATH) > last_model_mtime

def load_ai_model():
    global ai_mcts, is_model_loading, last_model_mtime
    with model_lock:
        if is_model_loading: return
        is_model_loading = True
    
    timestamp_str, current_mtime = get_model_timestamp()
    print(f"开始加载AI模型... (文件时间: {timestamp_str})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        net = AlphaZeroNet()
        if os.path.exists(MODEL_PATH):
            try: 
                checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
                net.load_state_dict(checkpoint['model_state_dict'])
            except Exception: 
                net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        else: 
            print("警告: 模型文件不存在，使用随机初始化的模型")
        
        # 2400次模拟对于C++来说很快，对于Python来说太慢，_run_python会自动降级
        ai_mcts = KataGoMCTS(net, num_simulations=2400, c_puct=1.5)
        last_model_mtime = current_mtime
        print(f"AI模型加载完毕，使用设备: {device}")
        
    except Exception as e: 
        print(f"模型加载失败: {e}")
    finally: 
        is_model_loading = False

def check_and_reload_model_if_needed():
    if check_model_updated() and not is_model_loading:
        print("检测到模型文件已更新，启动后台重新加载...")
        threading.Thread(target=load_ai_model, daemon=True).start()

# ==============================================================================
#  6. API路由
# ==============================================================================
@app.route('/')
def home():
    check_and_reload_model_if_needed()
    timestamp_str, _ = get_model_timestamp()
    return render_template('index.html', model_time=timestamp_str)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    if is_model_loading or not ai_mcts:
        return jsonify({'success': False, 'message': 'AI模型仍在加载中或加载失败，请稍候...'})

    data = request.json
    board_state = np.array(data['board'], dtype=np.int8)
    ai_color = data.get('player', -1)

    legal_actions = ReversiLogic.get_legal_actions(board_state, ai_color)
    if not legal_actions:
        return jsonify({'success': True, 'ai_move': None, 'message': 'AI无棋可走'})
    
    mode = "C++ MCTS" if USE_CPP_EXTENSION else "Python MCTS"
    print(f"AI (Color: {ai_color}) 正在使用 {mode} 思考...")
    
    # 运行MCTS
    visit_counts = ai_mcts.run(board_state, ai_color)
    
    # 从合法动作中选取最好的
    valid_visit_counts = {action: visit_counts[action] for action in legal_actions}
    ai_action = max(valid_visit_counts, key=valid_visit_counts.get)
    
    ai_row = int(ai_action // ReversiLogic.BOARD_SIZE)
    ai_col = int(ai_action % ReversiLogic.BOARD_SIZE)
    
    print(f"AI选择移动到: ({ai_row}, {ai_col})")
    return jsonify({'success': True, 'ai_move': {'row': ai_row, 'col': ai_col}})

# ==============================================================================
#  7. 启动服务器
# ==============================================================================
if __name__ == '__main__':
    print("启动黑白棋Web服务器...")
    # 启动后台加载模型线程
    threading.Thread(target=load_ai_model, daemon=True).start()
    app.run(host='0.0.0.0', port=8082, debug=False)