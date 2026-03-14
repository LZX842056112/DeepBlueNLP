#!/usr/bin/env python3
"""
RoPE 旋转位置编码实现

**参考论文**: RoFormer: Enhanced Transformer with Rotary Position Embedding
"""

import math
from typing import List, Tuple


# ==================== RoPE 实现 ====================

class RoPE:
    """
    旋转位置编码 (Rotary Position Embedding)
    
    通过旋转矩阵编码位置信息
    """
    
    def __init__(self, dim: int, base: int = 10000):
        """
        初始化
        
        参数:
            dim: 嵌入维度（必须是偶数）
            base: 频率基数
        """
        assert dim % 2 == 0, "dim 必须是偶数"
        self.dim = dim
        self.base = base
        self.inv_freq = self._compute_inv_freq()
    
    def _compute_inv_freq(self) -> List[float]:
        """计算逆频率"""
        # θ_i = base^(-2i/dim), i = 0, 1, ..., dim/2-1
        inv_freq = []
        for i in range(0, self.dim, 2):
            inv_freq.append(1.0 / (self.base ** (i / self.dim)))
        return inv_freq
    
    def _rotate(self, x: List[float]) -> List[float]:
        """
        旋转向量（将向量旋转 90 度）
        
        对于向量 [x0, x1, x2, x3, ...]
        旋转后为 [-x1, x0, -x3, x2, ...]
        """
        rotated = []
        for i in range(0, len(x), 2):
            rotated.append(-x[i + 1])
            rotated.append(x[i])
        return rotated
    
    def _apply_rotary(self, x: List[float], freqs: List[float]) -> List[float]:
        """
        应用旋转
        
        参数:
            x: 输入向量
            freqs: 频率对应的 cos 和 sin 值
        
        返回:
            旋转后的向量
        """
        output = []
        for i in range(0, len(x), 2):
            cos = freqs[i // 2]
            sin = freqs[len(freqs) // 2 + i // 2]
            
            # 旋转公式
            # [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
            x0, x1 = x[i], x[i + 1]
            output.append(x0 * cos - x1 * sin)
            output.append(x0 * sin + x1 * cos)
        
        return output
    
    def compute_freqs(self, max_len: int) -> Tuple[List[List[float]], List[List[float]]]:
        """
        计算频率（cos 和 sin）
        
        参数:
            max_len: 最大序列长度
        
        返回:
            (cos 矩阵，sin 矩阵)
        """
        cos_matrix = []
        sin_matrix = []
        
        for pos in range(max_len):
            cos_row = []
            sin_row = []
            
            for freq in self.inv_freq:
                theta = pos * freq
                cos_row.append(math.cos(theta))
                sin_row.append(math.sin(theta))
            
            cos_matrix.append(cos_row)
            sin_matrix.append(sin_row)
        
        return cos_matrix, sin_matrix
    
    def apply_rope(self, x: List[float], pos: int) -> List[float]:
        """
        对单个向量应用 RoPE
        
        参数:
            x: 输入向量 (dim,)
            pos: 位置
        
        返回:
            旋转后的向量
        """
        # 计算该位置的频率
        cos = []
        sin = []
        for freq in self.inv_freq:
            theta = pos * freq
            cos.append(math.cos(theta))
            sin.append(math.sin(theta))
        
        # 合并 cos 和 sin
        freqs = cos + sin
        
        # 应用旋转
        return self._apply_rotary(x, freqs)
    
    def apply_rope_qk(self, q: List[List[float]], k: List[List[float]], 
                      positions: List[int] = None) -> Tuple[List[List[float]], List[List[float]]]:
        """
        对 Q 和 K 应用 RoPE
        
        参数:
            q: Query 矩阵 [seq_len, dim]
            k: Key 矩阵 [seq_len, dim]
            positions: 位置列表（可选）
        
        返回:
            (rotated_q, rotated_k)
        """
        seq_len = len(q)
        
        if positions is None:
            positions = list(range(seq_len))
        
        # 计算频率
        max_pos = max(positions) + 1
        cos_matrix, sin_matrix = self.compute_freqs(max_pos)
        
        # 应用 RoPE
        rotated_q = []
        for i in range(seq_len):
            pos = positions[i]
            q_rot = self._apply_rotary(q[i], cos_matrix[pos] + sin_matrix[pos])
            rotated_q.append(q_rot)
        
        rotated_k = []
        for i in range(seq_len):
            pos = positions[i]
            k_rot = self._apply_rotary(k[i], cos_matrix[pos] + sin_matrix[pos])
            rotated_k.append(k_rot)
        
        return rotated_q, rotated_k
    
    def __repr__(self):
        return f"RoPE(dim={self.dim}, base={self.base})"


# ==================== RoPE 与 Attention 结合 ====================

class RoPEAttention:
    """
    使用 RoPE 的 Self-Attention
    """
    
    def __init__(self, dim: int, n_heads: int = 8, base: int = 10000):
        """
        初始化
        
        参数:
            dim: 总维度
            n_heads: 注意力头数
            base: RoPE 频率基数
        """
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = RoPE(self.head_dim, base)
        
        # 模拟权重矩阵（实际中通过训练学习）
        self.W_q = self._init_weight(dim, dim)
        self.W_k = self._init_weight(dim, dim)
        self.W_v = self._init_weight(dim, dim)
        self.W_o = self._init_weight(dim, dim)
    
    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重矩阵（模拟）"""
        import random
        random.seed(42)
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]
    
    def _matmul(self, x: List[float], w: List[List[float]]) -> List[float]:
        """矩阵乘法：x @ w"""
        out_dim = len(w[0])
        result = []
        for j in range(out_dim):
            val = sum(x[i] * w[i][j] for i in range(len(x)))
            result.append(val)
        return result
    
    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """
        分割注意力头
        
        参数:
            x: [seq_len, dim]
        
        返回:
            [n_heads, seq_len, head_dim]
        """
        seq_len = len(x)
        heads = []
        
        for h in range(self.n_heads):
            head_data = []
            for seq_idx in range(seq_len):
                start = h * self.head_dim
                end = start + self.head_dim
                head_data.append(x[seq_idx][start:end])
            heads.append(head_data)
        
        return heads
    
    def _concat_heads(self, heads: List[List[List[float]]]) -> List[List[float]]:
        """
        拼接注意力头
        
        参数:
            heads: [n_heads, seq_len, head_dim]
        
        返回:
            [seq_len, dim]
        """
        seq_len = len(heads[0])
        output = []
        
        for seq_idx in range(seq_len):
            token_vec = []
            for h in range(self.n_heads):
                token_vec.extend(heads[h][seq_idx])
            output.append(token_vec)
        
        return output
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax 函数"""
        max_val = max(x)
        exp_x = [math.exp(val - max_val) for val in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def forward(self, x: List[List[float]], positions: List[int] = None) -> List[List[float]]:
        """
        前向传播
        
        参数:
            x: 输入 [seq_len, dim]
            positions: 位置列表
        
        返回:
            输出 [seq_len, dim]
        """
        seq_len = len(x)
        
        if positions is None:
            positions = list(range(seq_len))
        
        # 1. 线性投影
        q = [self._matmul(x[i], self.W_q) for i in range(seq_len)]
        k = [self._matmul(x[i], self.W_k) for i in range(seq_len)]
        v = [self._matmul(x[i], self.W_v) for i in range(seq_len)]
        
        # 2. 分割注意力头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        
        # 3. 应用 RoPE（只对 Q 和 K）
        for h in range(self.n_heads):
            q_heads[h], k_heads[h] = self.rope.apply_rope_qk(
                q_heads[h], k_heads[h], positions
            )
        
        # 4. Self-Attention
        attn_outputs = []
        for h in range(self.n_heads):
            # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
            head_dim = self.head_dim
            scale = math.sqrt(head_dim)
            
            head_output = []
            for i in range(seq_len):
                # 计算注意力分数
                scores = []
                for j in range(seq_len):
                    score = sum(q_heads[h][i][d] * k_heads[h][j][d] for d in range(head_dim))
                    scores.append(score / scale)
                
                # Softmax
                attn_weights = self._softmax(scores)
                
                # 加权求和
                output_vec = [0.0] * head_dim
                for j in range(seq_len):
                    for d in range(head_dim):
                        output_vec[d] += attn_weights[j] * v_heads[h][j][d]
                
                head_output.append(output_vec)
            
            attn_outputs.append(head_output)
        
        # 5. 拼接注意力头
        concat_output = self._concat_heads(attn_outputs)
        
        # 6. 输出投影
        output = [self._matmul(concat_output[i], self.W_o) for i in range(seq_len)]
        
        return output
    
    def __repr__(self):
        return f"RoPEAttention(dim={self.dim}, n_heads={self.n_heads})"


# ==================== 测试函数 ====================

def test_rope_basic():
    """测试 RoPE 基础功能"""
    print("=" * 60)
    print("测试：RoPE 基础功能")
    print("=" * 60)
    
    # 创建 RoPE
    rope = RoPE(dim=4, base=10000)
    print(f"\nRoPE: {rope}")
    print(f"逆频率：{[f'{f:.6f}' for f in rope.inv_freq]}")
    
    # 测试单个向量
    x = [0.1, 0.2, 0.3, 0.4]
    print(f"\n输入向量：{x}")
    
    # 应用 RoPE
    for pos in range(5):
        rotated = rope.apply_rope(x, pos)
        print(f"位置 {pos}: {[f'{v:.4f}' for v in rotated]}")
    
    print("\n观察:")
    print("1. 位置 0 时，向量不变（旋转 0 度）")
    print("2. 随着位置增加，向量逐渐旋转")
    print("3. 向量的模长保持不变")
    

def test_rope_attention():
    """测试 RoPE Attention"""
    print("=" * 60)
    print("测试：RoPE Attention")
    print("=" * 60)
    
    # 创建 RoPE Attention
    attn = RoPEAttention(dim=8, n_heads=2)
    print(f"\nAttention: {attn}")
    
    # 模拟输入 (seq_len=3, dim=8)
    x = [
        [0.1] * 8,
        [0.2] * 8,
        [0.3] * 8
    ]
    
    print(f"\n输入形状：seq_len=3, dim=8")
    
    # 前向传播
    output = attn.forward(x)
    
    print(f"\n输出形状：seq_len=3, dim=8")
    print(f"输出[0] (前 4 维): {[f'{v:.4f}' for v in output[0][:4]]}")
    print(f"输出[1] (前 4 维): {[f'{v:.4f}' for v in output[1][:4]]}")
    print(f"输出[2] (前 4 维): {[f'{v:.4f}' for v in output[2][:4]]}")
    

def test_rope_property():
    """测试 RoPE 性质"""
    print("=" * 60)
    print("测试：RoPE 性质")
    print("=" * 60)
    
    rope = RoPE(dim=4, base=10000)
    
    # 测试 1: 模长不变性
    print("\n1. 模长不变性测试:")
    x = [-0.5, 0.3, 0.4, 0.6]
    original_norm = math.sqrt(sum(v ** 2 for v in x))
    
    for pos in [0, 5, 10, 50, 100]:
        rotated = rope.apply_rope(x, pos)
        rotated_norm = math.sqrt(sum(v ** 2 for v in rotated))
        diff = abs(original_norm - rotated_norm)
        print(f"   位置 {pos}: 原模长={original_norm:.6f}, 旋转后={rotated_norm:.6f}, 差值={diff:.8f}")
    
    # 测试 2: 相对位置编码
    print("\n2. 相对位置编码测试:")
    x1 = [0.1, 0.2, 0.3, 0.4]
    x2 = [0.2, 0.3, 0.4, 0.5]
    
    # 位置差为 5
    r1 = rope.apply_rope(x1, 10)
    r2 = rope.apply_rope(x2, 15)
    
    # 计算点积
    dot_product = sum(a * b for a, b in zip(r1, r2))
    print(f"   位置 10 和 15 的点积：{dot_product:.6f}")
    print(f"   相对位置：5")
    

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RoPE 旋转位置编码实现")
    print("=" * 60 + "\n")
    
    # 运行测试
    test_rope_basic()
    test_rope_attention()
    test_rope_property()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
