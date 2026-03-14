#!/usr/bin/env python3
"""
基础位置编码实现

"""

import math
from typing import List


# ==================== 1. 可学习位置编码 ====================

class LearnablePositionEmbedding:
    """
    可学习位置编码
    
    为每个位置学习一个嵌入向量
    """
    
    def __init__(self, max_len: int, dim: int):
        """
        初始化
        
        参数:
            max_len: 最大序列长度
            dim: 嵌入维度
        """
        self.max_len = max_len
        self.dim = dim
        # 模拟学习后的位置嵌入（实际中通过训练学习）
        self.embeddings = self._init_embeddings()
    
    def _init_embeddings(self) -> List[List[float]]:
        """初始化位置嵌入（模拟学习后的结果）"""
        embeddings = []
        for pos in range(self.max_len):
            emb = []
            for i in range(self.dim):
                # 使用随机值模拟学习后的嵌入
                emb.append((pos + i) % 100 / 100.0)
            embeddings.append(emb)
        return embeddings
    
    def forward(self, x: List[List[float]], positions: List[int] = None) -> List[List[float]]:
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, dim]
            positions: 位置列表（可选，默认为 [0, 1, 2, ...]）
        
        返回:
            添加位置编码后的序列
        """
        batch_size = len(x)
        seq_len = len(x[0]) if x else 0
        
        if positions is None:
            positions = list(range(seq_len))
        
        # 添加位置编码
        output = []
        for batch_idx in range(batch_size):
            batch_output = []
            for seq_idx in range(seq_len):
                pos = positions[seq_idx] if seq_idx < len(positions) else seq_idx
                pos_emb = self.embeddings[pos % self.max_len]
                
                # 输入 + 位置编码
                token_emb = x[batch_idx][seq_idx]
                combined = [t + p for t, p in zip(token_emb, pos_emb)]
                batch_output.append(combined)
            output.append(batch_output)
        
        return output
    
    def __repr__(self):
        return f"LearnablePositionEmbedding(max_len={self.max_len}, dim={self.dim})"


# ==================== 2. 正弦位置编码 ====================

class SinusoidalPositionEmbedding:
    """
    正弦位置编码 (Transformer 原始实现)
    
    使用不同频率的正弦和余弦函数
    """
    
    def __init__(self, max_len: int = 5000, dim: int = 512):
        """
        初始化
        
        参数:
            max_len: 最大序列长度
            dim: 嵌入维度
        """
        self.max_len = max_len
        self.dim = dim
        self.pe = self._create_pe_matrix()
    
    def _create_pe_matrix(self) -> List[List[float]]:
        """创建位置编码矩阵"""
        pe = [[0.0] * self.dim for _ in range(self.max_len)]
        
        for pos in range(self.max_len):
            for i in range(0, self.dim, 2):
                # 偶数维度使用 sin
                pe[pos][i] = math.sin(pos / math.pow(10000, (2 * (i // 2)) / self.dim))
                # 奇数维度使用 cos
                if i + 1 < self.dim:
                    pe[pos][i + 1] = math.cos(pos / math.pow(10000, (2 * (i // 2)) / self.dim))
        
        return pe
    
    def forward(self, x: List[List[float]], positions: List[int] = None) -> List[List[float]]:
        """
        前向传播
        
        参数:
            x: 输入序列 [seq_len, dim]
            positions: 位置列表（可选）
        
        返回:
            添加位置编码后的序列
        """
        seq_len = len(x)
        
        if positions is None:
            positions = list(range(seq_len))
        
        output = []
        for seq_idx in range(seq_len):
            pos = positions[seq_idx] if seq_idx < len(positions) else seq_idx
            pos_emb = self.pe[pos % self.max_len]
            
            # 输入 + 位置编码
            token_emb = x[seq_idx]
            combined = [t + p for t, p in zip(token_emb, pos_emb)]
            output.append(combined)
        
        return output
    
    def get_pe_matrix(self, start: int = 0, end: int = 10) -> List[List[float]]:
        """
        获取位置编码矩阵的一部分（用于可视化）
        
        参数:
            start: 起始位置
            end: 结束位置
        
        返回:
            位置编码矩阵
        """
        return self.pe[start:min(end, self.max_len)]
    
    def __repr__(self):
        return f"SinusoidalPositionEmbedding(max_len={self.max_len}, dim={self.dim})"


# ==================== 3. 测试函数 ====================

def test_learnable_pe():
    """测试可学习位置编码"""
    print("=" * 60)
    print("测试：可学习位置编码")
    print("=" * 60)
    
    # 创建位置编码
    pe = LearnablePositionEmbedding(max_len=10, dim=4)
    print(f"\n位置编码层：{pe}")
    
    # 模拟输入 (batch_size=1, seq_len=3, dim=4)
    x = [
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 1.0, 1.1, 1.2]]
    ]
    
    print(f"\n输入形状：batch_size=1, seq_len=3, dim=4")
    print(f"输入[0]: {x[0]}")
    
    # 前向传播
    output = pe.forward(x)
    
    print(f"\n输出[0]: {output[0]}")


def test_sinusoidal_pe():
    """测试正弦位置编码"""
    print("=" * 60)
    print("测试：正弦位置编码")
    print("=" * 60)
    
    # 创建位置编码
    pe = SinusoidalPositionEmbedding(max_len=10, dim=8)
    print(f"\n位置编码层：{pe}")
    
    # 获取位置编码矩阵
    pe_matrix = pe.get_pe_matrix(0, 5)
    
    print(f"\n位置编码矩阵 (前 5 个位置，dim=8):")
    for pos, row in enumerate(pe_matrix):
        print(f"  Pos {pos}: {[f'{v:.4f}' for v in row]}")
    
    # 模拟输入 (seq_len=3, dim=8)
    x = [
        [0.1] * 8,
        [0.2] * 8,
        [0.3] * 8
    ]
    
    print(f"\n输入形状：seq_len=3, dim=8")
    
    # 前向传播
    output = pe.forward(x)
    
    print(f"\n输出[0] (前 4 维): {[f'{v:.4f}' for v in output[0][:4]]}")
    print(f"输出[1] (前 4 维): {[f'{v:.4f}' for v in output[1][:4]]}")
    print(f"输出[2] (前 4 维): {[f'{v:.4f}' for v in output[2][:4]]}")
    


def visualize_sinusoidal_pe():
    """可视化正弦位置编码"""
    print("=" * 60)
    print("可视化：正弦位置编码")
    print("=" * 60)
    
    pe = SinusoidalPositionEmbedding(max_len=20, dim=16)
    
    print(f"\n位置编码矩阵 (前 10 个位置，前 8 维):")
    print("-" * 60)
    
    pe_matrix = pe.get_pe_matrix(0, 10)
    
    # 打印表头
    print(f"{'Pos':>5}", end="")
    for i in range(8):
        print(f"{f'd[{i}]':>8}", end="")
    print()
    print("-" * 60)
    
    # 打印数据
    for pos, row in enumerate(pe_matrix):
        print(f"{pos:>5}", end="")
        for val in row[:8]:
            print(f"{val:>8.4f}", end="")
        print()
    
    print("-" * 60)
    print("\n观察:")
    print("1. 偶数维度使用 sin 函数")
    print("2. 奇数维度使用 cos 函数")
    print("3. 不同维度有不同频率")
    print("4. 位置信息被编码到不同频率中\n")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("位置编码基础实现")
    print("=" * 60 + "\n")
    
    # 运行测试
    test_learnable_pe()
    test_sinusoidal_pe()
    visualize_sinusoidal_pe()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
