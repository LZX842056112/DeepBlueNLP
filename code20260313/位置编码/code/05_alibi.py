#!/usr/bin/env python3
"""
ALiBi 位置编码实现

**参考论文**: Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
"""

import math
from typing import List


# ==================== ALiBi 实现 ====================

class ALiBi:
    """
    Attention with Linear Biases (ALiBi)
    
    不使用位置嵌入，在 Attention 分数中添加与距离成比例的偏置
    """

    def __init__(self, n_heads: int):
        """
        初始化
        
        参数:
            n_heads: 注意力头数
        """
        self.n_heads = n_heads
        self.slopes = self._compute_slopes()

    def _compute_slopes(self) -> List[float]:
        """
        计算每个注意力头的斜率
        
        公式：m_j = 1 / 2^(ceil(j * log2(n_heads) / n_heads))
        """
        slopes = []
        log2_n_heads = math.log2(self.n_heads)

        for j in range(self.n_heads):
            # 计算斜率
            power = math.ceil(j * log2_n_heads / self.n_heads)
            slope = 1.0 / (2 ** power)
            slopes.append(slope)

        return slopes

    def compute_bias(self, seq_len: int) -> List[List[float]]:
        """
        计算 Attention 偏置矩阵
        
        参数:
            seq_len: 序列长度
        
        返回:
            偏置矩阵 [seq_len, seq_len]
        """
        bias_matrix = []

        for i in range(seq_len):  # query 位置
            row = []
            for j in range(seq_len):  # key 位置
                # 相对位置距离（因果掩码：只能看到过去）
                if j <= i:
                    distance = i - j
                else:
                    distance = -1  # 未来位置（会被掩码）
                row.append(distance)
            bias_matrix.append(row)

        return bias_matrix

    def apply_alibi(self, attention_scores: List[List[float]],
                    head_idx: int) -> List[List[float]]:
        """
        对 Attention 分数应用 ALiBi 偏置
        
        参数:
            attention_scores: Attention 分数 [seq_len, seq_len]
            head_idx: 注意力头索引
        
        返回:
            添加偏置后的分数
        """
        seq_len = len(attention_scores)
        slope = self.slopes[head_idx]

        # 计算偏置矩阵
        bias_matrix = self.compute_bias(seq_len)

        # 应用偏置
        output = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if bias_matrix[i][j] >= 0:
                    # 过去位置：添加负偏置
                    biased_score = attention_scores[i] + slope * (-bias_matrix[i][j])
                else:
                    # 未来位置：设置为负无穷（掩码）
                    biased_score = float('-inf')
                row.append(biased_score)
            output.append(row)

        return output

    def get_slopes_info(self) -> List[dict]:
        """获取斜率信息"""
        info = []
        for i, slope in enumerate(self.slopes):
            info.append({
                'head': i,
                'slope': slope,
                '2^k': 1.0 / slope
            })
        return info

    def __repr__(self):
        return f"ALiBi(n_heads={self.n_heads})"


# ==================== ALiBi Attention ====================

class ALiBiAttention:
    """
    使用 ALiBi 的 Self-Attention
    """

    def __init__(self, dim: int, n_heads: int = 8):
        """
        初始化
        
        参数:
            dim: 总维度
            n_heads: 注意力头数
        """
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.alibi = ALiBi(n_heads)

        # 模拟权重矩阵
        self.W_q = self._init_weight(dim, dim)
        self.W_k = self._init_weight(dim, dim)
        self.W_v = self._init_weight(dim, dim)
        self.W_o = self._init_weight(dim, dim)

    def _init_weight(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """初始化权重矩阵"""
        import random
        random.seed(42)
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        return [[random.gauss(0, scale) for _ in range(out_dim)] for _ in range(in_dim)]

    def _matmul(self, x: List[float], w: List[List[float]]) -> List[float]:
        """矩阵乘法"""
        out_dim = len(w[0])
        return [sum(x[i] * w[i][j] for i in range(len(x))) for j in range(out_dim)]

    def _split_heads(self, x: List[List[float]]) -> List[List[List[float]]]:
        """分割注意力头"""
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
        """拼接注意力头"""
        seq_len = len(heads[0])
        output = []
        for seq_idx in range(seq_len):
            token_vec = []
            for h in range(self.n_heads):
                token_vec.extend(heads[h][seq_idx])
            output.append(token_vec)
        return output

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax（处理负无穷）"""
        # 过滤负无穷
        max_val = max(v for v in x if v != float('-inf'))
        exp_x = []
        for val in x:
            if val == float('-inf'):
                exp_x.append(0.0)
            else:
                exp_x.append(math.exp(val - max_val))

        sum_exp = sum(exp_x)
        if sum_exp == 0:
            return [0.0] * len(x)
        return [e / sum_exp for e in exp_x]

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """
        前向传播
        
        参数:
            x: 输入 [seq_len, dim]
        
        返回:
            输出 [seq_len, dim]
        """
        seq_len = len(x)

        # 1. 线性投影
        q = [self._matmul(x[i], self.W_q) for i in range(seq_len)]
        k = [self._matmul(x[i], self.W_k) for i in range(seq_len)]
        v = [self._matmul(x[i], self.W_v) for i in range(seq_len)]

        # 2. 分割注意力头
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        # 3. Self-Attention with ALiBi
        attn_outputs = []
        for h in range(self.n_heads):
            head_dim = self.head_dim
            scale = math.sqrt(head_dim)

            head_output = []
            for i in range(seq_len):
                # 计算注意力分数
                scores = []
                for j in range(seq_len):
                    score = sum(q_heads[h][i][d] * k_heads[h][j][d] for d in range(head_dim))
                    scores.append(score / scale)

                # 应用 ALiBi 偏置
                biased_scores = self.alibi.apply_alibi(scores, h)

                # Softmax
                attn_weights = self._softmax(biased_scores)

                # 加权求和
                output_vec = [0.0] * head_dim
                for j in range(seq_len):
                    if attn_weights[j] > 0:
                        for d in range(head_dim):
                            output_vec[d] += attn_weights[j] * v_heads[h][j][d]

                head_output.append(output_vec)

            attn_outputs.append(head_output)

        # 4. 拼接注意力头
        concat_output = self._concat_heads(attn_outputs)

        # 5. 输出投影
        output = [self._matmul(concat_output[i], self.W_o) for i in range(seq_len)]

        return output

    def __repr__(self):
        return f"ALiBiAttention(dim={self.dim}, n_heads={self.n_heads})"


# ==================== 测试函数 ====================

def test_alibi_basic():
    """测试 ALiBi 基础功能"""
    print("=" * 60)
    print("测试：ALiBi 基础功能")
    print("=" * 60)

    # 创建 ALiBi
    alibi = ALiBi(n_heads=8)
    print(f"\nALiBi: {alibi}")

    # 斜率信息
    print(f"\n各注意力头的斜率:")
    for info in alibi.get_slopes_info():
        print(f"  Head {info['head']}: slope={info['slope']:.6f} (2^k={info['2^k']:.1f})")

    # 偏置矩阵
    print(f"\n偏置矩阵 (seq_len=5):")
    bias_matrix = alibi.compute_bias(5)

    print("     ", end="")
    for j in range(5):
        print(f"K{j:>5}", end="")
    print()

    for i in range(5):
        print(f"Q{i}:  ", end="")
        for j in range(5):
            print(f"{bias_matrix[i][j]:>5}", end="")
        print()

    print("\n观察:")
    print("1. 对角线为 0（当前位置）")
    print("2. 左侧为正数（过去位置，距离）")
    print("3. 右侧为 -1（未来位置，掩码）")


def test_alibi_attention():
    """测试 ALiBi Attention"""
    print("=" * 60)
    print("测试：ALiBi Attention")
    print("=" * 60)

    # 创建 ALiBi Attention
    attn = ALiBiAttention(dim=8, n_heads=2)
    print(f"\nAttention: {attn}")

    # 模拟输入
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


def test_alibi_extrapolation():
    """测试 ALiBi 外推能力"""
    print("=" * 60)
    print("测试：ALiBi 外推能力")
    print("=" * 60)

    alibi = ALiBi(n_heads=8)

    # 不同序列长度的偏置
    for seq_len in [5, 10, 20, 50]:
        bias_matrix = alibi.compute_bias(seq_len)
        # 最后一个 query 对第一个 key 的偏置
        max_bias = bias_matrix[seq_len - 1][0]
        print(f"seq_len={seq_len:2d}: 最大偏置 = {-alibi.slopes[0] * max_bias:.4f} (head 0)")

    print("\n观察:")
    print("1. ALiBi 可以处理任意长度的序列")
    print("2. 偏置与距离成线性关系")
    print("3. 不需要重新训练即可外推")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("ALiBi 位置编码实现")
    print("=" * 60 + "\n")

    # 运行测试
    test_alibi_basic()
    test_alibi_attention()
    test_alibi_extrapolation()

    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
