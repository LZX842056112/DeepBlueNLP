# -*- coding: utf-8 -*-
"""
BP神经网络结构可视化
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_network():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 定义各层神经元位置
    input_layer = [(1, 3), (1, 2)]  # 输入层 (x1, x2)
    hidden_layer = [(2, 4), (2, 3), (2, 2)]  # 隐藏层3个神经元
    output_layer = [(3, 3), (3, 2)]  # 输出层2个神经元
    
    # 绘制神经元
    # 输入层
    for i, (x, y) in enumerate(input_layer):
        circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black')
        ax.add_patch(circle)
        ax.text(x, y, f'x{i+1}\n({5 if i==0 else 10})', ha='center', va='center', fontsize=10)
    
    # 隐藏层
    for i, (x, y) in enumerate(hidden_layer):
        circle = plt.Circle((x, y), 0.3, color='lightgreen', ec='black')
        ax.add_patch(circle)
        ax.text(x, y, f'h{i+1}', ha='center', va='center', fontsize=10)
    
    # 输出层
    for i, (x, y) in enumerate(output_layer):
        circle = plt.Circle((x, y), 0.3, color='lightcoral', ec='black')
        ax.add_patch(circle)
        ax.text(x, y, f'o{i+1}\n(target:{0.01 if i==0 else 0.99})', ha='center', va='center', fontsize=10)
    
    # 绘制连接线和权重标记
    weights = [
        'w1', 'w2',   # x1->h1, x2->h1
        'w3', 'w4',   # x1->h2, x2->h2  
        'w5', 'w6',   # x1->h3, x2->h3
        'w7', 'w8',   # h1->o1, h1->o2
        'w9', 'w10',  # h2->o1, h2->o2
        'w11', 'w12'  # h3->o1, h3->o2
    ]
    
    weight_idx = 0
    
    # 输入层到隐藏层的连接
    for i, (x1, y1) in enumerate(input_layer):
        for j, (x2, y2) in enumerate(hidden_layer):
            ax.plot([x1+0.3, x2-0.3], [y1, y2], 'k-', alpha=0.7)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, weights[weight_idx], fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            weight_idx += 1
    
    # 隐藏层到输出层的连接
    for i, (x1, y1) in enumerate(hidden_layer):
        for j, (x2, y2) in enumerate(output_layer):
            ax.plot([x1+0.3, x2-0.3], [y1, y2], 'k-', alpha=0.7)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, weights[weight_idx], fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            weight_idx += 1
    
    # 添加偏置项
    ax.text(0.5, 4, 'b1=0.35', fontsize=10, ha='center')
    ax.text(2.5, 5, 'b2=0.65', fontsize=10, ha='center')
    
    # 添加箭头指向偏置
    ax.annotate('', xy=(1, 4), xytext=(0.5, 4), 
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('', xy=(2, 5), xytext=(2.5, 5),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlim(0, 4)
    ax.set_ylim(1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('BP神经网络结构图\n2输入→3隐藏→2输出', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('bp_network_structure.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    visualize_network()