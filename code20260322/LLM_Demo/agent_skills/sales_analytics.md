---
name: sales_analytics
description: 用于销售数据分析的数据库结构和业务逻辑，包括客户、订单和收入。
---

# 销售分析数据结构

## 表结构

### customers（客户表）
- customer_id（主键）
- name（姓名）
- email（邮箱）
- signup_date（注册日期）
- status（状态：active/active 表示活跃，inactive 表示非活跃）
- customer_tier（客户等级：bronze/银牌/silver/金牌/gold/白金/platinum）

### orders（订单表）
- order_id（主键）
- customer_id（外键 → customers）
- order_date（下单日期）
- status（订单状态：pending/待处理、completed/已完成、cancelled/已取消、refunded/已退款）
- total_amount（订单总金额）
- sales_region（销售区域：north/北区、south/南区、east/东区、west/西区）

### order_items（订单明细表）
- item_id（主键）
- order_id（外键 → orders）
- product_id（产品ID）
- quantity（数量）
- unit_price（单价）
- discount_percent（折扣百分比）

## 业务逻辑

**活跃客户**：status = 'active' 且 signup_date <= 当前日期 - 90天

**收入计算**：仅统计 status = 'completed' 的订单。orders 表中的 total_amount 字段已包含折扣后的金额。

**客户生命周期价值（CLV）**：某客户所有已完成订单金额的总和。

**高价值订单**：total_amount > 1000 的订单

## 示例查询

-- 获取最近一个季度收入排名前10的客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;