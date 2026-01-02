"""
Sum Tree数据结构

============================================================
核心思想 (Core Idea)
============================================================
SumTree是一种特殊的二叉树数据结构，用于高效实现优先经验回放(PER)。
其核心思想是：

1. **叶节点**: 存储每个样本的优先级值
2. **内部节点**: 存储子节点优先级之和
3. **根节点**: 存储所有优先级的总和

============================================================
数学基础 (Mathematical Foundation)
============================================================
对于N个叶节点的树结构：

- 总节点数: 2N - 1
- 叶节点索引: [N-1, 2N-2]
- 节点i的父节点: (i - 1) // 2
- 节点i的子节点: 2i + 1 (左), 2i + 2 (右)

比例采样使用累积和：

.. math::
    \\text{sample}(u) \\to i \\text{ 使得 } \\sum_{j<i} p_j < u \\leq \\sum_{j \\leq i} p_j

其中 u ~ Uniform(0, total_priority)。

============================================================
复杂度分析 (Complexity Analysis)
============================================================
+------------------+------------+----------------------------------+
| 操作             | 复杂度     | 说明                             |
+==================+============+==================================+
| add()            | O(log N)   | 插入 + 更新祖先节点              |
+------------------+------------+----------------------------------+
| update_priority()| O(log N)   | 更新叶节点 + 传播                |
+------------------+------------+----------------------------------+
| get()            | O(log N)   | 按累积和二分搜索                 |
+------------------+------------+----------------------------------+
| total_priority   | O(1)       | 根节点值                         |
+------------------+------------+----------------------------------+
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


class SumTree:
    """
    用于O(log N)优先采样的二叉求和树
    
    Attributes
    ----------
    total_priority : float
        所有优先级之和（根节点值）
    capacity : int
        最大元素数量
    """
    
    __slots__ = ("_capacity", "_tree", "_data", "_write_idx", "_size")
    
    def __init__(self, capacity: int) -> None:
        """
        初始化Sum Tree
        
        Parameters
        ----------
        capacity : int
            最大叶节点数量（数据元素数量）
        
        Raises
        ------
        ValueError
            如果capacity不是正整数
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity必须是正整数，得到{capacity!r}")
        
        self._capacity = capacity
        # 树数组: 内部节点 [0, capacity-2], 叶节点 [capacity-1, 2*capacity-2]
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Any]] = [None] * capacity
        self._write_idx = 0
        self._size = 0
    
    @property
    def total_priority(self) -> float:
        """总优先级和（根节点值）。O(1)"""
        return float(self._tree[0])
    
    @property
    def capacity(self) -> int:
        """最大元素数量"""
        return self._capacity
    
    def __len__(self) -> int:
        """当前存储的元素数量"""
        return self._size
    
    def add(self, priority: float, data: Any) -> None:
        """
        添加具有指定优先级的元素
        
        Parameters
        ----------
        priority : float
            优先级值（必须非负）
        data : Any
            要存储的数据（通常是Transition）
        
        Notes
        -----
        - 由于优先级传播，时间复杂度为O(log N)
        - 当达到容量时覆写最旧的元素（FIFO）
        """
        if priority < 0:
            raise ValueError(f"priority必须非负，得到{priority}")
        
        tree_idx = self._write_idx + self._capacity - 1
        self._data[self._write_idx] = data
        self._update(tree_idx, priority)
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def _update(self, tree_idx: int, priority: float) -> None:
        """更新tree_idx处的优先级并将差值传播到根节点"""
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # 向上传播变化到根节点
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta
    
    def update_priority(self, tree_idx: int, priority: float) -> None:
        """
        更新现有元素的优先级
        
        Parameters
        ----------
        tree_idx : int
            叶节点的树索引（由get()返回）
        priority : float
            新的优先级值
        """
        if priority < 0:
            raise ValueError(f"priority必须非负，得到{priority}")
        self._update(tree_idx, priority)
    
    def get(self, cumsum: float) -> Tuple[int, float, Any]:
        """
        通过累积和采样元素（比例采样）
        
        Parameters
        ----------
        cumsum : float
            目标累积和，范围 [0, total_priority)
        
        Returns
        -------
        tree_idx : int
            树数组中的索引（用于优先级更新）
        priority : float
            采样元素的优先级
        data : Any
            存储的数据元素
        
        Notes
        -----
        从根到叶的O(log N)二分搜索。
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self._tree):
                # 到达叶节点
                break
            if cumsum <= self._tree[left]:
                parent = left
            else:
                cumsum -= self._tree[left]
                parent = right
        
        data_idx = parent - self._capacity + 1
        return parent, float(self._tree[parent]), self._data[data_idx]
    
    def min_priority(self) -> float:
        """
        获取存储元素中的最小非零优先级
        
        Returns
        -------
        float
            最小优先级，如果为空则返回0.0
        """
        if self._size == 0:
            return 0.0
        start = self._capacity - 1
        priorities = self._tree[start:start + self._size]
        positive_priorities = priorities[priorities > 0]
        if len(positive_priorities) == 0:
            return 0.0
        return float(np.min(positive_priorities))
