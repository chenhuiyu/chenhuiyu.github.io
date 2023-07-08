---
title: "Intersection & Union"
date: 2023-02-20 12:02:03
categories:
- Code Chronicles
tags:
- Python Basic
---
# Intersection & Union

交集和并集

# Intersection

Python中的`intersection()`方法是用于获取两个集合的交集。这个方法是set对象的方法，因此它只能用于set对象中。

如果您想在其他类型的容器中获取交集，可以使用Python的内置函数set()将它们转换为set对象。例如，您可以使用以下代码来获取两个列表的交集：

```python
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
intersection = set(a).intersection(set(b))
```

这里，我们首先使用set()将两个列表a和b转换为set对象，然后使用intersection()方法获取它们的交集。注意，由于集合是无序的，因此结果集合中的元素顺序可能与原始列表中的顺序不同。

除了set对象之外，Python的字典类型也具有类似的intersection()方法，用于获取两个字典的相同键的交集，但这与上述情况不同。

# Union

如果想获取两个容器的并集，可以使用Python的内置函数`set()`将它们转换为set对象，然后使用`union()`方法获取它们的并集。例如，以下代码可以获取两个列表的并集：

```python
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
union = set(a).union(set(b))
```

这里，我们首先使用`set()`将两个列表`a`和`b`转换为`set`对象，然后使用`union()`方法获取它们的并集。注意，由于集合是无序的，因此结果集合中的元素顺序可能与原始列表中的顺序不同。

除了set对象之外，Python的字典类型不支持并集操作。如果您需要对字典进行并集操作，可以将其键或值转换为set对象，然后进行集合操作。例如，以下代码可以获取两个字典的键的并集：

```python
a = {'x': 1, 'y': 2, 'z': 3}
b = {'z': 4, 'w': 5}
union = set(a.keys()).union(set(b.keys()))
```

这里，我们将两个字典的键转换为set对象，然后使用union()方法获取它们的并集。