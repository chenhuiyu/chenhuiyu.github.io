---
title: "Python yield"
date: 2023-02-01 10:02:03
categories:
- Code Chronicles
tags:
- Python Basic
---
# Python yield

在 Python 中，**`yield`** 是一个关键字，它通常用于生成器函数中，用于生成序列化的值而不需要将整个序列保存在内存中。

当函数被调用并包含 **`yield`** 语句时，它并不会立即执行函数体的所有代码。相反，它将返回一个生成器对象，每次调用生成器对象的 **`__next__()`** 方法时，函数体将从上次 **`yield`** 语句停止的位置继续执行，直到遇到下一个 **`yield`** 语句或函数结束。

举个例子，下面的代码展示了一个简单的生成器函数，它使用 **`yield`** 语句产生数字序列：

```python
def number_generator(n):
    for i in range(n):
        yield i

# 使用生成器对象打印数字序列
my_generator = number_generator(5)
print(next(my_generator))  # 0
print(next(my_generator))  # 1
print(next(my_generator))  # 2
print(next(my_generator))  # 3
print(next(my_generator))  # 4
```

这里，**`number_generator()`** 函数使用 **`yield`** 语句产生数字序列。当函数被调用时，它将返回一个生成器对象 **`my_generator`**。每次调用生成器对象的 **`__next__()`** 方法时，函数体将从上次 **`yield`** 语句停止的位置继续执行，直到函数结束或者再次遇到 **`yield`** 语句。由于生成器只在需要时才产生值，因此可以减少内存的占用。