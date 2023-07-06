---
title: "[Python]-Python -m 参数解释"
date: 2023-02-01 10:02:03
categories:
- Code Chronicles
tags:
- Python Basic
---
# Python -m 参数解释

## 1. Python -m 参数解释

### 在shell调用时，python -m的参数是什么意思？

当在命令行中输入 "python -m <module-name>" 时，这表示使用Python的内置模块运行程序，其中<module-name>是要运行的模块的名称。这类似于在Python脚本中使用 "import <module-name>" 并运行 "module-name.run()"。

例如： "python -m http.server" 将启动Python内置的HTTP服务器，可以在浏览器中访问当前目录中的文件。 "python -m unittest discover" 将在当前目录中查找并运行所有以test_*.py命名的单元测试脚本。

### module-name.run() 需要自己在模块中编写run函数吗？

不需要。在调用 "python -m <module-name>" 时，Python会自动在模块中寻找并运行名为 "run()" 的函数。如果该函数不存在，则会在模块中寻找并运行名为 "main()" 的函数。

在大多数情况下，如果模块是一个可执行程序，则应该在其中编写一个名为 "main()" 的函数，并在该函数中编写程序逻辑。

如果模块是一个库，而不是可执行程序，则不需要编写 "main()" 或 "run()" 函数。这些函数不会被调用，因为模块被导入时，其中的函数和变量都可以在其他代码中调用。