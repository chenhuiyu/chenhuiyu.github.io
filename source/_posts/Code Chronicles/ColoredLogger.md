---
title: ColoredLogger-彩色打印日志到控制台并记录到文件
date: 2023-07-10 15:02:44
categories:
- Code Chronicles
tags:
- Python
---
# 彩色打印日志到控制台并记录到文件

本文档介绍了一个名为 ColoredLogger 的日志记录器类，它可以根据不同的消息类型以不同的颜色打印日志，并将日志记录到文件中。该类使用了 colorama 库来实现在控制台中显示带颜色的文本。为了使控制台输出的日志更加易于阅读和理解，我们通常会使用彩色的输出。同时，将日志记录到文件中可以方便我们后续的调试和分析。在Python中，我们可以使用`logging`和`colorama`库来实现这样的功能。
{% asset_img image.png %}
以下是一个如何使用这两个库的详细介绍。


## 功能

- 可以根据不同的消息类型以不同的颜色打印日志消息。
- 将日志消息记录到文件中，使用标准的 `logging` 模块进行记录。
- 在控制台中显示带颜色的日志消息。

## 原理
`logging`库提供了强大的日志记录功能，允许我们将日志记录到控制台、文件或者其他输出设备，并提供了详细的配置选项。

`colorama`库可以使我们在控制台输出彩色的文本。它提供了对ANSI颜色编码的支持，可以在几乎所有的平台和终端中使用。

我们先初始化`colorama`，然后定义了一个`ColoredLogger`类，它包含了各种彩色的输出样式和对应的日志级别。然后，我们设置了`logging`的配置，定义了日志的输出级别、输出文件和文件模式。最后，我们使用`ColoredLogger`的`log`方法来记录日志，它会同时将彩色的文本输出到控制台，并去除颜色控制字符后写入文件。
`ColoredLogger` 类使用了 `colorama` 库来设置控制台中的颜色输出。它通过定义不同类型消息的颜色，并使用 `Fore` 和 `Style` 类来应用相应的颜色。

`ColoredLogger` 类的 `log` 方法接受两个参数：`msg_type` 和 `msg`。根据 `msg_type` 参数的值，方法将选择适当的颜色，并将带有颜色的消息打印到控制台上。然后，它使用正则表达式去除颜色控制字符，并使用 `logging.info` 方法将不带颜色的消息记录到文件中。

## 使用方法

1. 安装 `colorama` 库，使用以下命令：
   ```
   pip install colorama
   ```

2. 导入所需的模块和类：
   ```python
   import re
   import logging
   from colorama import init, Fore, Style
   ```

3. 初始化 `colorama`：
   ```python
   init(autoreset=True)
   ```

4. 定义 `ColoredLogger` 类，并设置不同类型消息的颜色。例如：
   ```python
   class ColoredLogger:
       def __init__(self):
            self.type_A = Fore.CYAN + Style.BRIGHT
            self.type_B = Fore.GREEN + Style.BRIGHT
            self.type_C = Fore.YELLOW + Style.BRIGHT
            self.type_D = Fore.MAGENTA + Style.BRIGHT
            self.type_E = Fore.BLUE + Style.BRIGHT
            self.RESET = Style.RESET_ALL
   ```

5. 实例化 `ColoredLogger` 类，并使用 `log` 方法打印和记录日志消息。例如：
   ```python
   logger = ColoredLogger()
   logger.log('type_A', 'This is a test message for type A.')
   logger.log('type_B', 'This is a test message for type B.')
   # ...
   ```

6. 通过以下命令配置日志记录器，将日志消息写入文件：
   ```python
   logging.basicConfig(level=logging.INFO, filename='example.log', filemode='w')
   ```

7. 运行代码，查看在控制台和文件中显示的带颜色的日志消息。

  
## 完整代码实现
```python
import re
import logging
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class ColoredLogger:

    def __init__(self):
        self.type_A = Fore.CYAN + Style.BRIGHT
        self.type_B = Fore.GREEN + Style.BRIGHT
        self.type_C = Fore.YELLOW + Style.BRIGHT
        self.type_D = Fore.MAGENTA + Style.BRIGHT
        self.type_E = Fore.BLUE + Style.BRIGHT
        self.RESET = Style.RESET_ALL

    def log(self, msg_type, msg):
        if msg_type == "type_A":
            self.print_and_log(self.type_A + f"type_A: {msg}" + self.RESET)
        elif msg_type == "type_B":
            self.print_and_log(self.type_B + f"type_B: {msg}" + self.RESET)
        elif msg_type == "type_C":
            self.print_and_log(self.type_C + f"type_C: {msg}" + self.RESET)
        elif msg_type == "type_D":
            self.print_and_log(self.type_D + f"type_D: {msg}" + self.RESET)
        elif msg_type == "type_E":
            self.print_and_log(self.type_E + f"type_E: {msg}" + self.RESET)

    def print_and_log(self, msg):
        # Print to console with color
        print(msg)

        # Remove color control chars for logging to file
        msg_without_color = re.sub('\x1b\[[0-9;]*m', '', msg)
        logging.info(msg_without_color)

# Setup logging configuration
logging.basicConfig(level=logging.INFO, filename='example.log', filemode='w')

# Use the ColoredLogger
logger = ColoredLogger()
logger.log('type_A', 'This is a test message for type A.')
logger.log('type_B', 'This is a test message for type B.')
logger.log('type_C', 'This is a test message for type C.')
logger.log('type_D', 'This is a test message for type D.')
logger.log('type_E', 'This is a test message for type E.')
```![Alt text](image.png)

## 注意事项
1. `colorama`库的颜色和样式可能在不同的平台和终端上有不同的效果，所以需要在目标环境上进行测试。
2. 在写入文件时，需要去除颜色控制字符，否则会在文本中留下一些无法识别的字符。
3. 在设置`logging`的配置时，需要注意文件模式的设置。'w'模式会在每次运行时覆盖之前的日志，而'a'模式则会在之前的日志后追加新的日志。
4. 使用`logging`库记录日志时，需要注意日志的级别。不同级别的日志会有不同的输出效果和记录方式。
5. 需要注意线程安全。如果在多线程环境下使用`logging`库，可能需要额外的配置来保证线程安全。