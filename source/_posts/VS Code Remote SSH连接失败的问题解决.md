---
layout: vscode
title: VS Code Remote SSH连接失败的问题解决
date: 2023-07-06 23:27:10
categories:
- Debugging Diaries
tags:
- IssueFix
---
# VS Code Remote SSH连接失败的问题解决

本文档针对VS Code中的Remote SSH插件在尝试连接远程服务器时出现"Failed to parse remote port from server output"错误的情况提供解决方案。作者在经过一系列的排查和尝试后，最终找到了解决的方法。

## 问题描述

在使用VS Code的Remote SSH插件尝试连接远程服务器时，遇到了错误提示"Failed to parse remote port from server output"。此错误提示可能是由于VS Code不能正确地从SSH服务器的输出中解析出远程端口。

## 尝试的解决方法

1. 检查SSH配置文件
2. 更新VS Code和Remote SSH扩展
3. 手动SSH连接
4. 检查远程服务器的状态
5. 重启VS Code

以上常见的解决方法都未能解决问题。

## 成功的解决方案

最终，作者尝试了取消勾选VS Code设置中的`Remote.SSH: Use Local Server`选项，成功连接到了远程服务器。当该选项被选中（默认）时，VS Code会在本地机器上启动一个服务器，然后通过该本地服务器连接到远程SSH服务器。当取消勾选此选项时，VS Code会直接连接到远程SSH服务器，而不通过本地服务器。

### 解决步骤

1. 打开VS Code。
2. 在左侧的活动栏点击齿轮图标打开设置。
3. 在设置搜索框中输入`Remote.SSH: Use Local Server`。
4. 取消选中出现的`Remote.SSH: Use Local Server`复选框。

## 注意事项

虽然直接连接到远程SSH服务器可以解决某些连接问题，但由于没有利用到本地服务器的优势，可能会导致VS Code的性能稍有下降。但只要没有遇到性能问题，这个设置就不需要过于担心。

## 结论

如果你在使用VS Code的Remote SSH插件连接远程服务器时遇到了类似的问题，你也可以试试这个方法，希望这个解决方案能帮助到你。