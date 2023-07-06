---
title: "[K8s]-Kubernetes基础"
date: 2023-06-06 14:27:00
categories:
- Tech Toolbox
tags:
- K8s
---

> 前言：本文将介绍Kubernetes基础中的重要概念——Pod，以及它在Kubernetes中的应用和功能。首先，我们将详细解释Pod的基本概念和构成，包括它作为最小可部署单元的特点和包含的资源。然后，我们将探讨Pod在Kubernetes中的作用与功能，包括调度单位、网络单元、存储单元、生命周期管理和水平扩展。通过深入了解Pod，您将对Kubernetes中的核心概念有更全面的理解。


## **I. Kubernetes基础：Pod理解与应用**

### **1.1 Pod的基本概念与构成**

在Kubernetes（简称K8s）中，Pod（容器组）是最小的可部署单元。它是Kubernetes集群中可以运行的一组一个或多个容器的逻辑主机。Pod提供了一个独立的环境，其中包含运行应用程序所需的所有资源，如存储、网络和其他依赖项。

Pod通常由一个或多个紧密相关的容器组成，这些容器共享相同的命名空间、网络和存储卷。它们可以通过本地主机上的localhost进行通信，并且可以共享文件系统的一部分或全部内容。

### **1.2 Pod在Kubernetes中的作用与功能**

Pod在Kubernetes中的作用是以下几个方面：

1. 调度单位：Kubernetes将Pod作为调度的基本单位，决定在哪个节点上运行Pod。
2. 网络单元：每个Pod都有自己的IP地址，并且可以通过Kubernetes集群内部和外部的服务发现机制与其他Pod或外部服务通信。
3. 存储单元：Pod可以共享存储卷，容器之间可以共享文件系统中的数据。
4. 生命周期管理：Pod可以创建、启动、停止和销毁，它们的生命周期由Kubernetes控制器管理。
5. 水平扩展：可以通过复制Pod的方式水平扩展应用程序的实例。

## ****II. Kubernetes应用实践：在Kubernetes中安装和配置Miniconda****

### ****2.1 安装和配置Miniconda的步骤****

本文介绍了如何在Kubernetes集群中安装和配置Miniconda。Miniconda是一个轻量级的Python环境管理工具，可用于创建和管理Python环境及其相关包。

### 步骤 1：登录到Kubernetes Pod的终端

``` bash
kubectl exec -it <pod-name> -- /bin/bash
```

将 `<pod-name>` 替换为要登录的Pod的名称。

### 步骤 2：下载和安装Miniconda

``` bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

这将下载Miniconda的安装脚本并启动安装过程。

### 步骤 3：完成安装向导

根据安装向导的提示，选择安装路径、环境变量配置等选项完成Miniconda的安装。

### 步骤 4：激活Miniconda环境

``` bash
source ~/.bashrc
```

重新加载终端或执行上述命令来激活Miniconda环境。

### 步骤 5：使用Miniconda

在激活的Miniconda环境中，您可以使用以下命令来管理环境和安装Python包：

``` bash
conda create --name myenv python=3.9
conda activate myenv
conda install package_name
```

### ****2.2 安装和使用Miniconda的注意事项****

- 请根据Pod的操作系统和架构调整Miniconda的下载链接和安装命令。
- 为了自动化安装和配置Miniconda，请将相关步骤和环境配置包含在Pod的初始化脚本或容器镜像构建过程中。
- 请在安装和使用Miniconda时遵循适当的最佳实践和安全性措施，并根据具体需求进行配置和管理。

### ****2.3 安装和配置Miniconda的总结****

通过按照本文中的步骤，在Kubernetes中安装和配置Miniconda，您可以轻松管理Python环境和包，并为您的应用程序提供所需的依赖项。Miniconda的灵活性和可扩展性使其成为在Kubernetes环境中开发和部署Python应用程序的理想选择。

## ****III. Kubernetes数据操作：在本地Mac电脑将文件传输到Kubernetes集群的流程****

本文介绍了如何在本地Mac电脑上将文件传输到Kubernetes集群中的Pod。我们使用lrzsz工具来实现文件的上传和下载操作。

### ****3.1 文件传输前的准备工作****

- 本地Mac电脑已经安装了Homebrew。
- Kubernetes集群已经安装了lrzsz工具。

**[在本地执行]**

1. 打开终端应用程序。
2. 安装lrzsz工具。在终端中执行以下命令：
    
    ``` bash
    brew install lrzsz
    ```
    
3. 确保Kubernetes集群中已经安装了lrzsz工具。在Kubernetes集群中的终端中执行以下命令：
    
    ``` bash
    apt-get update
    apt-get install lrzsz
    ```
    

[**在Kubernetes集群中执行]**

1. 登录到Pod的终端。在Kubernetes集群中的终端中执行以下命令：
    
    ``` bash
    kubectl exec -it <pod-name> -- /bin/bash
    ```
    
    将 `<pod-name>` 替换为目标Pod的名称。
    
2. 在Pod的终端中，使用以下命令来接收文件：
    
    ``` bash
    rz
    ```
    
    执行该命令后，将弹出一个文件选择窗口。
    

### ****3.2 文件传输的具体步骤****

**[在本地执行]**

1. 在本地终端中，使用以下命令将文件发送到Kubernetes集群的Pod：
将 `/path/to/environ.yaml` 替换为 `environ.yaml` 文件在本地计算机上的路径。
    
    ``` bash
    sz /path/to/environ.yaml
    ```
    

**[在Kubernetes集群中执行]**

1. 在Kubernetes集群中的终端的文件选择窗口中，选择要上传的文件 `environ.yaml`。

文件传输将在本地和Kubernetes集群之间进行，使用了lrzsz工具的上传和下载命令。请确保在本地和Kubernetes集群中都按照相应的步骤进行安装和操作，并使用正确的命令进行文件传输。

请在进行文件传输操作时，遵循适当的安全性和最佳实践，以保护数据和系统的安全性。

## **IV. 如何检查在终端断开连接后 Linux 命令是否继续执行**

在 Linux 终端中运行命令或脚本时，如果终端连接断开，您可能会想知道命令或脚本是否仍在后台执行。以下是几种方法来检查 Linux 命令在终端断开连接后是否继续执行。

### 4.1 **方法一：使用 ps 命令**

1. 打开新的终端窗口。
2. 运行以下命令：`ps aux | grep <命令或脚本关键词>`
3. 检查输出结果中是否存在与命令或脚本相关的进程。如果存在，表示命令或脚本仍在后台执行。

### 4.2 **方法二：使用 pgrep 命令**

1. 打开新的终端窗口。
2. 运行以下命令：`pgrep -f <命令或脚本关键词>`
3. 检查输出结果中是否存在与命令或脚本相关的进程 ID。如果存在，表示命令或脚本仍在后台执行。

### 4.3 **方法三：使用日志文件或输出文件**

1. 如果在命令或脚本中使用了输出重定向（如 `tee`），请检查日志文件或输出文件。
2. 打开新的终端窗口。
3. 使用 `tail` 命令查看日志文件或输出文件的最后几行：`tail -n <行数> <文件路径>`
4. 检查最后几行是否包含与命令或脚本的输出相关的内容。如果有新的输出，表示命令或脚本仍在执行。

需要注意的是，即使命令或脚本在终端断开连接后仍在后台执行，如果发生错误或问题，它们可能会终止或停止运行。因此，还应检查命令或脚本本身是否存在问题。

总结：
通过使用 ps 命令、pgrep 命令或查看日志文件或输出文件，您可以检查在终端断开连接后 Linux 命令是否继续执行。这些方法提供了一种了解命令或脚本是否在后台持续执行的方式，以确保任务能够正常进行。

## **V.** 使用 Bash 脚本执行 Python 脚本

本文档介绍了如何使用 Bash 脚本来执行指定的 Python 脚本，并提供了一个示例脚本。该脚本还涉及使用 conda 环境来运行 Python。

### 5.1 简介

在某些情况下，您可能需要在终端中执行长时间运行的 Python 脚本。为了确保持久性并方便管理，可以编写一个 Bash 脚本来运行 Python 脚本。本文档提供了一个示例脚本，演示如何使用 Bash 脚本来执行 Python 脚本。

### 5. 2 脚本示例

以下是一个示例 Bash 脚本，用于执行特定的 Python 脚本，并使用 conda 环境：

``` bash
#!/bin/bash

# 定义 conda 环境名称
conda_env="python3.10"

# 激活 conda 环境
source activate $conda_env

# 执行 Python 脚本，并使用 tee 将输出同时重定向到文件和控制台
python data_gen_updated.py conversations_0607_v1_500 500 | tee output.log

# 停用 conda 环境
conda deactivate

```

上述脚本包含以下步骤：

1. 定义 conda 环境的名称（根据需要进行修改）。
2. 使用 `source activate` 命令激活指定的 conda 环境。
3. 使用 `python` 命令执行特定的 Python 脚本。同时，使用 `tee` 命令将输出同时重定向到文件和控制台。
4. 使用 `conda deactivate` 命令停用 conda 环境。

### 5.3 使用脚本

按照以下步骤在终端中使用脚本：

1. 使用文本编辑器创建一个新文件，并将上述示例脚本粘贴进去。
2. 保存文件并关闭文本编辑器。
3. 在终端中，赋予脚本执行权限：
    
    ``` bash
    chmod +x script.sh
    ```
    
4. 运行脚本：
    
    ``` bash
    ./script.sh
    ```
    

确保在运行脚本之前，已正确安装并配置所需的 conda 环境，并将脚本中的 `python data_gen_updated.py conversations_0607_v1_500 500` 替换为您要执行的实际命令。

### 5.4 结论

使用 Bash 脚本可以在终端中执行 Python 脚本，并提供持久性和管理灵活性。本文档提供了一个示例脚本，帮助您开始使用 Bash 脚本来执行 Python 脚本，并演示了使用 conda 环境的方法。根据您的实际需求，可以修改和调整脚