---
title: "GitLog"
date: 2023-05-23 00:59:40
categories:
- Tech Toolbox
tags:
- Git
---
# Git Log

当使用Git进行版本控制时，**`git log`**命令是一个有用的工具，它可以显示提交历史记录和分支之间的关系。

### 基本用法

在终端或命令行中使用以下命令格式来调用`git log`：

```bash
git log
```

这将显示包含所有提交历史记录的列表，最新的提交显示在最上面。

### 限制输出

`git log`提供了一些选项来限制输出，以满足不同的需求。

- `-oneline`：以紧凑的一行摘要形式显示提交历史记录。

```bash
git log --oneline
```

- `-decorate`：在输出中显示分支和标签的引用名称。

```
git log --decorate
```

- `-graph`：使用图形表示法展示分支和合并的关系。

```bash
git log --graph
```

可以将这些选项组合在一起使用：

```bash
git log --oneline --decorate --graph
```

当您行 **`git log --oneline --decorate --graph`** 命令时，输出结果会以一种紧凑、图形化的方式显示提交历史记录和分支之间的关系。下面是对输出结果中每个部分的解释：

1. Commit Hash（提交哈希值）：每个提交的唯一标识符，通常使用短的哈希值。这些哈希值是提交的独特标识，可以用来引用和检索特定的提交。
2. Commit Message（提交消息）：提交时输入的描述性消息，用于说明该提交所做的更改和目的。
3. Branches and Tags（分支和标签）：显示当前提交所在的分支和相关标签的引用名称。这些引用名称显示在提交哈希值后的括号内，以及在分支和标签之前的装饰符 **`decorate`**。
4. Graphical Representation（图形表示）：使用字符（如斜线、反斜线、竖线和星号）表示分支和合并的关系。这部分使用图形表示法展示了提交历史记录中不同分支的发展和合并情况。斜线（/）和反斜线（\）表示分支的发展，竖线（|）表示分支的分叉，星号（*）表示合并点。

命令的输出结果可以通过以下方式进行阅读：

1. 每行表示一个提交，包含简短的提交哈希值和提交消息。例如：**`579ac2d Resolved merge conflicts with master branch`**。
2. 分支和标签的引用名称显示在每个提交的后面。它们用括号括起来，并在引用名称前加上 **`tag:`** 或 **`HEAD ->`** 的标识符。例如：**`(HEAD -> huiyu/product_search_similarity_test, origin/huiyu/product_search_similarity_test)`** 表示当前所在的分支和远程分支。
3. 图形表示法展示了分支和合并的关系。合并提交显示为一个或多个分支合并在一起的线条。例如，**`\`** 和 **`/`** 字符表示不同的分支合并。**`|`** 字符表示分支的分叉。这种图形表示法可以帮助您理解提交历史中不同分支之间的关系。

通过阅读这些输出结果，您可以了解每个提交的信息，包括提交哈希值、提交消息、分支和标签的引用名称，以及分支和合并的关系。这有助于您跟踪代码的发展历程、分支的合并情况以及不同分支之间的关系。

### 过滤和排序提交

您可以使用一些选项来过滤和排序提交历史记录。

- `-author=<author>`：仅显示特定作者的提交历史记录。

```bash
git log --author=John
```

- `-since=<date>`：仅显示指定日期之后的提交历史记录。

```bash
git log --since="2023-01-01"
```

- `-until=<date>`：仅显示指定日期之前的提交历史记录。

```bash
git log --until="2023-02-01"
```

- `-grep=<pattern>`：仅显示包含指定模式的提交消息。

```bash
git log --grep="bug fix"
```

- `-follow <file>`：跟踪指定文件的改动历史记录。

```bash
git log --follow file.txt
```

- `-reverse`：按照提交时间的逆序显示提交历史记录。

```bash
git log --reverse
```

### 分支和标签

默认情况下，`git log`显示所有分支的提交历史记录。您还可以指定特定的分支或标签。

```bash
git log <branch-name>
```

```bash
git log <tag-name>
```

### 其他选项

`git log`命令还提供其他一些有用的选项，例如：

- `-stat`：显示每个提交的简要统计信息，包括改动的文件和插入/删除的行数。

```bash
git log --stat
```

- `-pretty=<format>`：使用自定义的输出格式显示提交历史记录。

```bash
git log --pretty=format:"%h - %an, %ar : %s"
```

- `-graph`和`-oneline`可以与其他选项组合使用。