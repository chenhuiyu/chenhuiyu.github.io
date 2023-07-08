---
title: "Hexo博客的创建与部署"
date: 2023-01-11 00:00:00
categories:
- Tech Toolbox
tags:
- Blog
---
（生活反思）
（代码编年史）
（流浪癖笔记）
NLP Insights（自然语言处理洞察）
Tech Toolbox（技术工具箱）
Travel Tales（旅行故事）
Debugging Diaries
> 前言：大家好，我是博主黑头呆鱼。之前我的旧电脑退休了，这导致我之前博客的内容找不到了。所以，我决定在新博客的第一篇文章中分享如何创建博客并上传源代码到 GitHub。现在，让我们开始吧！

# Hexo 博客的创建与部署

以下是创建新的 Hexo 博客并部署到 GitHub 的详细步骤：

## 安装前置软件

### 安装 Node.js 和 npm

Hexo 是基于 Node.js 构建的，所以首先你需要安装 Node.js 和 npm（Node 包管理器）。访问 [Node.js 的官方网站](https://nodejs.org/) 进行下载安装。

### 安装 Hexo

在 Node.js 和 npm 安装完成后，通过 npm 全局安装 Hexo。在命令行中运行以下命令：

```bash
npm install -g hexo-cli
```

## 创建新的 Hexo 博客

### 初始化新的 Hexo 博客

创建一个新的文件夹作为你的博客的根目录，然后在命令行中运行以下命令：

```bash
hexo init blog
cd blog
```

这将在 "blog" 文件夹下创建一个新的 Hexo 博客。

### 安装博客依赖

进入你的博客目录，然后运行以下命令来安装博客所需的依赖：

```bash
npm install
```

## 配置你的博客

### 配置 Hexo

使用你的文本编辑器打开 `_config.yml` 文件，这是 Hexo 博客的配置文件。你需要将 `url` 设置为你的 GitHub Pages 的 URL（通常是 `https://<username>.github.io`），并且你可能还想配置其他的一些选项，比如博客的标题、描述和作者信息。

## 部署到 GitHub

### 安装 Hexo 部署插件

首先，你需要安装 `hexo-deployer-git` 插件，这个插件可以让你直接将你的博客部署到 GitHub。在命令行中运行以下命令来安装：

```bash
npm install hexo-deployer-git --save
```

### 配置部署参数

在 `_config.yml` 文件中添加以下配置：

```yml
deploy:
  type: git
  repository: git@github.com:<username>/<username>.github.io.git
  branch: master
```

将 `<username>` 替换为你的 GitHub 用户名。

### 生成静态文件并部署

在命令行中运行以下命令来生成静态文件并将它们部署到 GitHub：

```bash
hexo generate
hexo deploy
```

或者你可以使用下面的单个命令来完成这两个步骤：

```bash
hexo g -d
```

## 添加新文章

你可以使用 Hexo 的 `new` 命令来快速创建新的文章。在命令行中运行以下命令：

```bash
hexo new "文章标题"
```

将 "文章标题" 替换为你想要的文章标题。这将在 `source/_posts` 目录下创建一个新的 Markdown 文件，文件名就是你指定的文章标题（把空格替换为 `-`）。

你可以使用任何你喜欢的文本编辑器打开这个文件，并在里面写下你的文章内容。Hexo 使用 Markdown 语法，你可以查看 [Markdown 语法手册](https://markdown-zh.readthedocs.io/en/latest/) 来学习如何使用 Markdown。

完成后，你可以重新生成并部署你的博客，新的文章就会出现在你的博客上了。

## 文章分类

你可以在你的文章中使用 YAML 前置课（Front-matter）来为文章分配分类（categories）和标签（tags）。前置课应该放在每篇文章的顶部，举例如下：

```markdown
---
title: 文章标题
date: 2023-07-06 00:00:00
categories:
- 分类1
- 分类2
tags:
- 标签1
- 标签2
---

这里是文章的内容。
```

在这个例子中，这篇文章被分配到了 "分类1" 和 "分类2" 这两个分类，同时也被分配了 "标签1" 和 "标签2" 这两个标签。

当你生成你的博客时，Hexo 会自动根据这些分类和标签创建索引，访问者可以通过分类和标签来查找文章。

## 将博客源文件保存到 GitHub

### 创建一个新的 GitHub 仓库

登录到你的 GitHub 账号，然后创建一个新的仓库。你可以给这个仓库取任何你喜欢的名字，比如 `my-hexo-blog`。不需要初始化 README、.gitignore 或者许可证。

### 初始化 Git

在你的博客目录中，运行以下命令来初始化 Git：

```bash
git init
```

### 添加所有文件到 Git

运行以下命令来添加所有文件到 Git：

```bash
git add .
```

### 提交你的更改

运行以下命令来提交你的更改：

```bash
git commit -m "Initial commit"
```

### 添加远程仓库

运行以下命令来添加你刚才在 GitHub 上创建的仓库作为远程仓库：

```bash
git remote add origin git@github.com:<username>/my-hexo-blog.git
```

将 `<username>` 替换为你的 GitHub 用户名。

### 推送到 GitHub

运行以下命令来将你的博客源文件推送到 GitHub：

```bash
git push -u origin master
``````

完成这些步骤后，你的 Hexo 博客就已经部署到 GitHub Pages 上了。你可以访问 `https://<username>.github.io` 来查看你的博客。未来每次你想要添加新的文章，只需在 `source/_posts` 目录下添加新的 Markdown 文件，然后重新生成并部署你的博客就可以了。

在未来，每次你修改了博客源文件（比如添加新的文章），你都需要运行 `git add .`，`git commit -m "your message"` 和 `git push` 命令来更新你在 GitHub 上的备份。