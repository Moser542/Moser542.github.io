# PyTorch Loss 函数深度解析博客

一个关于 PyTorch 损失函数的技术博客网站。

## 🌐 访问地址

[https://moser542.github.io/](https://moser542.github.io/)

## 📚 博客内容

深入解析 12 种 PyTorch 常用损失函数：

1. **nn.MSELoss** - 均方误差损失
2. **nn.CrossEntropyLoss** - 交叉熵损失
3. **nn.BCELoss / nn.BCEWithLogitsLoss** - 二分类交叉熵
4. **nn.NLLLoss** - 负对数似然损失
5. **nn.L1Loss** - L1 损失
6. **nn.SmoothL1Loss** - 平滑 L1 损失
7. **nn.KLDivLoss** - KL 散度损失
8. **nn.MarginRankingLoss** - 边际排序损失
9. **nn.TripletMarginLoss** - 三元组边际损失
10. **nn.CosineEmbeddingLoss** - 余弦嵌入损失
11. **nn.CTCLoss** - CTC 损失
12. **nn.HingeEmbeddingLoss** - 铰链嵌入损失

每个损失函数包含：数学公式、用途场景、代码示例、注意事项。

## 🛠️ 技术栈

- [Next.js 15](https://nextjs.org/) - React 框架
- [Tailwind CSS](https://tailwindcss.com/) - 样式
- [shadcn/ui](https://ui.shadcn.com/) - UI 组件
- [KaTeX](https://katex.org/) - 数学公式渲染
- [react-syntax-highlighter](https://github.com/react-syntax-highlighter/react-syntax-highlighter) - 代码高亮

## 🚀 本地开发

```bash
# 安装依赖
bun install

# 启动开发服务器
bun run dev

# 构建静态网站
bun run build
```

## 📦 部署

本项目使用 GitHub Actions 自动部署到 GitHub Pages。

## 📄 License

MIT
