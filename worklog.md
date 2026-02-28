# Project Work Log

---
Task ID: 1
Agent: full-stack-developer
Task: 创建博客网站，包含PyTorch Loss函数解析文章

Work Log:
- 创建了 worklog.md 工作日志文件
- 创建了 ThemeProvider 组件支持深色/浅色模式切换
- 创建了 ThemeToggle 主题切换按钮组件
- 创建了 CodeBlock 代码高亮组件（使用 react-syntax-highlighter）
- 创建了 MarkdownRenderer Markdown渲染组件
- 创建了详细的 PyTorch Loss函数解析文章数据（12种Loss函数详解）
- 创建了 BlogList 博客列表组件
- 创建了 BlogDetail 博客详情页组件
- 创建了 BlogHeader 导航头部组件
- 创建了 useMounted 自定义 Hook 处理客户端水合
- 更新了 layout.tsx 添加 ThemeProvider
- 创建了主页组件整合所有功能
- 运行 lint 检查并修复了所有问题

Stage Summary:
- 完成的主要功能：
  1. 单页应用博客网站，支持文章列表和详情页切换
  2. 深色/浅色模式自动切换
  3. 代码语法高亮（支持Python等多种语言）
  4. 响应式设计，适配移动端
  5. 现代简洁的UI风格

- 使用的核心技术：
  1. Next.js 15 App Router
  2. Tailwind CSS + shadcn/ui
  3. react-syntax-highlighter 代码高亮
  4. next-themes 主题切换
  5. react-markdown Markdown渲染

- 产出物：
  1. src/components/theme-provider.tsx - 主题提供者
  2. src/components/theme-toggle.tsx - 主题切换按钮
  3. src/components/code-block.tsx - 代码高亮组件
  4. src/components/markdown-renderer.tsx - Markdown渲染器
  5. src/components/blog-list.tsx - 博客列表
  6. src/components/blog-detail.tsx - 博客详情
  7. src/components/blog-header.tsx - 导航头部
  8. src/lib/posts.ts - 博客数据（PyTorch Loss函数解析文章）
  9. src/hooks/use-mounted.ts - 客户端水合Hook
  10. src/app/page.tsx - 主页组件
  11. src/app/layout.tsx - 布局文件（更新）

- 博客文章内容包含：
  1. nn.MSELoss - 均方误差损失
  2. nn.CrossEntropyLoss - 交叉熵损失
  3. nn.BCELoss / nn.BCEWithLogitsLoss - 二分类交叉熵
  4. nn.NLLLoss - 负对数似然损失
  5. nn.L1Loss - L1损失
  6. nn.SmoothL1Loss - 平滑L1损失
  7. nn.KLDivLoss - KL散度损失
  8. nn.MarginRankingLoss - 边际排序损失
  9. nn.TripletMarginLoss - 三元组边际损失
  10. nn.CosineEmbeddingLoss - 余弦嵌入损失
  11. nn.CTCLoss - CTC损失
  12. nn.HingeEmbeddingLoss - 铰链嵌入损失

  每个Loss函数包含：数学公式、用途场景、代码示例、注意事项
