import type { NextConfig } from "next";

// 用户页面部署到 https://Moser542.github.io/
// basePath 设为空字符串
const nextConfig: NextConfig = {
  // 使用静态导出模式用于 GitHub Pages
  output: "export",
  
  // 用户页面部署到根路径，basePath 为空
  basePath: '',
  
  // 静态导出不支持图片优化，需要禁用
  images: {
    unoptimized: true,
  },
  
  // 确保 trailingSlash 一致性
  trailingSlash: true,
  
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
