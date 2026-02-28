import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/components/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TechBlog - 深度学习技术博客",
  description: "探索深度学习、机器学习和人工智能的前沿技术博客，包含PyTorch、神经网络等深度技术文章。",
  keywords: ["PyTorch", "深度学习", "损失函数", "机器学习", "AI", "技术博客"],
  authors: [{ name: "TechBlog Team" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "TechBlog - 深度学习技术博客",
    description: "探索深度学习、机器学习和人工智能的前沿技术",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "TechBlog - 深度学习技术博客",
    description: "探索深度学习、机器学习和人工智能的前沿技术",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground min-h-screen`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
