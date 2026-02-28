'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Calendar, Clock, ArrowLeft, Share2 } from 'lucide-react'
import { MarkdownRenderer } from './markdown-renderer'
import { getPost } from '@/lib/posts'

interface BlogDetailProps {
  postId: string
  onBack: () => void
}

export function BlogDetail({ postId, onBack }: BlogDetailProps) {
  const post = getPost(postId)

  if (!post) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8 text-center">
        <h1 className="text-2xl font-bold mb-4">文章未找到</h1>
        <Button onClick={onBack}>返回文章列表</Button>
      </div>
    )
  }

  const handleShare = async () => {
    if (navigator.share) {
      await navigator.share({
        title: post.title,
        text: post.summary,
        url: window.location.href,
      })
    } else {
      await navigator.clipboard.writeText(window.location.href)
    }
  }

  return (
    <article className="max-w-4xl mx-auto px-4 py-8">
      {/* 返回按钮 */}
      <Button
        variant="ghost"
        className="mb-6 -ml-2 text-muted-foreground hover:text-foreground"
        onClick={onBack}
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        返回文章列表
      </Button>

      {/* 文章头部 */}
      <header className="mb-8 pb-6 border-b border-border">
        <h1 className="text-4xl font-bold tracking-tight mb-4 leading-tight">
          {post.title}
        </h1>
        <p className="text-xl text-muted-foreground mb-4 leading-relaxed">
          {post.summary}
        </p>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              {post.date}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              {post.readingTime}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={handleShare}
          >
            <Share2 className="h-4 w-4" />
            分享
          </Button>
        </div>
        <div className="flex flex-wrap gap-2 mt-4">
          {post.tags.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>
      </header>

      {/* 文章内容 */}
      <div className="article-content">
        <MarkdownRenderer content={post.content} />
      </div>

      {/* 文章底部 */}
      <footer className="mt-12 pt-6 border-t border-border">
        <Button
          variant="outline"
          className="w-full sm:w-auto"
          onClick={onBack}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          返回文章列表
        </Button>
      </footer>
    </article>
  )
}
