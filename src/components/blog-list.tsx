'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Calendar, Clock } from 'lucide-react'
import { posts } from '@/lib/posts'

interface BlogListProps {
  onSelectPost: (postId: string) => void
}

export function BlogList({ onSelectPost }: BlogListProps) {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight mb-2">技术博客</h1>
        <p className="text-muted-foreground text-lg">
          探索深度学习、机器学习和人工智能的前沿技术
        </p>
      </div>

      <div className="space-y-6">
        {posts.map((post) => (
          <Card
            key={post.id}
            className="cursor-pointer transition-all duration-200 hover:shadow-lg hover:border-primary/50"
            onClick={() => onSelectPost(post.id)}
          >
            <CardHeader className="pb-3">
              <CardTitle className="text-xl font-semibold hover:text-primary transition-colors">
                {post.title}
              </CardTitle>
              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2">
                <span className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  {post.date}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  {post.readingTime}
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed mb-4">
                {post.summary}
              </p>
              <div className="flex flex-wrap gap-2">
                {post.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
