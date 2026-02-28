'use client'

import { useState } from 'react'
import { BlogHeader } from '@/components/blog-header'
import { BlogList } from '@/components/blog-list'
import { BlogDetail } from '@/components/blog-detail'
import { useMounted } from '@/hooks/use-mounted'

export default function Home() {
  const [selectedPostId, setSelectedPostId] = useState<string | null>(null)
  const mounted = useMounted()

  const handleSelectPost = (postId: string) => {
    setSelectedPostId(postId)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleBack = () => {
    setSelectedPostId(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">加载中...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <BlogHeader
        onLogoClick={handleBack}
        showBackButton={selectedPostId !== null}
      />

      <main className="pb-16">
        {selectedPostId ? (
          <BlogDetail postId={selectedPostId} onBack={handleBack} />
        ) : (
          <BlogList onSelectPost={handleSelectPost} />
        )}
      </main>

      <footer className="border-t border-border py-6 bg-muted/30">
        <div className="max-w-4xl mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>© 2024 TechBlog. 用心记录技术成长.</p>
        </div>
      </footer>
    </div>
  )
}
