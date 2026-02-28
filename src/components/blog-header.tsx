'use client'

import { ThemeToggle } from './theme-toggle'
import { Button } from '@/components/ui/button'
import { BookOpen, Github } from 'lucide-react'

interface BlogHeaderProps {
  onLogoClick: () => void
  showBackButton?: boolean
}

export function BlogHeader({ onLogoClick, showBackButton = false }: BlogHeaderProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-4xl mx-auto px-4 h-14 flex items-center justify-between">
        <Button
          variant="ghost"
          className="flex items-center gap-2 px-2 -ml-2"
          onClick={onLogoClick}
        >
          <BookOpen className="h-5 w-5 text-primary" />
          <span className="font-semibold text-lg">TechBlog</span>
        </Button>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" asChild>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Github className="h-4 w-4" />
            </a>
          </Button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
