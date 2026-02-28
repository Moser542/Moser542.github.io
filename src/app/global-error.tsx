'use client'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html lang="zh-CN">
      <body>
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h1>出错了</h1>
          <p>页面加载时发生错误</p>
          <button onClick={reset}>重试</button>
        </div>
      </body>
    </html>
  )
}
