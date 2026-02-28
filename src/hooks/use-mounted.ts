'use client'

import { useSyncExternalStore, useCallback } from 'react'

function subscribe(callback: () => void) {
  return () => {}
}

export function useMounted() {
  return useSyncExternalStore(
    subscribe,
    useCallback(() => true, []),
    useCallback(() => false, [])
  )
}
