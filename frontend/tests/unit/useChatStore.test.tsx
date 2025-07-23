import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { useChatStore } from '../../src/store/useChatStore'

describe('useChatStore', () => {
  beforeEach(() => {
    localStorage.clear()
    useChatStore.setState({ messages: [], cid: undefined })
  })

  it('manages messages lifecycle', () => {
    const { result } = renderHook(() => useChatStore())
    
    // Add message
    act(() => {
      result.current.addMessage({
        id: '1',
        role: 'user',
        content: 'What are symptoms of diabetes?'
      })
    })
    
    expect(result.current.messages).toHaveLength(1)
    expect(localStorage.getItem('medimaven.chat')).toContain('diabetes')
    
    // Edit message
    act(() => {
      result.current.editMessage('1', m => ({ 
        ...m, 
        content: 'What are early symptoms of diabetes?' 
      }))
    })
    
    expect(result.current.messages[0].content).toContain('early symptoms')
    
    // Reset
    act(() => result.current.reset())
    expect(result.current.messages).toHaveLength(0)
  })

  it('persists conversation ID', () => {
    const { result } = renderHook(() => useChatStore())
    
    act(() => result.current.setCid('conv-123'))
    
    expect(result.current.cid).toBe('conv-123')
    const stored = JSON.parse(localStorage.getItem('medimaven.chat') || '{}')
    expect(stored.cid).toBe('conv-123')
  })

  it('handles loading state', () => {
    const { result } = renderHook(() => useChatStore())
    
    expect(result.current.isLoadingFromHistory).toBe(false)
    
    act(() => result.current.setLoadingFromHistory(true))
    expect(result.current.isLoadingFromHistory).toBe(true)
  })
})
