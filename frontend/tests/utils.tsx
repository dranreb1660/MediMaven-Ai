import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { ReactElement } from 'react'
import { DrawerProvider } from '../src/context/DrawerContext'

// Test data factories
export const createMockMessage = (overrides = {}) => ({
  id: '1',
  role: 'assistant' as const,
  content: 'Based on medical literature...',
  meta: {
    citations: [
      { id: '1', source: 'Mayo Clinic', url: 'https://mayo.edu', rank: 1 }
    ],
    latency: 1.2,
  },
  ...overrides,
})

export const createMockCitation = (overrides = {}) => ({
  id: '1',
  source: 'Medical Journal',
  url: 'https://example.com',
  rank: 1,
  ...overrides,
})

// Custom render with providers
export const renderWithProviders = (
  ui: ReactElement,
  options?: RenderOptions
) => {
  return render(ui, {
    wrapper: ({ children }) => (
      <BrowserRouter>
        <DrawerProvider>
          {children}
        </DrawerProvider>
      </BrowserRouter>
    ),
    ...options,
  })
}

// API mock helpers
export const mockChatResponse = {
  answer: 'Based on current medical guidelines, migraines can have various causes...',
  citations: [createMockCitation()],
  latency: 1.5,
  conversation_id: 'test-conv-123',
  messages: [],
}

// Wait helpers
export const waitForMs = (ms: number) => 
  new Promise(resolve => setTimeout(resolve, ms))
