import '@testing-library/jest-dom'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

// Add custom matchers types
declare module 'vitest' {
  interface Assertion<T = any> {
    toBeInTheDocument(): T
    toBeVisible(): T
    toBeDisabled(): T
    toHaveClass(className: string): T
    toHaveTextContent(text: string | RegExp): T
  }
}

// Cleanup after each test
afterEach(() => {
  cleanup()
  localStorage.clear()
})

// Mock Auth0
vi.mock('@auth0/auth0-react', () => ({
  useAuth0: () => ({
    isAuthenticated: true,
    user: { sub: 'test-user', email: 'test@medimaven.ai' },
    loginWithRedirect: vi.fn(),
    logout: vi.fn(),
    getAccessTokenSilently: vi.fn().mockResolvedValue('mock-token'),
  }),
  Auth0Provider: ({ children }: { children: React.ReactNode }) => children,
}))

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock Element.prototype.scrollIntoView
Element.prototype.scrollIntoView = vi.fn()
