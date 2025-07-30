import type { Preview, Decorator } from '@storybook/react'
import React from 'react'
import { BrowserRouter } from 'react-router-dom'
import '../src/index.css'

const withRouter: Decorator = (Story) => (
  <BrowserRouter>
    <Story />
  </BrowserRouter>
)

const withBackground: Decorator = (Story) => (
  <div className="min-h-screen bg-gray-50">
    <Story />
  </div>
)

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  decorators: [withRouter, withBackground],
}

export default preview