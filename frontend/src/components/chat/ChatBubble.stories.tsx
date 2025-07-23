import type { Meta, StoryObj } from '@storybook/react'
import ChatBubble from './ChatBubble'

const meta: Meta<typeof ChatBubble> = {
  title: 'MediMaven/ChatBubble',
  component: ChatBubble,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    role: {
      control: 'radio',
      options: ['user', 'assistant'],
    },
  },
}

export default meta
type Story = StoryObj<typeof meta>

export const UserMessage: Story = {
  args: {
    role: 'user',
    content: 'What are the symptoms of diabetes?',
  },
}

export const AssistantResponse: Story = {
  args: {
    role: 'assistant',
    content: 'Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision.',
    meta: {
      latency: 1.2,
    },
  },
}

export const WithCitations: Story = {
  args: {
    role: 'assistant',
    content: 'According to recent studies, metformin remains the first-line treatment for type 2 diabetes.',
    meta: {
      citations: [
        { id: '1', source: 'American Diabetes Association', url: '#', rank: 1 },
        { id: '2', source: 'Mayo Clinic', url: '#', rank: 2 },
      ],
      latency: 2.1,
    },
  },
}

export const StreamingResponse: Story = {
  args: {
    role: 'assistant',
    content: 'Analyzing your medical query...',
    meta: {
      streaming: true,
    },
  },
}

export const ErrorState: Story = {
  args: {
    role: 'assistant',
    content: 'Sorry, I encountered an error processing your request.',
    meta: {
      error: true,
    },
    retry: () => console.log('Retry clicked'),
  },
}