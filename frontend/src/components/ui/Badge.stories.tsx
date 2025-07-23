import type { Meta, StoryObj } from '@storybook/react'
import Badge from './Badge'

const meta: Meta<typeof Badge> = {
  title: 'MediMaven/UI/Badge',
  component: Badge,
  parameters: {
    layout: 'centered',
  },
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    children: '1.2s',
  },
}

export const LongLatency: Story = {
  args: {
    children: '15.7s',
  },
}

export const WithText: Story = {
  args: {
    children: 'Verified',
  },
}