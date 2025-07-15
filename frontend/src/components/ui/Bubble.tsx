interface BubbleProps {
  variant?: 'user' | 'assistant';
  children: React.ReactNode;
}

export default function Bubble({ variant = 'assistant', children }: BubbleProps) {
  const styles = {
    user: 'bg-mm-bubble-user text-white',
    assistant: 'bg-mm-bubble-assistant text-gray-800 dark:bg-mm-bubble-assistantDark dark:text-gray-200',
  };

  const align = variant === 'user' ? 'self-end' : 'self-start';

  return (
    <div
      className={`w-fit max-w-[80%] rounded-3xl px-4 py-2 text-sm leading-relaxed shadow-sm ${styles[variant]} ${align}`}
    >
      {children}
    </div>
  );
}

