interface BubbleProps {
  variant?: 'user' | 'assistant';
  children: React.ReactNode;
}

export default function Bubble({ variant = 'assistant', children }: BubbleProps) {
  const styles = {
    user: 'bg-mm-bubble-user text-white shadow-lg',
    assistant: 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 shadow-md',
  };

  const align = variant === 'user' ? 'self-end' : 'self-start';
  const radius = variant === 'user' 
    ? 'rounded-3xl rounded-br-xl' 
    : 'rounded-3xl rounded-bl-xl';

  return (
    <div
      className={`w-fit max-w-[85%] sm:max-w-[75%] px-5 py-3.5 ${radius} ${styles[variant]} ${align} animate-fade-in`}
    >
      {children}
    </div>
  );
}

