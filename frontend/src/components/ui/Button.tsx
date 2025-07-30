import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'solid' | 'outline';
  size?: 'sm' | 'md' | 'lg';
}

export default function Button({
  variant = 'solid',
  size = 'md',
  className = '',
  ...props
}: ButtonProps) {
  const base =
    'rounded-lg transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed font-medium active:scale-95 touch-manipulation';

  const variants = {
    solid:
      'bg-mm-accent-500 text-white hover:bg-mm-accent-600 dark:bg-mm-accent-600 dark:hover:bg-mm-accent-500 shadow-md hover:shadow-lg',
    outline:
      'border-2 border-mm-accent-500 text-mm-accent-500 hover:bg-mm-accent-50 dark:border-mm-accent-400 dark:text-mm-accent-400 dark:hover:bg-mm-accent-900/20',
  };

  const sizes = {
    sm: 'text-sm px-4 py-2.5 min-h-[44px]',
    md: 'text-base px-5 py-3 min-h-[48px]',
    lg: 'text-lg px-6 py-3.5 min-h-[52px]',
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    />
  );
}
