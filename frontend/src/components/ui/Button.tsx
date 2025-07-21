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
    'rounded-lg transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed';

  const variants = {
    solid:
      'bg-mm-accent-500 text-white hover:bg-mm-accent-600 dark:bg-mm-accent-700 dark:hover:bg-mm-accent-600',
    outline:
      'border border-mm-accent-500 text-mm-accent-500 hover:bg-mm-accent-50 dark:border-mm-accent-700 dark:text-mm-accent-400 dark:hover:bg-mm-accent-900',
  };

  const sizes = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5',
    lg: 'text-base px-4 py-2',
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    />
  );
}
