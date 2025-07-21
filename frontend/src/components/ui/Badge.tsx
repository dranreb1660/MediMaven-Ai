interface BadgeProps {
  children: React.ReactNode;
  color?: 'default' | 'success' | 'error';
}

export default function Badge({ children, color = 'default' }: BadgeProps) {
  const colors = {
    default: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
    success: 'bg-green-100 text-green-800 dark:bg-green-700 dark:text-green-200',
    error: 'bg-red-100 text-red-800 dark:bg-red-700 dark:text-red-200',
  };

  return (
    <span
      className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${colors[color]}`}
    >
      {children}
    </span>
  );
}
