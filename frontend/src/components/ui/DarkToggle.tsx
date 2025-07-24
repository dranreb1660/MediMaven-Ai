import { useContext } from 'react';
import { ThemeCtx } from '../../context/theme-constants';

interface Props {
  className?: string;
}

export default function DarkToggle({ className = '' }: Props) {
  const { dark, toggle } = useContext(ThemeCtx);

  // base styles
  const base =
    'rounded-full p-2 bg-gray-200 dark:bg-gray-700 ' +
    'hover:bg-gray-300 dark:hover:bg-gray-600';

  return (
    <button
      onClick={toggle}
      title="Toggle dark mode"
      className={`${base} ${className}`.trim()}
    >
      {dark ? '🌙' : '☀️'}
    </button>
  );
}
