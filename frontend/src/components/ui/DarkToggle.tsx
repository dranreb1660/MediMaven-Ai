import { useContext } from 'react';
import { ThemeCtx } from '../../context/theme-constants';

export default function DarkToggle() {
  const { dark, toggle } = useContext(ThemeCtx);
  return (
    <button
      onClick={toggle}
      className="rounded-full p-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600"
      title="Toggle dark mode"
    >
      {dark ? '🌙' : '☀️'}
    </button>
  );
}
