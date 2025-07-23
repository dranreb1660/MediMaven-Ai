import { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import Sidebar from './Sidebar';

export default function MyHistoryControl() {
  const { isAuthenticated, login } = useAuth();
  const [open, setOpen] = useState(false);

  const toggle = () => setOpen((o) => !o);

  return (
    <>
    <button
      onClick={() => (isAuthenticated ? toggle() : login())}
      className={
        isAuthenticated
          ? 'text-xs text-mm-accent-600 hover:underline'
          : 'text-xs text-gray-300 dark:text-gray-700 pointer-events-none'
      }
    >
      My&nbsp;History
    </button>
      {open && <Sidebar onClose={() => setOpen(false)} />}
    </>
  );
}
