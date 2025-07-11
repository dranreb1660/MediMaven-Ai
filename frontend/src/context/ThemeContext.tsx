// src/context/ThemeContext.tsx
import { createContext, useState, useEffect } from 'react';

export const ThemeCtx = createContext({ dark: false, toggle: () => {} });

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [dark, setDark] = useState(() => localStorage.theme === 'dark');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.theme = dark ? 'dark' : 'light';
  }, [dark]);

  return (
    <ThemeCtx.Provider value={{ dark, toggle: () => setDark(!dark) }}>
      {children}
    </ThemeCtx.Provider>
  );
}
