// src/context/DrawerContext.tsx        🆕 NEW
import { createContext, useContext, useState, ReactNode, useCallback, useEffect } from 'react';

interface DrawerCtx {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const Ctx = createContext<DrawerCtx | null>(null);

export function DrawerProvider({ children }: { children: ReactNode }) {
  const [isOpen, set] = useState(false);
  const open  = useCallback(() => set(true), []);
  const close = useCallback(() => set(false), []);
  const toggle = useCallback(() => set(s => !s), []);
  
  useEffect(()=>{ document.body.classList.toggle('drawer-open',isOpen); },[isOpen]);

  return <Ctx.Provider value={{ isOpen, open, close, toggle }}>{children}</Ctx.Provider>;
}

export const useDrawer = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useDrawer must be inside <DrawerProvider>');
  return ctx;
};
