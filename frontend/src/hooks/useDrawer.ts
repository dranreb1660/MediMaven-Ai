import { useContext } from 'react';
import { DrawerCtx } from '../context/DrawerContext';

export function useDrawer() {
  const ctx = useContext(DrawerCtx);
  if (!ctx) throw new Error('useDrawer must be inside <DrawerProvider>');
  return ctx;
}
