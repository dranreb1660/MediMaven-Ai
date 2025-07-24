import { useDrawer } from '../../context/DrawerContext';
import { Menu } from 'lucide-react';

export default function DrawerToggleFAB() {
  const { toggle } = useDrawer();

  return (
    <button
      aria-label="Open quick menu"
      onClick={toggle}
      className="
        fixed top-4 right-4 z-40                /* ⬅️ was bottom-4 right-4 */
        flex items-center justify-center
        w-12 h-12 rounded-full bg-mm-accent text-white
        shadow-lg hover:scale-105 active:scale-95
        transition-transform
      "
    >
      <Menu size={22} />
    </button>
  );
}