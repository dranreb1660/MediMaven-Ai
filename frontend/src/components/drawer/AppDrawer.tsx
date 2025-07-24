import { useDrawer } from '../../context/DrawerContext';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import HistoryPanel from './HistoryPanel';
import DarkToggle from '../ui/DarkToggle';
import { useAuth } from '../../hooks/useAuth';

export default function AppDrawer() {
  const { isOpen, close } = useDrawer();
  const { isAuthenticated, isLoading, login, logout, user } = useAuth();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            onClick={close}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black z-40"
          />

          {/* Panel */}
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 260, damping: 25 }}
            className="fixed right-0 top-0 h-full w-72 sm:w-80
                       bg-gray-900 text-gray-100 z-50 flex flex-col"
          >
            {/* header */}
            <div className="flex items-center justify-between px-4 py-3 border-mm-accent border-b">
              <h3 className="text-lg font-semibold">Quick&nbsp;Menu</h3>
              <button onClick={close}>
                <X size={22} />
              </button>
            </div>

            {/* scrollable content (history list) */}
            <div className="flex-1 overflow-y-auto">
              <HistoryPanel onSelect={close} />
            </div>

            {/* footer controls */}
            <div className="border-t border-gray-700 p-4 space-y-4">
              {/* Dark-mode row */}
              <div className="flex items-center justify-between">
                <span className="text-sm hidden sm:inline">Dark&nbsp;mode</span>
                <DarkToggle />
              </div>

              {/* Auth row */}
              {!isLoading && (
                isAuthenticated ? (
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <img
                        src={user?.picture ?? '/avatar.svg'}
                        alt="avatar"
                        className="w-7 h-7 rounded-full"
                      />
                      <span className="text-sm truncate max-w-[7rem]">
                        {user?.name ?? 'User'}
                      </span>
                    </div>
                    <button
                      onClick={logout}
                      className="text-xs text-red-400 hover:text-red-300">
                      Sign&nbsp;out
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={login}
                    className="w-full text-center text-sm py-2 rounded
                               bg-mm-accent hover:bg-mm-accentDark">
                    Sign&nbsp;in
                  </button>
                )
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
