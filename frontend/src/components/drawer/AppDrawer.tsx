import { useDrawer } from '../../context/DrawerContext';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import HistoryPanel from './HistoryPanel';
import DarkToggle from '../ui/DarkToggle';
import { useAuth } from '../../hooks/useAuth';
import Disclaimer from '../ui/Disclaimer';

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
            className="fixed right-0 top-0 h-full w-80 sm:w-96
                       bg-white dark:bg-gray-900 shadow-2xl z-50 flex flex-col will-change-transform"
          >
            {/* header */}
            <div className="flex items-center justify-between px-6 py-5 border-b border-gray-200 dark:border-gray-800">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Menu</h3>
              <button 
                onClick={close}
                className="p-2 -m-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <X size={24} className="text-gray-600 dark:text-gray-400" />
              </button>
            </div>

            {/* scrollable content (history list) */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              <HistoryPanel onSelect={close} />
            </div>
            
            <Disclaimer />

            {/* footer controls */}
            <div className="border-t border-gray-200 dark:border-gray-800 p-3 pb-safe space-y-0 bg-gray-50 dark:bg-gray-800">
              {/* Dark-mode row */}
              <div className="flex items-center justify-end">
                <DarkToggle />
              </div>

              {/* Auth row */}
              {!isLoading && (
                isAuthenticated ? (
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <img
                        src={user?.picture ?? '/avatar.svg'}
                        alt="avatar"
                        className="w-10 h-10 rounded-full border-2 border-gray-200 dark:border-gray-700"
                      />
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate max-w-[150px]">
                        {user?.name ?? 'User'}
                      </span>
                    </div>
                    <button
                      onClick={logout}
                      className="text-sm text-red-500 hover:text-red-600 dark:text-red-400 dark:hover:text-red-300 font-medium">
                      Sign out
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={login}
                    className="w-full text-center py-3 rounded-lg
                               bg-mm-accent hover:bg-mm-accent-600 text-white font-medium
                               shadow-md hover:shadow-lg transition-all duration-200">
                    Sign in
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
