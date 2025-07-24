// src/App.tsx                             🔄 UPDATED
import { Routes, Route } from 'react-router-dom';
import Chat from './pages/Chat';
import WelcomePage from './pages/WelcomePage';
import { DrawerProvider } from './context/DrawerContext';
import AppDrawer        from './components/drawer/AppDrawer';

function App() {
  return (
    <DrawerProvider>
      <Routes>
        <Route path="/"     element={<WelcomePage />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>


      <AppDrawer />
    </DrawerProvider>
  );
}

export default App;
