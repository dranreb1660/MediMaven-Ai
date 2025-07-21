// import './App.css'
import { Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import WelcomePage from './pages/WelcomePage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<WelcomePage />} />
      <Route path="/chat" element={<ChatPage />} />
    </Routes>
  );
}

export default App
