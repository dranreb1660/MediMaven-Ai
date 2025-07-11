
// import './App.css'
import { Routes, Route } from 'react-router-dom';
import Chat from './pages/Chat';
import WelcomePage from './pages/WelcomePage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<WelcomePage />} />
      <Route path="/chat" element={<Chat />} />
    </Routes>
  );
}

export default App
