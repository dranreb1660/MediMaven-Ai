import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../ui/Button';
import doctorIllustration from '../../assets/ai_doc2.png';
import { useChatStore } from '../../store/useChatStore';

export default function WelcomeCard() {
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const comingSoon = (feature: string) =>
    alert(`${feature} coming soon! Describe it in the message box below.`);

  const handleSend = () => {
    const msg = draft.trim();
    if (!msg || loading) return;

    setLoading(true);
    useChatStore.getState().reset();
    navigate('/chat', { state: { sendMessage: msg } });
  };

  return (
    <div className="w-full max-w-lg mx-auto flex flex-col gap-8 p-6">
      
      <img
        src={doctorIllustration}
        alt="Virtual medical assistant"
        className="w-48 h-48 mx-auto rounded-2xl shadow-lg object-cover"
      />

      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Medical AI Assistant
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Your personal health companion
        </p>
      </div>

      <div className="flex flex-col xs:flex-row gap-3">
        <Button className="flex-1 py-3" onClick={() => comingSoon('Symptoms')}>
          🩺 Symptoms
        </Button>
        <Button variant="outline" className="flex-1 py-3" onClick={() => comingSoon('History')}>
          📋 History
        </Button>
      </div>

      <div className="flex justify-between items-center bg-mm-blue-500 text-white rounded-xl px-4 py-3">
        <span>🌡️ Temperature</span>
        <span className="font-semibold">98.5°F</span>
      </div>

      <div className="flex flex-col xs:flex-row gap-3">
        <input
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask your health question..."
          disabled={loading}
          maxLength={500}
          className="flex-1 rounded-full border-2 border-gray-300 dark:border-gray-600 px-4 py-3 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-mm-accent disabled:opacity-50"
        />
        <Button
          onClick={handleSend}
          disabled={!draft.trim() || loading}
          className="px-6 py-3 w-3/5 mx-auto xs:w-auto xs:mx-0 xs:min-w-[100px]"
        >
          {loading ? '⏳' : '📤'}
        </Button>
      </div>
      
    </div>
  );
}
