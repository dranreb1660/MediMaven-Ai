import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../ui/Button';
import doctorIllustration from '../../assets/medimaven-logo-new-comp.png';
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
    <div className="w-full max-w-lg mx-auto flex flex-col gap-6 p-6 pb-safe">
      
      <img
        src={doctorIllustration}
        alt="Virtual medical assistant"
        className="w-40 h-40 sm:w-48 sm:h-48 mx-auto rounded-3xl shadow-xl object-cover"
      />

      <div className="text-center space-y-2">
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">
          MediMaven AI Assistant
        </h1>
        <p className="text-gray-600 dark:text-gray-400 text-sm sm:text-base">
          ...Your personal health companion
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Button className="flex-1" onClick={() => comingSoon('Symptoms')}>
          <span className="flex items-center justify-center gap-2">
            <span className="text-xl">🩺</span>
            <span>Symptoms</span>
          </span>
        </Button>
        <Button variant="outline" className="flex-1" onClick={() => comingSoon('History')}>
          <span className="flex items-center justify-center gap-2">
            <span className="text-xl">📋</span>
            <span>History</span>
          </span>
        </Button>
      </div>

      <div className="bg-gradient-to-r from-mm-blue-400 to-mm-blue-500 text-white rounded-2xl p-4 shadow-lg">
        <div className="flex justify-between items-center">
          <span className="flex items-center gap-2">
            <span className="text-2xl">🌡️</span>
            <span className="font-medium">Temperature</span>
          </span>
          <span className="text-xl font-bold">98.5°F</span>
        </div>
      </div>

      <div className="flex gap-3 mt-4">
        <input
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask your health question..."
          disabled={loading}
          maxLength={500}
          className="flex-1 rounded-full border-2 border-gray-300 dark:border-gray-600 px-5 py-4 bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-base focus:outline-none focus:ring-2 focus:ring-mm-accent focus:border-transparent disabled:opacity-50 transition-all duration-200"
        />
        <Button
          onClick={handleSend}
          disabled={!draft.trim() || loading}
          className="h-[56px] w-[56px] rounded-full p-0 flex items-center justify-center shadow-lg hover:shadow-xl transition-shadow"
          aria-label="Send message"
        >
          {loading ? (
            <div className="animate-pulse text-xl">⏳</div>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )}
        </Button>
      </div>
      
    </div>
  );
}
