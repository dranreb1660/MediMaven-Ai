import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../ui/Button';
import Card   from '../ui/Card';
import doctorIllustration from '../../assets/ai_doc2.png';
import { useChatStore } from '../../store/useChatStore';

export default function WelcomeCard() {
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  /** Quick placeholder for inactive buttons */
  const comingSoon = (feature: string) =>
    alert(`${feature} coming soon! Describe it in the message box below.`);

  /** Send + route */
  const handleSend = async () => {
    const msg = draft.trim();
    if (!msg || loading) return;

    setLoading(true);

    /* 1️⃣ Hard-reset any existing chat (cid & messages) */
    useChatStore.getState().reset();

    /* 2️⃣ Route to /chat and hand off the message through location.state  */
    navigate('/chat', { state: { sendMessage: msg } });
  };

  return (
    <Card className="w-full max-w-2xl min-h-[85vh] mx-auto flex flex-col gap-6 p-4 sm:p-8 shadow-xl">
      <img
        src={doctorIllustration}
        alt="Virtual medical assistant"
        className="max-w-full h-auto max-h-48 sm:max-h-64 mx-auto rounded-2xl shadow"
      />

      <h1 className="text-2xl sm:text-4xl font-extrabold text-center text-gray-900 dark:text-white">
        Medical AI<br />
        <span className="text-mm-accent">Assistant</span>
      </h1>

      {/* Quick actions */}
      <div className="flex flex-col sm:flex-row gap-2">
        <Button className="flex-1 py-3"           onClick={() => comingSoon('Symptom Checker')}>🩺 Symptoms</Button>
        <Button variant="outline" className="flex-1 py-3" onClick={() => comingSoon('Medical History')}>📋 History</Button>
      </div>

      {/* Temperature dummy card */}
      <div className="flex justify-between items-center bg-mm-blue text-white rounded-2xl px-6 py-3">
        <span>🌡️ Temperature</span>
        <span className="font-bold">98.5°F</span>
      </div>

      {/* Input + send */}
      <div className="flex flex-col sm:flex-row gap-3 mt-auto">
        <input
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Describe your symptoms or ask a health question..."
          disabled={loading}
          maxLength={500}
          className="flex-1 rounded-full border-2 border-gray-300 dark:border-gray-600 px-4 py-3 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-mm-accent disabled:opacity-50"
        />
        <Button
          onClick={handleSend}
          disabled={!draft.trim() || loading}
          className="px-6 py-3 min-w-[80px]"
        >
          {loading ? '⏳' : '📤 Send'}
        </Button>
      </div>
    </Card>
  );
}
