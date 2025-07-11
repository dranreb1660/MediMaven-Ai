import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../ui/Button';
import doctorIllustration from '../../assets/ai_doc2.png';

/**
 * WelcomeCard – sends the first user query via `location.state` instead of a
 * URL search param, so page refreshes won’t re‑fire the same request.
 */
export default function WelcomeCard() {
  const [draft, setDraft] = useState('');
  const navigate = useNavigate();

  const handleSend = () => {
    const msg = draft.trim();
    if (!msg) return;
    // Pass the query via navigation *state* (not query‑string)
    navigate('/chat', { state: { prefill: msg } });
  };

  return (
    <div className="bg-bg-white dark:bg-gray-800 w-[90vw] max-w-4xl min-h-[80vh] mx-auto flex flex-col gap-8 p-12 rounded-[2rem] shadow-xl">
      {/* Illustration */}
      <div className="flex justify-center">
        <img src={doctorIllustration} alt="Medical AI Assistant" className="rounded-2xl shadow
             dark:brightness-90 dark:contrast-110" />
      </div>

      <h1 className="text-4xl md:text-5xl font-extrabold text-center leading-tight">
        Medical AI<br />Assistant
      </h1>

      <div className="flex flex-col flex-1 gap-6">
        <div className="grid grid-cols-2 rounded-full overflow-hidden text-lg font-semibold">
          {/* <button onClick={() => navigate('/chat')} className="py-4 bg-mm-accent text-white">Symptom</button> */}
          <button onClick={() => alert('Symptoms coming soon')} className="py-4 bg-mm-accent text-white">Symptom</button>
          <button onClick={() => alert('Medical History coming soon')} className="py-4 bg-mm-info text-gray-900">Medical History</button>
        </div>

        <div className="flex justify-between items-center bg-blue-500 text-white rounded-2xl px-6 py-3 text-lg">
          <span className="font-medium">Temperature</span>
          <span className="font-medium">98/5 °F</span>
        </div>

        <div className="flex items-center gap-4 mt-auto">
          <input
            type="text"
            value={draft}
            onChange={e => setDraft(e.target.value)}
            placeholder="message"
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            className="flex-1 rounded-full border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-mm-accent"
          />
          <Button label="Send" onClick={handleSend} />
        </div>
      </div>
    </div>
  );
}
