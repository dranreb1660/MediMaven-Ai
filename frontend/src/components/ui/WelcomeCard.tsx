import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../ui/Button';
import Card from '../ui/Card';
import doctorIllustration from '../../assets/ai_doc2.png';

export default function WelcomeCard() {
  const [draft, setDraft] = useState('');
  const navigate = useNavigate();

  const handleSend = () => {
    const msg = draft.trim();
    if (!msg) return;
    navigate('/chat', { state: { prefill: msg } });
  };

  return (
    <Card className="w-[90vw] max-w-4xl min-h-[80vh] mx-auto flex flex-col gap-8 p-12 shadow-xl">
      <div className="flex justify-center">
        <img
          src={doctorIllustration}
          alt="Medical AI Assistant"
          className="rounded-2xl shadow dark:brightness-90 dark:contrast-110"
        />
      </div>

      <h1 className="text-4xl md:text-5xl font-extrabold text-center leading-tight">
        Medical AI<br />Assistant
      </h1>

      <div className="flex flex-col flex-1 gap-6">
        <div className="grid grid-cols-2 gap-2 rounded-full overflow-hidden text-lg font-semibold">
          <Button onClick={() => alert('Symptoms coming soon')}>Symptom</Button>
          <Button variant="outline" onClick={() => alert('Medical History coming soon')}>
            Medical History
          </Button>
        </div>

        <div className="flex justify-between items-center bg-mm-blue text-white rounded-2xl px-6 py-3 text-lg">
          <span className="font-medium">Temperature</span>
          <span className="font-medium">98.5°F</span>
        </div>

        <div className="flex items-center gap-4 mt-auto">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Message"
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            className="flex-1 rounded-full border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-mm-accent"
          />
          <Button onClick={handleSend}>Send</Button>
        </div>
      </div>
    </Card>
  );
}
