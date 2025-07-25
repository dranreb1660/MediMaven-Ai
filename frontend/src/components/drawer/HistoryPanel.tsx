// src/components/history/HistoryPanel.tsx   🔄 UPDATED (was Sidebar.tsx)
import { useEffect, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { useChatStore } from '../../store/useChatStore';
import { fetchJson } from '../../lib/fetchJson';

const API_BASE    = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const AUDIENCE    = import.meta.env.VITE_AUTH0_AUDIENCE;
const STORAGE_KEY = 'medimaven.chatHistory';

interface ConvMeta { cid:string; preview:string; messages:any[] }

export default function HistoryPanel({ onSelect }:{ onSelect: () => void }) {
  const { isAuthenticated, getAccessToken } = useAuth();
  const [items,setItems] = useState<ConvMeta[]>([]);
  const [loading,setLoading]=useState(false);

  // hydrate immediately
  useEffect(()=>{
    if(!isAuthenticated) return;
    try { setItems(JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')); }
    catch{}
  },[isAuthenticated]);

  // refresh once per open
  useEffect(()=>{
    let done=false;
    (async()=>{
      if(!isAuthenticated) return;
      setLoading(true);
      const token = await getAccessToken({ authorizationParams:{ audience:AUDIENCE }});
      const fresh:ConvMeta[]=await fetchJson(`${API_BASE}/chat/list`,{method:'GET'},token);
      if(!done){ setItems(fresh); localStorage.setItem(STORAGE_KEY,JSON.stringify(fresh)); }
    })().finally(()=>!done&&setLoading(false));
    return()=>{done=true};
  },[isAuthenticated,getAccessToken]);

  const loadChat = (it:ConvMeta)=>{
    const { reset,setCid,addMessage } = useChatStore.getState();
    reset(); setCid(it.cid);
    it.messages.forEach((t,i)=>{
      if(t.user)       addMessage({ id:`u-${i}`, role:'user',       content:t.user       });
      if(t.assistant)  addMessage({ id:`a-${i}`, role:'assistant',  content:t.assistant, meta:{ citations:t.citations, latency:t.latency }});
    });
    onSelect();
  };

  return (
    <ul className="px-4 py-2 overflow-y-auto space-y-1 flex-1 ">
      {items.map(it=>(
        <li key={it.cid}
            onClick={()=>loadChat(it)}
            className="p-2 rounded cursor-pointer truncate text-gray-700 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 text-sm">
          {it.preview}
        </li>
      ))}
      {loading && <p className="p-2 text-xs text-gray-500 animate-pulse">Retrieving chats…</p>}
      {!loading && items.length===0 && <p className="p-2 text-xs text-gray-500">No history yet.</p>}
    </ul>
  );
}
