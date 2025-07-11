
import asyncio, uvloop, itertools, json, os, pathlib
from collections import defaultdict

import pandas as pd
from aiolimiter import AsyncLimiter                # pip install aiolimiter>=1.1
from openai import AsyncOpenAI, OpenAIError, RateLimitError
from tqdm.asyncio import tqdm as tqdm_asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ── CONFIG ───────────────────────────────────────────────
READ_PATH   = "../data/processed/redo_df.parquet"
OUT_DIR     = pathlib.Path("../data/redo_checkpoints"); OUT_DIR.mkdir(exist_ok=True)

KEYS        = [os.getenv("OPENAI_API_KEY_PHADE"),
               os.getenv("OPENAI_API_KEY_DRANREB")
               ]
MODEL       = "gpt-4o-mini-2024-07-18"

MAX_TOKENS  = 250                              # answer cap
TOKENS_PER_REQ = 475 + MAX_TOKENS              # prompt+passage+out
TPM_LIMIT   = 200_000                          # tokens/min/key
RPM_LIMIT   = 400                              # requests/min/key (dashboard)

CONCURRENCY = 8                                # in-flight per key
CHUNK_SIZE  = 2_000
SYSTEM      = "You are a helpful medical Q&A assistant."

# ── LIMITERS, CLIENTS, SEMAPHORES ────────────────────────
clients   = [AsyncOpenAI(api_key=k) for k in KEYS]
token_lims = [AsyncLimiter(TPM_LIMIT, 60)  for _ in KEYS]
req_lims   = [AsyncLimiter(RPM_LIMIT, 60)  for _ in KEYS]
sems       = [asyncio.Semaphore(CONCURRENCY) for _ in KEYS]
key_cycle         = itertools.cycle(range(len(KEYS)))

# ── GLOBAL COUNTERS ──────────────────────────────────────
lock = asyncio.Lock()
total_req = total_tok = 0

async def ticker():
    global total_req, total_tok
    while True:
        await asyncio.sleep(60)
        async with lock:
            r, t = total_req, total_tok
            total_req = total_tok = 0
        pct = t / TPM_LIMIT * 100
        print(f"\n⏰ last 60 s → {r} req | {t:,} tok  ({pct:4.1f}% of 200 k)\n")

# ── HELPERS ──────────────────────────────────────────────
def build_prompt(p:str)->str:
    return f"""PASSAGE:
```{p}```
You are a patient who has just read the passage above.
Generate EXACTLY 2 lay-person questions answerable ONLY from the passage.
For each, give a concise (<=75 words) doctor reply grounded in the passage.
Output JSON list: [{{"question":"...","answer":"..."}}] ### DO NOT change key names ###"""

def parse_qa(raw:str):
    try:
        obj = json.loads(raw)
        if "questions" in obj: return obj["questions"]
        if "question"  in obj: return [obj]
    except json.JSONDecodeError:
        pass
    return None

def save_jsonl(d:dict, path:pathlib.Path):
    if d:
        with path.open("w") as f:
            for cid, qa in d.items():
                f.write(json.dumps({"id":cid,"qa":qa}, ensure_ascii=False)+"\n")
                
async def write_checkpoint(path, data):
    await asyncio.to_thread(save_jsonl, data, path)     
# rolling_bucket.py
import asyncio, collections, time

class SlidingTokenBucket:
    """
    60-second sliding-window token bucket.
    `await bucket.acquire(weight)` sleeps until weight tokens fit.
    """

    def __init__(self, max_tokens: int):
        self.max = max_tokens
        self._total = 0                         # tokens currently in window
        self._q = collections.deque()           # (timestamp, weight)
        self._lock = asyncio.Lock()

    async def acquire(self, weight: int):
        while True:
            async with self._lock:
                self._evict_expired()
                if self._total + weight <= self.max:
                    now = time.monotonic()
                    self._q.append((now, weight))
                    self._total += weight
                    return                    # we’re in!
                # need to wait until earliest event ages out
                oldest_ts, oldest_w = self._q[0]
                wait = max( (oldest_ts + 60) - time.monotonic(), 0.05)
            await asyncio.sleep(wait)

    def _evict_expired(self):
        """Pop events older than 60 s."""
        cutoff = time.monotonic() - 60
        while self._q and self._q[0][0] <= cutoff:
            _, w = self._q.popleft()
            self._total -= w


# ── SINGLE CALL ──────────────────────────────────────────
async def one_call(text, cid, kidx, buckets, retries=3):
    bucket = buckets[kidx]
    for attempt in range(retries):
        try:
            async with sems[kidx]:             # 1️⃣ worker slot
                await bucket.acquire(TOKENS_PER_REQ)   # 2️⃣ tokens
                async with req_lims[kidx]:     # 3️⃣ 400-RPM gate
                    resp = await clients[kidx].chat.completions.create(
                        model=MODEL,
                        messages=[{"role":"system","content":SYSTEM},
                                {"role":"user","content":build_prompt(text)}],
                        response_format={"type":"json_object"},
                        max_tokens=MAX_TOKENS,
                        temperature=0.2,
                        timeout=90
                    )
            qa = parse_qa(resp.choices[0].message.content)

            async with lock:
                global total_req, total_tok
                total_req  += 1
                total_tok  += TOKENS_PER_REQ
            return cid, qa

        except RateLimitError:
            await asyncio.sleep(30)           # let the window drain
        except OpenAIError:
            await asyncio.sleep(2 ** attempt + 0.5)
    return cid, None

# ── MAIN LOOP ────────────────────────────────────────────
async def process_df(df,chunk_no=26):
    buckets = [SlidingTokenBucket(200_000) for _ in KEYS]   # per key
    monitor = asyncio.create_task(ticker())
    total_ok = 0
 
    try:
        for start in range(0, len(df), CHUNK_SIZE):
            batch = df.iloc[start:start+CHUNK_SIZE]
            tasks = [asyncio.create_task(
                        one_call(r.text, r.id, next(key_cycle), buckets))
                    for r in batch.itertuples()]

            results, fails = {}, {}
            with tqdm_asyncio(total=len(tasks), desc=f"chunk {chunk_no}") as bar:
                i = 1
                for fut in asyncio.as_completed(tasks):
                    cid, qa = await fut
                    (results if qa else fails)[cid] = qa
                        
                    if i == int(0.5 * len(tasks)):
                        path = OUT_DIR / f"cp_{chunk_no:03d}part_{i}.jsonl"
                        # schedule the write, but don't await it here
                        asyncio.create_task(write_checkpoint(path, results.copy()))
                        await asyncio.sleep(5)
                        # break
                        
                    i += 1    
                    bar.update()
                    
                    await asyncio.sleep(60/550)
                    

            save_jsonl(results, OUT_DIR / f"cp_{chunk_no:03d}.jsonl")
            save_jsonl(fails,   OUT_DIR / f"fails_{chunk_no:03d}.jsonl")
            total_ok += len(results)
            print(f"✓ checkpoint {chunk_no} – cumulative OK {total_ok:,}")
            chunk_no += 1
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)

    print(f"🏁 done – parsed {total_ok:,} passages")

# ── RUN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_parquet(READ_PATH)
    asyncio.run(process_df(df[26000:]))
