# paced_checkpointed_redo.py
import asyncio, time, json, os, pathlib
from collections import defaultdict 
import contextlib

import pandas as pd
from openai import OpenAI, RateLimitError
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────
READ_PATH          = "../data/processed/redo_df.parquet"
OUT_DIR            = pathlib.Path("../data/min_cpts")
OUT_DIR.mkdir(exist_ok=True)

API_KEYS           = [os.getenv("OPENAI_API_KEY_PHADE"),
                      os.getenv("OPENAI_API_KEY_DRANREB")]
MODEL              = "gpt-4o-mini-2024-07-18"
MAX_TOKENS         = 250
TOKENS_PER_REQ     = 475 + MAX_TOKENS      # prompt + passage + output
TPM_LIMIT_PER_KEY  = 200_000               # tokens / minute per key
INTERVAL_PER_CALL  = TOKENS_PER_REQ / TPM_LIMIT_PER_KEY * 60

SYSTEM_PROMPT      = "You are a helpful medical Q&A assistant."
CHECKPOINT_PERIOD  = 60                    # seconds

# ── GLOBAL STATE & LOCKS ─────────────────────────────────
lock            = asyncio.Lock()
total_requests  = 0
total_tokens    = 0
minute_counter  = 0

# Shared results
results = {}   # id -> qa
fails   = {}   # id -> None

# ── HELPERS ──────────────────────────────────────────────
def build_prompt(passage: str) -> str:
    return f"""PASSAGE:
```{passage}```
You are a patient who has just read the passage above.
Generate EXACTLY 2 lay-person questions answerable ONLY from the passage.
For each, give a concise (<=75 words) doctor reply grounded in the passage.
Output JSON list: [{{"question":"...","answer":"..."}}]"""

def parse_qa(raw: str):
    try:
        obj = json.loads(raw)
        if "questions" in obj: return obj["questions"]
        if "question" in obj:  return [obj]
    except:
        pass
    return None

def save_jsonl(d: dict, path: pathlib.Path):
    if not d: return
    with path.open("w") as f:
        for cid, qa in d.items():
            f.write(json.dumps({"id": cid, "qa": qa}, ensure_ascii=False) + "\n")

# ── KEY WORKER ────────────────────────────────────────────
async def key_worker(key_idx, records):
    """
    Each key does its own paced loop.  On RateLimitError we retry up to
    3× with back-off; on 3rd failure we write resume_point.json and raise.
    """
    global total_requests, total_tokens

    client = OpenAI(api_key=API_KEYS[key_idx])
    resume_file = OUT_DIR / "resume_point.json"
    error_cpt = OUT_DIR / f"error_{key_idx:03d}_df_at"

    for pos, rec in tqdm(enumerate(records), desc=f"Key{key_idx}", total=len(records)):
        for attempt in range(1, 4):  # 1,2,3
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_prompt(rec.text)}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=MAX_TOKENS,
                    temperature=0.2
                )
                qa = parse_qa(resp.choices[0].message.content)
                (results if qa else fails)[rec.id] = qa

                # bump the per-minute counters
                async with lock:
                    total_requests += 1
                    total_tokens   += TOKENS_PER_REQ

                break   # success → exit retry loop

            except RateLimitError:
                if attempt < 3:
                    wait = 2 ** attempt  # 2s, then 4s
                    print(f"Key{key_idx} 429 on pos={pos} attempt={attempt}, sleeping {wait}s")
                    await asyncio.sleep(wait)
                else:
                    # after 3rd try, give up and write resume point
                    print(f"❗ Key{key_idx} hard 429 at pos={pos}, df_index={rec.Index}; stopping.")
                    path = pathlib.Path(f"{error_cpt}_{rec.Index}.jsonl")
                    with resume_file.open("w") as f:
                        json.dump({
                            "key": key_idx,
                            "partition_pos": pos,
                            "df_index": rec.Index
                        }, f)
                    save_jsonl(results, path)
                        
                    raise

            except Exception as e:
                # other unexpected errors → checkpoint & stop
                print(f"⚠️ Key{key_idx} error at pos={pos}, df_index={rec.Index}: {e}")
                path = pathlib.Path(f"{error_cpt}_{rec.Index}.jsonl")

                with resume_file.open("w") as f:
                    json.dump({
                        "key": key_idx,
                        "partition_pos": pos,
                        "df_index": rec.Index,
                        "error": str(e)
                    }, f)
                    
                
                raise
        # pace under token budget regardless of success or not
        await asyncio.sleep(INTERVAL_PER_CALL/2)


# ── MONITOR & CHECKPOINTER ───────────────────────────────
async def monitor_and_checkpoint():
    global total_requests, total_tokens, minute_counter
    while True:
        await asyncio.sleep(CHECKPOINT_PERIOD)
        minute_counter += 1
        async with lock:
            r = total_requests
            t = total_tokens
            total_requests = 0
            total_tokens   = 0
        pct = t / TPM_LIMIT_PER_KEY * 100
        print(f"\n⏰ Minute {minute_counter}: {r} req | {t:,} tok ({pct:4.1f}% TPM)\n")

        # save a checkpoint of everything so far
        cp_r = OUT_DIR / f"cp_minute_{400+minute_counter:05d}.jsonl"
        # cp_f = OUT_DIR / f"fails_minute_{minute_counter:03d}.jsonl"
        save_jsonl(results, cp_r)
        # save_jsonl(fails,   cp_f)
        print(f"↪ checkpoint saved: {cp_r}\n")

# ── MAIN ─────────────────────────────────────────────────
async def main():
    df = pd.read_parquet(READ_PATH)[24000:]
    rows = list(df.itertuples())
    # split records round-robin across keys
    partitions = [rows[i::len(API_KEYS)] for i in range(len(API_KEYS))]

    # start monitor
    mon_task = asyncio.create_task(monitor_and_checkpoint())

    # start key workers
    tasks = [
        asyncio.create_task(key_worker(i, partitions[i]))
        for i in range(len(API_KEYS))
    ]
    await asyncio.gather(*tasks)

    # final checkpoint
    save_jsonl(results, OUT_DIR / "cp_final.jsonl")
    save_jsonl(fails,   OUT_DIR / "fails_final.jsonl")
    print(f"\n🏁 Completed {len(results)} OK | {len(fails)} failed")

    # stop monitor
    mon_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await mon_task

if __name__ == "__main__":
    asyncio.run(main())
