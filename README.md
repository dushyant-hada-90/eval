# Realtime Agents Eval

Evaluate realtime voice agents (and optionally STT/TTS) with:

- Live checkpoint UI (human approvals + Gradio/Groq TTS)
- Agent-only latency (**TTF** / **FTL**)
- Groq LLM scoring + improvised caller lines
- SQLite results + conversation audio timeline at `/results`

---

## 1. Clone

```bash
git clone https://github.com/dushyant-hada-90 
```



---

## 2. Create a virtualenv and install deps

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configure environment

```powershell
# Windows
copy .env.dummy .env
```

```bash
# macOS / Linux
cp .env.dummy .env
```

Edit `.env` and fill at least:

| Key | Used for |
|-----|----------|
| `OPENAI_API_KEY` | GPT Realtime agent (default in UI) |
| `GROQ_API_KEY` | STT + scoring LLM (+ Groq TTS fallback) |
| `GEMINI_API_KEY` | Gemini Live agent (optional) |
| `GRADIO_TTS_URL` | Cloned-voice TTS share URL (optional) |

Optional but common:

```env
GRADIO_TTS_URL=https://xxxxxx.gradio.live
GRADIO_TTS_PATH=auto
GRADIO_TTS_FN_INDEX=0
DEFAULT_TTS_PROVIDER=gradio
DASHBOARD_PORT=8000
```

On dashboard startup the server probes Gradio. If it is down or unset, live TTS falls back to **Groq** and shows a banner on `/`.

---

## 4. Initialize the database

```bash
python main.py init-db
```

This creates `eval_results.db` (ignored by git).

Optional sanity check:

```bash
python main.py list-providers
```

---

## 5. Start the dashboard (main workflow)

```bash
python main.py dashboard
```

Open:

- Live checkpoints: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Results / timeline: [http://127.0.0.1:8000/results](http://127.0.0.1:8000/results)

### Defaults on `/`

- Scenario: `scenarios/myntra_support_rohit.yaml`
- Variation: `delayed_kurta`
- Agent: `gpt_realtime`
- TTS: `auto` (Gradio if healthy, else Groq)

### Live checkpoint loop

1. **Start Realtime** — agent connects and greets (TTF measured once)
2. **Approve transcript** — edit Groq STT text if needed → judge LLM scores + invents next caller line
3. **Approve utterance** → TTS (Gradio/Groq)
4. Optionally **edit text + Regenerate TTS**
5. **Send TTS → Realtime** — agent reply + FTL
6. Repeat / **Stop** — turns saved to SQLite

---

## 6. CLI evaluation (optional)

Keep the venv activated, then:

```bash
# Realtime scenario (scripted CLI path)
python main.py run --scenario scenarios/sales_pitch_test.yaml --variation neutral
python main.py run --scenario scenarios/myntra_support_rohit.yaml --agent gpt_realtime --variation delayed_kurta

# STT bench
python main.py run --scenario scenarios/stt_basic.yaml --provider groq

# TTS bench
python main.py run --scenario scenarios/tts_basic.yaml --provider groq
```

Results land in `eval_results.db` and under `recordings/`.

---

## 7. Useful commands cheat sheet

| Command | What it does |
|---------|----------------|
| `python main.py init-db` | Create / migrate SQLite |
| `python main.py list-providers` | List realtime / STT / TTS adapters |
| `python main.py dashboard` | Live UI + results on port 8000 |
| `python main.py run --scenario …` | Batch eval from YAML |
| `pytest -q` | Run unit tests |

---

## Project layout

```
agents/                 # realtime adapters (gpt_realtime, gemini_realtime, …)
stt/                    # STT adapters
tts/                    # TTS adapters (incl. gradio)
audio/                  # PCM helpers + session mix
scoring/                # Groq judge / caller LLM
engine/                 # CLI + live checkpoint session
dashboard/              # FastAPI UI (/ and /results)
scenarios/*.yaml        # realtime_prompt + testing_prompt + persona
utils/                  # config, logging, Gradio health check
```

---

## Scenario YAML (realtime)

Each file under `scenarios/` typically includes:

- `realtime_prompt` — system prompt for the realtime agent
- `testing_prompt` — scoring rubric for the judge LLM
- `test_variations[].persona` — caller persona (live UI improvises speech from this)
- optional `test_script` — soft exchange budget / CLI scripting

| `eval_type` | Purpose |
|-------------|---------|
| `realtime` | Voice agent eval (live or CLI) |
| `stt` | Transcription latency + WER |
| `tts` | Synthesis latency (+ optional WER) |

---

## Latency meanings

| Metric | Meaning |
|--------|---------|
| **TTF** | Session ready → first agent audio (once per call) |
| **FTL** | User/caller audio fully sent → first agent audio (per turn) |

Judge LLM, STT, TTS generation, and human approval time are **not** part of TTF/FTL. They appear on `/results` only as conversation audio + purple TTF/FTL wait bars.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gradio banner says unavailable | Set a live `GRADIO_TTS_URL`, or rely on Groq fallback |
| GPT Realtime fails to connect | Check `OPENAI_API_KEY` / model name in `.env` |
| Empty `/results` | Finish a live call and **approve** at least one transcript (persistence happens then) |
| Port in use | Change `DASHBOARD_PORT` in `.env` or stop the other process |
| Reset local data | Delete `eval_results.db` and `recordings/*`, then `python main.py init-db` |

---

## Add a new STT/TTS provider

1. Implement the base class in `stt/` or `tts/`.
2. Register with `@stt_registry.register("name")` / `@tts_registry.register("name")`.
3. Import the module in the package `__init__.py`.
4. Use `--provider name` or set the provider in scenario YAML.
