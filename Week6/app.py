from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json, os
from pathlib import Path
from sympy import sympify
import re, requests

BASE_DIR = Path(__file__).resolve().parent
KEY_PATH = Path(os.getenv("OPENAI_KEY_PATH", BASE_DIR / "secrets" / "openai_key.txt"))

def load_api_key(path: Path) -> str:
    try:
        key = path.read_text(encoding="utf-8").strip()
        if not key or not key.startswith("sk-"):
            raise ValueError("Key file is empty or not an OpenAI key.")
        return key
    except FileNotFoundError:
        raise RuntimeError(f"API key file not found at: {path}. "
                           f"Create it and paste your key (single line).")

OPENAI_API_KEY = load_api_key(KEY_PATH)
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Tools ---
def search_arxiv(query: str) -> str:
    url = 'http://export.arxiv.org/api/query'
    params = {'search_query': f'all:{query}', 'start': 0, 'max_results': 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        text = r.text
        title_m = re.search(r'<title>(.*?)</title>', text, re.S)
        summ_m  = re.search(r'<summary>(.*?)</summary>', text, re.S)
        title = title_m.group(1).strip() if title_m else 'arXiv result'
        import re as _re
        summary = _re.sub(r'\s+', ' ', (summ_m.group(1) if summ_m else '')).strip()
        return f"{title}: {summary[:400]}"
    except Exception as e:
        return f"(arXiv lookup failed: {e})"

def calculate(expression: str) -> str:
    try:
        return str(sympify(expression).evalf())
    except Exception as e:
        return f"(calc error: {e})"

TOOLS = {'search_arxiv': search_arxiv, 'calculate': calculate}
MODEL = 'gpt-4o-mini'

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    return s

def llm_decide(user_text: str):
    SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that can call tools. "
    "For ANY arithmetic, numeric conversion, percentage, unit math, or expression "
    "containing digits with + - * / ^ % ( ), you MUST call the tool "
    "\"calculate\" and you are NOT allowed to compute the result yourself. "
    "Return STRICT one-line JSON:\n"
    '- Tool call: {"action":"call_tool","tool":"<tool_name>","args":{...}}\n'
    '- Direct answer (non-math only): {"action":"final","answer":"<text>"}\n'
    "Available tools:\n"
    "1) search_arxiv(query: str)\n"
    "2) calculate(expression: str)\n"
    )
    prompt = SYSTEM_INSTRUCTION + '\n\nUSER: ' + user_text + '\n\nRespond with STRICT one-line JSON as specified.'
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{'role':'user','content': prompt}],
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    raw_json = _strip_code_fences(raw)
    try:
        data = json.loads(raw_json)
    except Exception:
        data = {'action':'final','answer': raw}
    return {'raw': raw, 'data': data}

def route_llm_output(decision):
    data = decision['data']
    trace = {'llm_raw': decision['raw'], 'tool_call': None, 'tool_result': None, 'final': None}
    if isinstance(data, dict) and data.get('action') == 'call_tool':
        tool = data.get('tool'); args = data.get('args', {}) or {}
        fn = TOOLS.get(tool)
        if fn is None:
            trace['final'] = f'(unknown tool: {tool})'
            return trace
        try:
            result = fn(**args)
        except TypeError:
            result = fn(*list(args.values())) if args else fn('')
        trace['tool_call'] = {'tool': tool, 'args': args}
        trace['tool_result'] = result
        compose = client.chat.completions.create(
            model=MODEL,
            messages=[{'role':'system','content':'Write a short spoken-style reply summarizing the tool result.'},
                     {'role':'user','content': f'User asked: {args}\nTool: {tool}\nResult: {result}'}],
            temperature=0.3,
        )
        trace['final'] = compose.choices[0].message.content.strip()
        return trace
    trace['final'] = data.get('answer') if isinstance(data, dict) else str(data)
    return trace

app = FastAPI(title='Voice Agent with Function Calling')
class VoiceQuery(BaseModel):
    text: str
@app.post('/api/voice-query')
def voice_query(q: VoiceQuery):
    decision = llm_decide(q.text)
    trace = route_llm_output(decision)
    return {'query': q.text, 'llm_raw': trace['llm_raw'], 'tool_call': trace['tool_call'], 'tool_result': trace['tool_result'], 'final': trace['final']}
