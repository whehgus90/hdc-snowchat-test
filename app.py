# app.py — Streamlit Cloud (PAT only) · Force JSON, Robust SSE, SQL via REST, Persistent chat

import json, re, time, requests, pandas as pd, streamlit as st

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="❄️ SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (Cloud / PAT only)")
st.caption("Cortex Search (calls) · Analyst (metrics) · SQL via REST · Persistent chat")

# ---------------------------
# Secrets (Streamlit Cloud -> secrets.toml)
# ---------------------------
SF = st.secrets["snowflake"]
ACCOUNT_BASE = SF["account_base"].rstrip("/")
PAT          = SF["pat"]
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# 모델
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# 항상 적용
AUTO_RUN_SQL = True
AUTO_QUALIFY = True

# 리소스
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"
AGENTS_ENDPOINT       = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"
SQL_ENDPOINT          = f"{ACCOUNT_BASE}/api/v2/statements"

# ---------------------------
# Helpers
# ---------------------------
_TABLE_FQN = {
    "SALES_CONVERSATIONS": "SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS",
    "SALES_METRICS":       "SALES_INTELLIGENCE.DATA.SALES_METRICS",
}
def qualify_sql(sql: str) -> str:
    s = (sql or "").strip().rstrip(";")
    for short, fqn in _TABLE_FQN.items():
        s = re.sub(rf"(?<!\.)\b{short}\b", fqn, s, flags=re.IGNORECASE)
    return s

def _normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.replace("【†","[").replace("†】","]").strip()

def _headers_common():
    return {
        "Authorization": f"Bearer {PAT}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "X-Snowflake-Role": ROLE,
        "X-Snowflake-Database": DATABASE,
        "X-Snowflake-Schema": SCHEMA,
        "X-Snowflake-Warehouse": WAREHOUSE,
    }

# ---------------------------
# Intent routing
# ---------------------------
_SEARCH_HINTS = {
    "call","calls","conversation","conversations","transcript","transcripts",
    "meeting","meetings","qbr","summary","summarize","tell me about","what did"
}
_SQL_HINTS = {
    "how many","count","sum","avg","average","total","compare","vs","trend","won","lost",
    "revenue","value","pipeline","close rate","ratio","by ","group by","top ","rank","percent"
}
def detect_intent(q: str) -> str:
    ql = (q or "").lower()
    if any(k in ql for k in _SEARCH_HINTS) and not any(k in ql for k in _SQL_HINTS):
        return "Search"
    if any(k in ql for k in _SQL_HINTS):
        return "SQL"
    return "Auto"

# ---------------------------
# Agents payload (JSON 강제)
# ---------------------------
def build_agents_payload(user_text: str, max_results: int = 5) -> dict:
    intent = detect_intent(user_text)
    use_search = intent in ("Auto", "Search")
    use_sql    = intent in ("Auto", "SQL")

    tools, tool_resources = [], {}
    if use_search:
        tools.append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
        tool_resources["search1"] = {
            "name": CORTEX_SEARCH_SERVICE,
            "max_results": max_results,
            "id_column": "conversation_id",
        }
    if use_sql:
        tools.append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
        tool_resources["analyst1"] = {"semantic_model_file": SEMANTIC_MODEL_FILE}

    if intent == "Search":
        sys_text = ("You are a helpful sales conversation assistant. "
                    "Prefer cortex_search to find relevant call transcripts and summarize them. "
                    "Return a concise natural language summary and include brief citations.")
    elif intent == "SQL":
        sys_text = ("You are a helpful SQL analyst. "
                    "Prefer cortex_analyst_text_to_sql for metrics/aggregations. "
                    "Also return a short natural language answer summarizing the result.")
    else:
        sys_text = ("You are a helpful assistant. "
                    "For conversations/calls/transcripts, use cortex_search to retrieve and summarize. "
                    "For metrics/aggregations, use cortex_analyst_text_to_sql to generate SQL. "
                    "Always include a short natural language answer.")

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user",   "content": [{"type": "text", "text": user_text}]}
        ],
        "tools": tools,
        "tool_resources": tool_resources,
        "stream": False,   # JSON 응답 강제
    }

# ---------------------------
# Agents call (JSON 고집 + SSE 완전대응 + Raw fallback)
# ---------------------------
def call_agents(payload: dict, timeout: int = 90):
    headers = {**_headers_common(),
               # ★여기 핵심: JSON만 명시해서 SSE 강제 방지
               "Accept": "application/json",
               "Content-Type": "application/json"}
    # ★그리고 혹시 모를 서버 기본값 방지로 stream=false 쿼리도 동시 전달
    r = requests.post(AGENTS_ENDPOINT, headers=headers, params={"stream":"false"}, json=payload, timeout=timeout)
    ctype = r.headers.get("Content-Type","")
    st.write(f"DEBUG: HTTP {r.status_code}, Content-Type={ctype}")
    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None

    # 1) JSON이면 그대로
    if "application/json" in ctype:
        try:
            return r.json()
        except Exception:
            st.error("JSON parse failed"); st.code(r.text[:2000], language="json")
            return None

    # 2) 혹시 여전히 SSE로 내려왔다면 — 초강력 파서 + Raw fallback
    body = r.content.decode("utf-8", errors="replace")
    if "text/event-stream" in ctype or body.startswith("event:") or "\ndata:" in body:
        events = _parse_sse_to_events(body)
        if not events:
            # Raw 보여주기라도 하자
            st.error("SSE body could not be parsed; showing raw snippet.")
            st.code(body[:2000], language="text")
            return None
        return events

    # 3) 알 수 없는 응답 → 원문 노출
    st.error("Unknown response type; showing raw snippet.")
    st.code((r.text or "")[:2000], language="text")
    return None

def _parse_sse_to_events(body_text: str) -> list:
    events, cur_event, data_lines = [], None, []
    for line in body_text.splitlines() + [""]:
        if line.startswith("event:"):
            if cur_event is not None:
                _flush_event(events, cur_event, data_lines)
            cur_event, data_lines = line.split("event:",1)[1].strip(), []
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
        elif line.strip() == "" and cur_event is not None:
            _flush_event(events, cur_event, data_lines)
            cur_event, data_lines = None, []
    return events

def _flush_event(events, cur_event, data_lines):
    data_str = "\n".join(data_lines).strip()
    try:
        data_json = json.loads(data_str) if data_str else {}
    except Exception:
        data_json = {"raw": data_str}
    # 다양한 이벤트명을 message.delta로 표준화
    name = cur_event.lower()
    if name.startswith("message"):
        events.append({"event": "message.delta", "data": data_json})
    elif name.startswith("output_text"):
        txt = data_json.get("text") or data_json.get("delta",{}).get("text")
        events.append({"event": "message.delta", "data": {"delta": {"content":[{"type":"text","text": txt or ""}]}}})
    else:
        events.append({"event": "message.delta", "data": {"delta": data_json}})

# ---------------------------
# Parse (JSON + SSE + 초강력 스캐너)
# ---------------------------
def _walk_any_text(obj, bag):
    """JSON 어디에 있든 'text'류 문자열 긁어모으기"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text","answer","output","summary") and isinstance(v, str):
                bag.append(v)
            elif k == "content":
                if isinstance(v, list):
                    for it in v:
                        _walk_any_text(it, bag)
                elif isinstance(v, dict):
                    _walk_any_text(v, bag)
            else:
                _walk_any_text(v, bag)
    elif isinstance(obj, list):
        for it in obj:
            _walk_any_text(it, bag)

def _pull_from_tool_result(result_obj, citations, set_sql_fn, text_parts):
    if not isinstance(result_obj, dict): return
    j = result_obj.get("json")
    if isinstance(j, str):
        try: j = json.loads(j)
        except: j = None
    if not isinstance(j, dict): return
    # SQL
    for k in ("sql","generated_sql","sql_query"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            set_sql_fn(v); break
    # Citations
    for s in j.get("searchResults") or j.get("results") or []:
        if isinstance(s, dict):
            citations.append({
                "source_id": s.get("source_id",""),
                "doc_id":    s.get("doc_id","") or s.get("id","")
            })
    # JSON 내부 텍스트도 긁기
    _walk_any_text(j, text_parts)

def parse_json_response(obj: dict):
    text_parts, citations, sql_holder = [], [], {"v": None}
    def set_sql(v): 
        if v and isinstance(v,str) and v.strip(): sql_holder["v"] = v
    def pull(content):
        if isinstance(content, dict): content = [content]
        if not isinstance(content, list): return
        for item in content:
            t = item.get("type")
            if t == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif t == "tool_results":
                for r in item.get("tool_results",{}).get("content",[]):
                    _pull_from_tool_result(r, citations, set_sql, text_parts)
            else:
                _walk_any_text(item, text_parts)
    for key in ("output","message","response","data","result"):
        node = obj.get(key)
        if isinstance(node, dict) and "content" in node:
            pull(node["content"])
        elif node is not None:
            _walk_any_text(node, text_parts)
    if not text_parts and "content" in obj:
        pull(obj["content"])
    return _normalize_text(" ".join(text_parts)), sql_holder["v"], citations

def parse_events_response(events: list):
    text_parts, citations, sql_holder = [], [], {"v": None}
    def set_sql(v):
        if v and isinstance(v,str) and v.strip() and not sql_holder["v"]:
            sql_holder["v"] = v
    for ev in events:
        data = ev.get("data",{})
        delta = data.get("delta", data)
        content = (delta.get("content") if isinstance(delta, dict) else None) or []
        if isinstance(content, dict): content = [content]
        if content:
            for item in content:
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif item.get("type") == "tool_results":
                    for r in item.get("tool_results",{}).get("content",[]):
                        _pull_from_tool_result(r, citations, set_sql, text_parts)
                else:
                    _walk_any_text(item, text_parts)
        else:
            _walk_any_text(delta, text_parts)
    return _normalize_text(" ".join(text_parts)), sql_holder["v"], citations

def parse_any(resp):
    if resp is None: return "", None, []
    if isinstance(resp, dict):  return parse_json_response(resp)
    if isinstance(resp, list):  return parse_events_response(resp)
    return "", None, []

# ---------------------------
# SQL via REST
# ---------------------------
def _sql_headers():
    return {**_headers_common(), "Accept":"application/json", "Content-Type":"application/json"}

def _extract_df(payload: dict) -> pd.DataFrame:
    if "resultSet" in payload:
        rt = payload["resultSet"].get("rowType", [])
        rows = payload["resultSet"].get("rows", [])
        if rt and rows and isinstance(rows[0], list):
            cols = [c["name"] for c in rt]
            return pd.DataFrame(rows, columns=cols)
    if "data" in payload and isinstance(payload["data"], dict):
        rows = payload["data"].get("rowset", [])
        cols  = [c["name"] for c in payload["data"].get("rowType", [])]
        return pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)
    return pd.DataFrame()

def run_sql_rest(sql: str, timeout_s: int = 90) -> pd.DataFrame | None:
    if not sql: return None
    stmt = qualify_sql(sql) if AUTO_QUALIFY else sql
    body = {
        "statement": stmt,
        "timeout": timeout_s,
        "resultSetMetaData": { "format": "json" },
        "warehouse": WAREHOUSE,
        "role": ROLE,
        "database": DATABASE,
        "schema": SCHEMA
    }
    r = requests.post(SQL_ENDPOINT, headers=_sql_headers(), json=body, timeout=timeout_s)
    if r.status_code != 200:
        st.error(f"SQL HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None
    resp = r.json()
    status = (resp.get("status") or "").lower()
    handle = resp.get("statementHandle")
    if status in ("success","succeeded","complete"):
        try: return _extract_df(resp)
        except Exception as e:
            st.error(f"Result parse error: {e}")
            st.code(json.dumps(resp)[:1500], language="json")
            return None
    get_url = f"{SQL_ENDPOINT}/{handle}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        g = requests.get(get_url, headers=_sql_headers(), timeout=timeout_s)
        if g.status_code != 200:
            st.error(f"SQL poll HTTP {g.status_code} - {g.reason}")
            st.code(g.text[:2000] or "<empty>", language="json")
            return None
        pj = g.json()
        stt = (pj.get("status") or "").lower()
        if stt in ("success","succeeded","complete"):
            try: return _extract_df(pj)
            except Exception as e:
                st.error(f"Result parse error: {e}")
                st.code(json.dumps(pj)[:1500], language="json")
                return None
        time.sleep(0.5)
    st.error("SQL poll timeout"); return None

def preview_transcript_rest(doc_id: str) -> tuple[str, str] | None:
    if not doc_id: return None
    did = doc_id.replace("'", "''")
    sql = f"""
    SELECT CONVERSATION_ID, CUSTOMER_NAME, SALES_REP, DEAL_STAGE,
           CONVERSATION_DATE, DEAL_VALUE, PRODUCT_LINE, TRANSCRIPT_TEXT
    FROM SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS
    WHERE CONVERSATION_ID = '{did}'
    """
    df = run_sql_rest(sql)
    if df is None or df.empty: return None
    row = df.iloc[0].to_dict()
    header = f"[{row.get('CONVERSATION_ID','')}] {row.get('CUSTOMER_NAME','')} · {row.get('SALES_REP','')} · {row.get('DEAL_STAGE','')} · {row.get('CONVERSATION_DATE','')}"
    return header, row.get("TRANSCRIPT_TEXT","(no transcript)")

# ---------------------------
# Chat state (persist)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("content",""))
        for t in m.get("tables", []):
            st.write(f"### {t.get('title','Query Result')}")
            st.dataframe(pd.DataFrame(t["data"]), use_container_width=True)
        for ex in m.get("expanders", []):
            with st.expander(ex.get("header","Details")):
                st.write(ex.get("body",""))

# ---------------------------
# Input & Call
# ---------------------------
query = st.chat_input("Ask anything. (ex) Tell me about the call with SecureBank?  /  How many deals did Sarah Johnson win vs lost?)")
if not query:
    st.stop()

st.session_state.messages.append({"role":"user","content":query})
with st.chat_message("user"):
    st.markdown(query)

with st.spinner("Calling Cortex Agents..."):
    resp = call_agents(build_agents_payload(query, max_results=5), timeout=90)

text, sql, citations = parse_any(resp)

assistant_chunks, tables_to_persist, expanders_to_persist = [], [], []

if text:
    assistant_chunks.append(text)

if sql:
    assistant_chunks.append("### Generated SQL\n```sql\n" + sql.strip() + "\n```")
    if AUTO_RUN_SQL:
        df = run_sql_rest(sql)
        if df is not None:
            st.write("### Query Result")
            st.dataframe(df, use_container_width=True)
            tables_to_persist.append({
                "title":"Query Result",
                "data": df.astype(str).to_dict(orient="records")
            })

if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        assistant_chunks.append("**Citations:** " + ", ".join(f"`{i}`" for i in ids))
        for doc_id in ids:
            pv = preview_transcript_rest(doc_id)
            if not pv: continue
            header, body = pv
            with st.expander(header):
                st.write(body)
            expanders_to_persist.append({"header": header, "body": body})

assistant_text = "\n\n".join(assistant_chunks).strip() if assistant_chunks else "_No answer returned._"
assistant_msg = {"role":"assistant","content":assistant_text,"tables":tables_to_persist,"expanders":expanders_to_persist}
st.session_state.messages.append(assistant_msg)

with st.chat_message("assistant"):
    st.markdown(assistant_text)
    for t in tables_to_persist:
        st.write(f"### {t.get('title','Query Result')}")
        st.dataframe(pd.DataFrame(t["data"]), use_container_width=True)
    for ex in expanders_to_persist:
        with st.expander(ex.get("header","Details")):
            st.write(ex.get("body",""))
