# app.py — Streamlit Cloud Ready (PAT + REST + SSE + Persistent History)
# 필요 패키지: streamlit, requests, pandas

import json, re, time, requests, pandas as pd, streamlit as st

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="❄️ SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (Cloud)")
st.caption("Cortex Search for calls · Analyst for metrics · Persistent chat (PAT only)")

# ---------------------------
# Secrets (Streamlit Cloud)
# ---------------------------
SF = st.secrets["snowflake"]

# Agents(에이전트)는 리전리스
ACCOUNT_BASE = SF["account_base"].rstrip("/")  # 예: https://qnehhfk-xxxxx.snowflakecomputing.com
PAT          = SF["pat"]                       # Programmatic Access Token (ASCII)

ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# SQL(Statements v2)은 "리전 전용" 호스트로
SQL_BASE     = SF.get("sql_base", "").rstrip("/")
if not SQL_BASE:
    # sql_base가 없으면 account_base를 폴백(권장하지 않음) — 가능한 secrets.toml에 sql_base 지정
    SQL_BASE = ACCOUNT_BASE

# 모델(리전에 맞게)
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# 리소스 FQN
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"

# REST Endpoint
AGENT_ENDPOINT = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"
SQL_ENDPOINT   = f"{SQL_BASE}/api/statements/v2/statements"

# 옵션: 디버그 로그
show_debug = st.sidebar.checkbox("Show debug logs", value=False)

# ---------------------------
# FQN 보정 (항상 적용)
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

# ---------------------------
# 라우팅 휴리스틱 (Search ↔ SQL)
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
# 텍스트 정리
# ---------------------------
def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n")
    # "띄어쓰기 대신 줄바꿈" 현상 간단 교정(연속 단어에 개행만 섞였을 때)
    s = re.sub(r"(?<=\w)\n(?=\w)", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 각종 특수 인용괄호 정리
    s = s.replace("【†", "[").replace("†】", "]")
    return s.strip()

# ---------------------------
# REST 공통 헤더
# ---------------------------
def _agent_headers():
    return {
        "Authorization": f"Bearer {PAT}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "X-Snowflake-Role": ROLE,
        "X-Snowflake-Database": DATABASE,
        "X-Snowflake-Schema": SCHEMA,
        "X-Snowflake-Warehouse": WAREHOUSE,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

def _sql_headers():
    return {
        "Authorization": f"Bearer {PAT}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "X-Snowflake-Role": ROLE,
        "X-Snowflake-Database": DATABASE,
        "X-Snowflake-Schema": SCHEMA,
        "X-Snowflake-Warehouse": WAREHOUSE,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

# ---------------------------
# Agents Payload
# ---------------------------
def build_payload(user_text: str, max_results: int = 5) -> dict:
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
                    "For metrics/counts/aggregates, use cortex_analyst_text_to_sql to generate SQL. "
                    "Always include a short natural language answer.")

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user",   "content": [{"type": "text", "text": user_text}]}
        ],
        "tools": tools,
        "tool_resources": tool_resources,
    }

# ---------------------------
# Agents 호출 (JSON or SSE)
# ---------------------------
def call_agents(payload: dict, timeout: int = 90):
    r = requests.post(AGENT_ENDPOINT, headers=_agent_headers(), json=payload, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    if show_debug:
        st.write(f"DEBUG: HTTP {r.status_code}, Content-Type={ctype}")

    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None

    # JSON
    if "application/json" in ctype:
        try:
            return r.json()
        except Exception:
            st.error("JSON parse error")
            st.code(r.text[:2000], language="json")
            return None

    # SSE
    body = r.text or ""
    if "text/event-stream" in ctype or body.startswith("event:") or "\ndata:" in body:
        events, cur_event, data_lines = [], None, []
        for line in (body.splitlines() + [""]):
            if line.startswith("event:"):
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try:
                        data_json = json.loads(data_str) if data_str else {}
                    except Exception:
                        data_json = {"raw": data_str}
                    events.append({"event": cur_event, "data": data_json})
                cur_event = line.split("event:", 1)[1].strip()
                data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.strip() == "":
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try:
                        data_json = json.loads(data_str) if data_str else {}
                    except Exception:
                        data_json = {"raw": data_str}
                    events.append({"event": cur_event, "data": data_json})
                    cur_event, data_lines = None, []
        return events

    # Unknown
    st.error("Unknown response format")
    st.code(r.text[:2000] or "<empty>")
    return None

# ---------------------------
# Agents 응답 파싱
# ---------------------------
def _pull_from_tool_result(result_obj, citations, set_sql_fn):
    if not isinstance(result_obj, dict): return
    j = result_obj.get("json")
    if isinstance(j, str):
        try: j = json.loads(j)
        except: j = None
    if not isinstance(j, dict): return
    for k in ("sql","generated_sql","sql_query"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            set_sql_fn(v); break
    # citations
    srs = j.get("searchResults") or j.get("results") or []
    if isinstance(srs, list):
        for s in srs:
            if isinstance(s, dict):
                citations.append({
                    "source_id": s.get("source_id",""),
                    "doc_id":    s.get("doc_id","") or s.get("id","")
                })

def parse_json_response(obj: dict):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v): 
        if v and isinstance(v,str) and v.strip(): sql_holder["v"] = v
    def pull(content):
        if isinstance(content, dict): content = [content]
        if not isinstance(content, list): return
        for item in content:
            t = item.get("type")
            if t == "text":
                txt = item.get("text")
                if isinstance(txt,str): text_parts.append(txt)
            elif t == "tool_results":
                for r in item.get("tool_results",{}).get("content",[]):
                    _pull_from_tool_result(r, citations, set_sql)

    for key in ("output","message","response","data"):
        node = obj.get(key)
        if isinstance(node, dict) and "content" in node:
            pull(node["content"])
    if not text_parts and "content" in obj:
        pull(obj["content"])

    text = _normalize_text("".join(text_parts))
    return text, sql_holder["v"], citations

def parse_events_response(events: list):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v):
        if v and isinstance(v,str) and v.strip() and not sql_holder["v"]:
            sql_holder["v"] = v
    for ev in events or []:
        if not isinstance(ev, dict): continue
        if ev.get("event") == "message.delta":
            delta = ev.get("data",{}).get("delta",{})
            content = delta.get("content") or []
            if isinstance(content, dict): content = [content]
            for item in content:
                if item.get("type") == "text":
                    t = item.get("text")
                    if isinstance(t,str): text_parts.append(t)
                elif item.get("type") == "tool_results":
                    for r in item.get("tool_results",{}).get("content",[]):
                        _pull_from_tool_result(r, citations, set_sql)
    text = _normalize_text("".join(text_parts))
    return text, sql_holder["v"], citations

def parse_any(resp):
    if resp is None: return "", None, []
    if isinstance(resp, dict):  return parse_json_response(resp)
    if isinstance(resp, list):  return parse_events_response(resp)
    if isinstance(resp, str) and resp.startswith("event:"):
        # 간단 SSE 문자열 파싱
        lines = resp.splitlines()
        events, cur_event, data_lines = [], None, []
        for line in (lines + [""]):
            if line.startswith("event:"):
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try:
                        data_json = json.loads(data_str) if data_str else {}
                    except Exception:
                        data_json = {"raw": data_str}
                    events.append({"event": cur_event, "data": data_json})
                cur_event = line.split("event:",1)[1].strip()
                data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.strip() == "":
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try:
                        data_json = json.loads(data_str) if data_str else {}
                    except Exception:
                        data_json = {"raw": data_str}
                    events.append({"event": cur_event, "data": data_json})
                    cur_event, data_lines = None, []
        return parse_events_response(events)
    return "", None, []

# ---------------------------
# SQL 실행 (Statements v2, PAT)
# ---------------------------
def _extract_df(stmt_resp: dict) -> pd.DataFrame | None:
    data = stmt_resp.get("result", {}).get("data")
    cols = stmt_resp.get("result", {}).get("resultSetMetaData", {}).get("rowType", [])
    if not data or not cols:
        return None
    colnames = [c.get("name") for c in cols]
    rows = [[cell.get("value") for cell in row] for row in data]
    return pd.DataFrame(rows, columns=colnames)

def run_sql_rest(sql: str, timeout_s: int = 120) -> pd.DataFrame | None:
    if not sql:
        return None
    stmt = qualify_sql(sql)

    # 우선 동기 실행 시도
    params = {"async": "false"}
    body = {
        "statement": stmt,
        "timeout": timeout_s,
        "resultSetMetaData": {"format": "json"},
        "warehouse": WAREHOUSE,
        "role": ROLE,
        "database": DATABASE,
        "schema": SCHEMA,
    }
    r = requests.post(SQL_ENDPOINT, headers=_sql_headers(), params=params, json=body, timeout=timeout_s+15)
    if r.status_code != 200:
        st.error(f"SQL HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None

    try:
        resp = r.json()
    except Exception:
        st.error("SQL response parse error")
        st.code(r.text[:2000], language="json")
        return None

    status = (resp.get("status") or "").lower()
    if status in ("success","succeeded","complete"):
        return _extract_df(resp)
    if status in ("failed","failed_with_error","aborted","cancelled"):
        err = resp.get("message") or resp.get("errorMessage") or resp.get("code") or json.dumps(resp)[:500]
        st.error(f"SQL failed: {err}")
        st.code(stmt, language="sql")
        return None

    # async=false 무시 → 폴링
    handle = resp.get("statementHandle")
    if not handle:
        st.error("SQL: no result and no handle returned.")
        st.code(json.dumps(resp)[:1500], language="json")
        return None

    get_url = f"{SQL_ENDPOINT}/{handle}"
    last_status = status
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        g = requests.get(get_url, headers=_sql_headers(), timeout=30)
        if g.status_code != 200:
            st.error(f"SQL poll HTTP {g.status_code} - {g.reason}")
            st.code(g.text[:2000] or "<empty>", language="json")
            return None
        pj = g.json()
        stt = (pj.get("status") or "").lower()
        if stt != last_status and show_debug:
            st.write(f"SQL status: {stt}")
            last_status = stt
        if stt in ("success","succeeded","complete"):
            return _extract_df(pj)
        if stt in ("failed","failed_with_error","aborted","cancelled"):
            err = pj.get("message") or pj.get("errorMessage") or pj.get("code") or json.dumps(pj)[:500]
            st.error(f"SQL failed: {err}")
            st.code(stmt, language="sql")
            return None
        time.sleep(1)

    st.error(f"SQL poll timeout (no completion within {timeout_s}s)")
    st.code(stmt, language="sql")
    return None

# ---------------------------
# Chat state (히스토리 + 테이블/인용 보존)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 히스토리 렌더링
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
# 입력 & 호출
# ---------------------------
query = st.chat_input("Ask anything. (ex) Tell me about the call with SecureBank?  /  How many deals did Sarah Johnson win vs lost?)")
if not query:
    st.stop()

# 사용자 메시지 저장/표시
st.session_state.messages.append({"role":"user","content":query})
with st.chat_message("user"):
    st.markdown(query)

# 에이전트 호출
with st.spinner("Calling Cortex Agents..."):
    resp = call_agents(build_payload(query, max_results=5), timeout=90)

text, sql, citations = parse_any(resp)

# 어시스턴트 메시지 구성 + 첨부들(테이블/expander) 저장
assistant_chunks, tables_to_persist, expanders_to_persist = [], [], []

if show_debug and isinstance(resp, list):
    st.write("DEBUG: Received SSE events")

if text:
    assistant_chunks.append(text)

if sql:
    assistant_chunks.append("### Generated SQL\n```sql\n" + sql.strip() + "\n```")
    # 항상 SQL 실행
    df = run_sql_rest(sql, timeout_s=120)
    if df is not None:
        st.write("### Query Result")
        st.dataframe(df, use_container_width=True)
        # 히스토리에 저장(문자형 안전 변환)
        df_safe = df.astype(str)
        tables_to_persist.append({
            "title": "Query Result",
            "data": df_safe.to_dict(orient="records")
        })
        assistant_chunks.append(f"_Query returned **{len(df)}** row(s)._")

# Citations → transcript 미리보기 (REST SQL 사용)
if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        assistant_chunks.append("**Citations:** " + ", ".join(f"`{i}`" for i in ids))
        st.markdown("### Citations")
        st.markdown("IDs: " + ", ".join(f"{i}" for i in ids))

    for doc_id in ids:
        esc = str(doc_id).replace("'", "''")
        preview_sql = f"""
        SELECT CONVERSATION_ID, CUSTOMER_NAME, SALES_REP, DEAL_STAGE,
               CONVERSATION_DATE, DEAL_VALUE, PRODUCT_LINE, TRANSCRIPT_TEXT
        FROM SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS
        WHERE CONVERSATION_ID = '{esc}'
        """
        dfp = run_sql_rest(preview_sql, timeout_s=60)
        if dfp is None or dfp.empty:
            continue
        row = dfp.iloc[0].to_dict()
        header = f"[{row.get('CONVERSATION_ID','')}] {row.get('CUSTOMER_NAME','')} · {row.get('SALES_REP','')} · {row.get('DEAL_STAGE','')} · {row.get('CONVERSATION_DATE','')}"
        body_text = row.get("TRANSCRIPT_TEXT","(no transcript)")
        with st.expander(header):
            st.write(body_text)
        expanders_to_persist.append({"header": header, "body": body_text})

assistant_text = "\n\n".join(assistant_chunks).strip() if assistant_chunks else "_No answer returned._"
assistant_msg = {
    "role":"assistant",
    "content":assistant_text,
    "tables":tables_to_persist,
    "expanders":expanders_to_persist
}
st.session_state.messages.append(assistant_msg)

with st.chat_message("assistant"):
    st.markdown(assistant_text)
    for t in tables_to_persist:
        st.write(f"### {t.get('title','Query Result')}")
        st.dataframe(pd.DataFrame(t["data"]), use_container_width=True)
    for ex in expanders_to_persist:
        with st.expander(ex.get("header","Details")):
            st.write(ex.get("body",""))
