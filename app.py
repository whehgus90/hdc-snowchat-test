# app.py — Streamlit Cloud Ready (PAT + REST + SSE + Persistent history)
# 필요한 패키지: streamlit, requests, pandas
# secrets.toml 예시:
# [snowflake]
# account_base = "https://qnehhfk-rub23142.snowflakecomputing.com"   # Snowsight(regionless)
# sql_base     = "https://hgb46705.us-west-2.aws.snowflakecomputing.com"  # locator+region+cloud
# pat          = "<YOUR_PAT>"
# role         = "SALES_INTELLIGENCE_RL"
# warehouse    = "SALES_INTELLIGENCE_WH"
# database     = "SALES_INTELLIGENCE"
# schema       = "DATA"

import json, re, requests, pandas as pd, streamlit as st

# ---------------------------------------
# App config
# ---------------------------------------
st.set_page_config(page_title="❄️ SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (Cloud)")
st.caption("Cortex Search for calls · Analyst for metrics · Persistent chat (PAT + REST)")

# ---------------------------------------
# Secrets
# ---------------------------------------
SF = st.secrets["snowflake"]

ACCOUNT_BASE = SF["account_base"].rstrip("/")    # e.g. https://qnehhfk-rub23142.snowflakecomputing.com (regionless)
SQL_BASE     = SF["sql_base"].rstrip("/")        # e.g. https://hgb46705.us-west-2.aws.snowflakecomputing.com (regional)
PAT          = SF["pat"]

ROLE      = SF.get("role",      "SALES_INTELLIGENCE_RL")
WAREHOUSE = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE  = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA    = SF.get("schema",    "DATA")

# Always-on behaviors
AUTO_RUN_SQL = True
AUTO_QUALIFY = True

# Resources
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"

# Endpoints
AGENT_ENDPOINT = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"
SQL_ENDPOINT   = f"{SQL_BASE}/api/statements/v2/statements"

# ---------------------------------------
# Model picker (region에서 사용가능한 모델로)
# ---------------------------------------
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# New Conversation
with st.sidebar:
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------
# Utils: qualify SQL (FQN 보정)
# ---------------------------------------
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
    return s.strip()

# ---------------------------------------
# Headers / Payload builders
# ---------------------------------------
def agent_headers():
    # PAT + 컨텍스트(ASCII만)
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

def sql_headers():
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

def build_agent_payload(user_text: str, max_results: int = 5) -> dict:
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

# ---------------------------------------
# Agents REST (JSON or SSE)
# ---------------------------------------
def parse_sse_events(body_text: str) -> list:
    events, cur_event, data_lines = [], None, []
    for line in (body_text or "").splitlines() + [""]:
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

def call_agents(payload: dict, timeout_s: int = 90):
    r = requests.post(AGENT_ENDPOINT, headers=agent_headers(), json=payload, timeout=timeout_s)
    ctype = r.headers.get("Content-Type", "")
    st.write(f"DEBUG: HTTP {r.status_code}, Content-Type={ctype}")
    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None
    if "application/json" in ctype:
        try:
            return r.json()
        except Exception:
            st.error("JSON parse failed; raw body:")
            st.code(r.text[:2000], language="json")
            return None
    if "text/event-stream" in ctype or r.text.startswith("event:") or "\ndata:" in r.text:
        return parse_sse_events(r.text)
    st.error("Unknown response type; raw body:")
    st.code(r.text[:2000] or "<empty>")
    return None

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
    for s in j.get("searchResults") or j.get("results") or []:
        if isinstance(s, dict):
            citations.append({"source_id": s.get("source_id",""),
                              "doc_id":    s.get("doc_id","") or s.get("id","")})

def parse_json(obj: dict):
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
                tval = item.get("text")
                if isinstance(tval,str): text_parts.append(tval)
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

def parse_events(events: list):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v):
        if v and isinstance(v,str) and v.strip() and not sql_holder["v"]:
            sql_holder["v"] = v
    for ev in events:
        if not isinstance(ev, dict): continue
        if ev.get("event") == "message.delta":
            delta = ev.get("data",{})
            content = delta.get("delta",{}).get("content") or []
            if isinstance(content, dict): content = [content]
            for item in content:
                if item.get("type") == "text":
                    tval = item.get("text")
                    if isinstance(tval,str): text_parts.append(tval)
                elif item.get("type") == "tool_results":
                    for r in item.get("tool_results",{}).get("content",[]):
                        _pull_from_tool_result(r, citations, set_sql)
    text = _normalize_text("".join(text_parts))
    return text, sql_holder["v"], citations

def parse_any(resp):
    if resp is None: return "", None, []
    if isinstance(resp, dict):  return parse_json(resp)
    if isinstance(resp, list):  return parse_events(resp)
    return "", None, []

# ---------------------------------------
# SQL REST (Statements v2, sync)
# ---------------------------------------
def run_sql_rest(sql: str, timeout_s: int = 60) -> pd.DataFrame | None:
    try:
        q = qualify_sql(sql) if AUTO_QUALIFY else sql
        body = {
            "statement": q,
            "timeout": timeout_s,
            "async": False,
            "sequence_id": 0,
            "parameters": {},
        }
        r = requests.post(SQL_ENDPOINT, headers=sql_headers(), params={"async": "false"}, json=body, timeout=timeout_s+15)
        if r.status_code != 200:
            st.error(f"SQL HTTP {r.status_code} - {r.reason}")
            st.code(r.text[:2000] or "<empty>", language="json")
            return None
        data = r.json()
        # 표준 응답: resultSetMetaData + data
        meta = data.get("resultSetMetaData", {})
        cols = [c.get("name","COL") for c in meta.get("rowType",[])]
        rows = data.get("data", [])
        # rows는 [["val1","val2",...], ...] 형태일 수 있음
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        return df
    except requests.exceptions.SSLError as e:
        st.error("❌ SSL error talking to SQL endpoint. Check secrets['snowflake']['sql_base'] "
                 "is the **locator.region.cloud** host (e.g. hgb46705.us-west-2.aws).")
        st.code(str(e)[:1000])
        return None
    except Exception as e:
        st.error(f"SQL REST error: {e}")
        return None

# ---------------------------------------
# Debug panel: endpoints & context
# ---------------------------------------
with st.expander("Endpoints (debug)", expanded=False):
    st.code(f"AGENT_ENDPOINT = {AGENT_ENDPOINT}\nSQL_ENDPOINT   = {SQL_ENDPOINT}")

# ---------------------------------------
# Persistent chat state
# ---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history (answers + tables + expanders 보존)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("content",""))
        for t in m.get("tables", []):
            st.write(f"### {t.get('title','Query Result')}")
            st.dataframe(pd.DataFrame(t["data"]), use_container_width=True)
        for ex in m.get("expanders", []):
            with st.expander(ex.get("header","Details")):
                st.write(ex.get("body",""))

# ---------------------------------------
# Input
# ---------------------------------------
query = st.chat_input("Ask anything. (ex) Tell me about the call with SecureBank?  /  How many deals did Sarah Johnson win vs lost?)")
if not query:
    st.stop()

# Store user message
st.session_state.messages.append({"role":"user","content":query})
with st.chat_message("user"):
    st.markdown(query)

# Call Agents
with st.spinner("Calling Cortex Agents..."):
    resp = call_agents(build_agent_payload(query, max_results=5), timeout_s=90)

text, sql, citations = parse_any(resp)

# Build assistant message
assistant_chunks, tables_to_persist, expanders_to_persist = [], [], []

if text:
    assistant_chunks.append(text)

# Generated SQL → run immediately via SQL REST
if sql:
    assistant_chunks.append("### Generated SQL\n```sql\n" + sql.strip() + "\n```")
    if AUTO_RUN_SQL:
        df = run_sql_rest(sql, timeout_s=60)
        if df is not None:
            st.write("### Query Result")
            st.dataframe(df, use_container_width=True)
            # persist
            df_safe = df.astype(str)
            tables_to_persist.append({
                "title": "Query Result",
                "data": df_safe.to_dict(orient="records")
            })
            assistant_chunks.append(f"_Query returned **{len(df)}** row(s)._")

# Citations → show IDs + expand previews (via SQL REST)
if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        st.markdown("### Citations")
        st.markdown("**IDs:** " + ", ".join(f"`{i}`" for i in ids))

        # 미리보기 (상위 3개만)
        for doc_id in ids[:3]:
            preview_sql = f"""
            SELECT CONVERSATION_ID, CUSTOMER_NAME, SALES_REP, DEAL_STAGE,
                   CONVERSATION_DATE, DEAL_VALUE, PRODUCT_LINE, TRANSCRIPT_TEXT
            FROM SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS
            WHERE CONVERSATION_ID = '{doc_id.replace("'", "''")}'
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
assistant_msg   = {"role":"assistant","content":assistant_text,"tables":tables_to_persist,"expanders":expanders_to_persist}
st.session_state.messages.append(assistant_msg)

with st.chat_message("assistant"):
    st.markdown(assistant_text)
    for t in tables_to_persist:
        st.write(f"### {t.get('title','Query Result')}")
        st.dataframe(pd.DataFrame(t["data"]), use_container_width=True)
    for ex in expanders_to_persist:
        with st.expander(ex.get("header","Details")):
            st.write(ex.get("body",""))
