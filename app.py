# app.py — Streamlit Cloud Ready (PAT + REST + snowflake-connector)
# - Auto Route: Conversations(Search) ↔ Metrics(SQL)
# - Clean text, Always run SQL + FQN, Persistent chat (tables & citations)

import json, re, requests, pandas as pd, streamlit as st
from snowflake import connector as sf

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="❄️ SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (Cloud)")
st.caption("Cortex Search for calls · Analyst for metrics · Persistent chat")

# ---------------------------
# Secrets (Streamlit Cloud에 설정)
# ---------------------------
SF = st.secrets["snowflake"]

ACCOUNT_BASE = SF["account_base"].rstrip("/")  # https://<acct>.<region>.snowflakecomputing.com
PAT          = SF["pat"]                       # Programmatic Access Token
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# (SQL 실행용 커넥터 자격 — 선택/권장)
SF_USER      = SF.get("user")
SF_PASSWORD  = SF.get("password")
SF_ACCOUNT   = SF.get("account")   # 예: fv93338.ap-northeast-2.aws

# 모델 선택(리전에 맞게)
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# 항상 적용
AUTO_RUN_SQL = True
AUTO_QUALIFY = True

# 리소스 이름(FQN)
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"
API_ENDPOINT          = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"

with st.sidebar:
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# FQN 보정
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
# Snowflake SQL 실행(Connector)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    if not (SF_USER and SF_PASSWORD and SF_ACCOUNT):
        return None
    return sf.connect(
        user=SF_USER, password=SF_PASSWORD, account=SF_ACCOUNT,
        warehouse=WAREHOUSE, database=DATABASE, schema=SCHEMA, role=ROLE,
        session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
    )

def run_sql(sql: str) -> pd.DataFrame | None:
    conn = get_conn()
    if conn is None:
        st.info("🔐 secrets에 user/password/account가 없어 SQL 실행은 생략합니다.")
        return None
    try:
        q = qualify_sql(sql) if AUTO_QUALIFY else sql
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"❌ SQL 실행 에러: {e}")
        st.code(sql, language="sql")
        return None

# ---------------------------
# 라우팅 휴리스틱
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
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------------------------
# Agents REST 호출
# ---------------------------
def build_headers():
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

def call_agents_rest(payload: dict, timeout: int = 60):
    r = requests.post(API_ENDPOINT, headers=build_headers(), data=json.dumps(payload), timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} - {r.reason}\n{r.text[:2000]}")
    try:
        return r.json()
    except Exception:
        # 혹시 SSE 텍스트로 올 경우 대비 (드뭄)
        return json.loads(r.text)

# ---------------------------
# 응답 파싱
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
    for s in j.get("searchResults") or j.get("results") or []:
        if isinstance(s, dict):
            citations.append({"source_id": s.get("source_id",""),
                              "doc_id":    s.get("doc_id","") or s.get("id","")})

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

def parse_any(resp):
    if resp is None: return "", None, []
    if isinstance(resp, dict):  return parse_json_response(resp)
    if isinstance(resp, list):  # SSE 배열이 올 수도 있음
        # 간단 파서: message.delta 누적
        text_parts, citations, sql = [], [], None
        for ev in resp:
            if isinstance(ev, dict) and ev.get("event") == "message.delta":
                for item in ev.get("data",{}).get("delta",{}).get("content",[]):
                    if item.get("type") == "text":
                        t = item.get("text")
                        if isinstance(t,str): text_parts.append(t)
                    elif item.get("type") == "tool_results":
                        for r in item.get("tool_results",{}).get("content",[]):
                            def set_sql(v): nonlocal sql; sql = sql or v
                            _pull_from_tool_result(r, citations, set_sql)
        return _normalize_text("".join(text_parts)), sql, citations
    return "", None, []

# ---------------------------
# SEARCH_PREVIEW (Search fallback)
# ---------------------------
def preview_search(query: str, limit: int = 3):
    if not get_conn():
        return None
    esc = (query or "").replace("'", "''")
    sql = f"""
    SELECT PARSE_JSON(
      SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        '{CORTEX_SEARCH_SERVICE}',
        OBJECT_CONSTRUCT(
          'query', '{esc}',
          'columns', ARRAY_CONSTRUCT(
            'CONVERSATION_ID','CUSTOMER_NAME','DEAL_STAGE','SALES_REP',
            'CONVERSATION_DATE','DEAL_VALUE','PRODUCT_LINE','TRANSCRIPT_TEXT'
          ),
          'limit', {limit}
        )
      )
    ) AS J
    """
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        st.error(f"SEARCH_PREVIEW error: {e}")
        return None

# ---------------------------
# Chat state (persistent with attachments)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("content",""))
        for t in m.get("tables", []):
            try:
                st.write(f"### {t.get('title','Query Result')}")
                df_hist = pd.DataFrame(t["data"]) if isinstance(t.get("data"), list) else pd.DataFrame()
                st.dataframe(df_hist, use_container_width=True)
            except Exception:
                pass
        for ex in m.get("expanders", []):
            try:
                with st.expander(ex.get("header","Details")):
                    st.write(ex.get("body",""))
            except Exception:
                pass

# ---------------------------
# Chat input
# ---------------------------
query = st.chat_input("Ask anything. (ex) Tell me about the call with SecureBank?  /  How many deals did Sarah Johnson win vs lost?)")
if not query:
    st.stop()

# Store user msg
st.session_state.messages.append({"role":"user","content":query})
with st.chat_message("user"):
    st.markdown(query)

# Call Agents
with st.spinner("Calling Cortex Agents..."):
    try:
        body = call_agents_rest(build_payload(query, max_results=5))
    except Exception as e:
        st.error(str(e))
        st.stop()

text, sql, citations = parse_any(body)

# Build assistant message with attachments
assistant_chunks, tables_to_persist, expanders_to_persist = [], [], []

if text:
    assistant_chunks.append(text)

if sql:
    assistant_chunks.append("### Generated SQL\n```sql\n" + sql.strip() + "\n```")
    if AUTO_RUN_SQL:
        df = run_sql(sql)
        if df is not None:
            st.write("### Query Result")
            st.dataframe(df, use_container_width=True)
            df_safe = df.astype(str)
            tables_to_persist.append({
                "title": "Query Result",
                "data": df_safe.to_dict(orient="records")
            })
            assistant_chunks.append(f"_Query returned **{len(df)}** row(s)._")

if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        assistant_chunks.append("**Citations:** " + ", ".join(f"`{i}`" for i in ids))
        st.markdown("### Citations")
    for doc_id in ids:
        if not get_conn():  # 커넥터 없으면 전문 미리보기 생략
            continue
        preview_sql = f"""
        SELECT CONVERSATION_ID, CUSTOMER_NAME, SALES_REP, DEAL_STAGE, CONVERSATION_DATE, DEAL_VALUE, PRODUCT_LINE, TRANSCRIPT_TEXT
        FROM SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS
        WHERE CONVERSATION_ID = '{doc_id.replace("'", "''")}'
        """
        dfp = run_sql(preview_sql)
        if dfp is None or dfp.empty:
            continue
        row = dfp.iloc[0].to_dict()
        header = f"[{row.get('CONVERSATION_ID','')}] {row.get('CUSTOMER_NAME','')} · {row.get('SALES_REP','')} · {row.get('DEAL_STAGE','')} · {row.get('CONVERSATION_DATE','')}"
        body_text = row.get("TRANSCRIPT_TEXT","(no transcript)")
        with st.expander(header):
            st.write(body_text)
        expanders_to_persist.append({"header": header, "body": body_text})

# Fallbacks
intent = detect_intent(query)
if not text and not sql:
    if intent in ("Search","Auto") and get_conn():
        j = preview_search(query, limit=3)
        if j:
            assistant_chunks.append("### Top matches (SEARCH_PREVIEW)\n```json\n" + json.dumps(j, ensure_ascii=False, indent=2)[:2000] + "\n```")

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
