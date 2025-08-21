# app.py — Streamlit Cloud Ready (PAT + REST + SSE 파서 + 영구 히스토리 + 커넥터 상태)
# 필요 패키지: streamlit, requests, pandas, snowflake-connector-python

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
ACCOUNT_BASE = SF["account_base"].rstrip("/")  # https://<acct>.<region>.<cloud>.snowflakecomputing.com (regionless도 OK)
PAT          = SF["pat"]                       # Programmatic Access Token
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# (SQL 실행용 커넥터 자격 — 있으면 결과 테이블 & 미리보기 제공)
SF_USER      = SF.get("user")
SF_PASSWORD  = SF.get("password")
SF_ACCOUNT   = SF.get("account")   # 예: qnehhfk-rub23142  (regionless 권장)

# 모델 선택(리전에 맞게)
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# 항상 적용되는 동작
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
# Connector 상태/이유 보이기 + 연결 함수
# ---------------------------
def _missing_secret_keys():
    miss = []
    if not SF_USER:     miss.append("user")
    if not SF_PASSWORD: miss.append("password")
    if not SF_ACCOUNT:  miss.append("account")
    return miss

@st.cache_resource(show_spinner=False)
def get_conn():
    """
    SQL 커넥터가 실패해도 앱이 죽지 않음.
    실패 사유는 session_state['sql_disabled_reason']에 저장.
    """
    miss = _missing_secret_keys()
    if miss:
        st.session_state["sql_disabled_reason"] = f"missing secrets: {', '.join(miss)}"
        return None
    try:
        return sf.connect(
            user=SF_USER,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,               # 예: qnehhfk-rub23142  (regionless 추천)
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA,
            role=ROLE,
            authenticator="snowflake",        # 중요!
            session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
        )
    except Exception as e:
        # Streamlit Cloud가 상세를 가릴 수 있으니 앞부분만 저장
        st.session_state["sql_disabled_reason"] = str(e)[:500]
        return None

def run_sql(sql: str) -> pd.DataFrame | None:
    conn = get_conn()
    if conn is None:
        return None
    q = qualify_sql(sql) if sql else sql
    try:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"❌ SQL 실행 에러: {e}")
        st.markdown("**Original SQL**"); st.code(sql or "<empty>", language="sql")
        if q and q != sql:
            st.markdown("**Qualified SQL (FQN 보정)**"); st.code(q, language="sql")
        return None

# ---------------------------
# Connector 상태 패널
# ---------------------------
with st.expander("SQL Connector Status", expanded=False):
    miss = _missing_secret_keys()
    conn = get_conn()
    if miss:
        st.error("🔒 SQL connector disabled: " + f"missing secrets: {', '.join(miss)}")
        st.caption("Streamlit Cloud → App → Settings → Secrets에서 snowflake.user/password/account 추가하세요.")
    elif conn is None:
        st.error("🔒 SQL connector disabled: " + st.session_state.get("sql_disabled_reason", "unknown"))
        st.caption("계정/비밀번호/네트워크 정책을 확인하세요.")
    else:
        try:
            ok = run_sql("SELECT 1 AS ok")
            if ok is not None and not ok.empty:
                st.success("✅ Connected")
        except Exception as e:
            st.error(f"Connector check failed: {e}")

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
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 인용 표식(【†1†】) 정규화
    s = s.replace("【†", "[").replace("†】", "]")
    return s.strip()

# ---------------------------
# Agents REST 호출 (JSON + SSE 모두 지원, UTF-8 강제)
# ---------------------------
def build_headers():
    # 헤더에는 ASCII만 들어가야 함 (PAT, Role 등은 ASCII)
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

def call_agents_rest(payload: dict, timeout: int = 90):
    r = requests.post(API_ENDPOINT, headers=build_headers(), json=payload, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    st.write(f"DEBUG: HTTP {r.status_code}, Content-Type={ctype}")

    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None

    # 1) JSON 응답
    if "application/json" in ctype:
        try:
            return r.json()
        except Exception:
            st.error("JSON 파싱 실패 → Raw 응답 표시")
            st.code(r.text[:2000] or "<empty>", language="json")
            return None

    # 2) SSE 응답: bytes → UTF-8로 강제 디코드 (모지바케 방지)
    if "text/event-stream" in ctype:
        body_text = r.content.decode("utf-8", errors="replace")
        # 간단 SSE 파서: event:/data: 블록 분해 → message.delta만 정규화
        events, cur_event, data_lines = [], None, []
        for line in body_text.splitlines() + [""]:
            if line.startswith("event:"):
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try: data_json = json.loads(data_str) if data_str else {}
                    except: data_json = {"raw": data_str}
                    if cur_event.startswith("message"):
                        events.append({"event":"message.delta","data":{"delta":data_json.get("delta",data_json)}})
                    else:
                        events.append({"event":cur_event,"data":data_json})
                cur_event, data_lines = line.split("event:",1)[1].strip(), []
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.strip() == "":
                if cur_event is not None:
                    data_str = "\n".join(data_lines).strip()
                    try: data_json = json.loads(data_str) if data_str else {}
                    except: data_json = {"raw": data_str}
                    if cur_event.startswith("message"):
                        events.append({"event":"message.delta","data":{"delta":data_json.get("delta",data_json)}})
                    else:
                        events.append({"event":cur_event,"data":data_json})
                    cur_event, data_lines = None, []
        return events

    # 3) 기타 → 원문
    st.error("알 수 없는 응답 형식 → Raw 바디 앞부분 표시")
    st.code(r.text[:2000] or "<empty>")
    return None

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

def parse_events_response(events: list):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v):
        if v and isinstance(v,str) and v.strip() and not sql_holder["v"]:
            sql_holder["v"] = v
    for ev in events:
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
    return "", None, []

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
    try:
        body = call_agents_rest(build_payload(query, max_results=5), timeout=90)
    except Exception as e:
        st.error(str(e))
        st.stop()

text, sql, citations = parse_any(body)

# 어시스턴트 메시지 구성 + 첨부들(테이블/expander)도 함께 저장
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
            # 히스토리에 저장
            df_safe = df.astype(str)
            tables_to_persist.append({
                "title": "Query Result",
                "data": df_safe.to_dict(orient="records")
            })
            assistant_chunks.append(f"_Query returned **{len(df)}** row(s)._")
        else:
            reason = st.session_state.get("sql_disabled_reason")
            if reason:
                st.info(f"🔒 SQL connector disabled: {reason}")

# Citations: 커넥터가 있어야 전문 미리보기 가능
if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        assistant_chunks.append("**Citations:** " + ", ".join(f"`{i}`" for i in ids))
    conn_ready = get_conn() is not None
    if conn_ready:
        for doc_id in ids:
            preview_sql = f"""
            SELECT CONVERSATION_ID, CUSTOMER_NAME, SALES_REP, DEAL_STAGE,
                   CONVERSATION_DATE, DEAL_VALUE, PRODUCT_LINE, TRANSCRIPT_TEXT
            FROM SALES_INTELLIGENCE.DATA.SALES_CONVERSATIONS
            WHERE CONVERSATION_ID = '{doc_id.replace("'", "''")}';
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
    else:
        # 연결이 없으면, ID만 표기하고 안내
        if ids:
            assistant_chunks.append("_Add `user/password/account` in secrets to open transcript previews._")

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
