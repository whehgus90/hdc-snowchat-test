# app.py — Streamlit Cloud (PAT + REST, SSE/JSON 자동, v1 Statements, 영구 히스토리)

import json, re, requests, pandas as pd, streamlit as st

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="❄️ SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (Cloud)")
st.caption("Cortex Search for calls · Analyst for metrics · Persistent chat")

# ---------------------------
# Secrets (Streamlit Cloud)
# ---------------------------
SF = st.secrets["snowflake"]
ACCOUNT_BASE = SF["account_base"].rstrip("/")    # ex) https://qnehhfk-rub23142.snowflakecomputing.com  (regionless)
SQL_BASE     = SF["sql_base"].rstrip("/")        # ex) https://hgb46705.snowflakecomputing.com          (regional)
PAT          = SF["pat"]
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# 모델 선택(리전에 맞는 것만)
MODEL_NAME = st.sidebar.selectbox("Model", ["mistral-large2", "llama3.3-70b"], index=0)

# 항상 적용
AUTO_RUN_SQL = True
AUTO_QUALIFY = True

# 리소스(FQN)
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"

AGENT_ENDPOINT = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"
SQL_ENDPOINT   = f"{SQL_BASE}/api/statements/v2/statements"

st.caption(f"AGENT_ENDPOINT = {AGENT_ENDPOINT}")
st.caption(f"SQL_ENDPOINT   = {SQL_ENDPOINT}")

# ---------------------------
# 유틸: FQN 보정
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
# Statements v1 실행
# ---------------------------
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

def run_sql_rest(sql: str, timeout_s: int = 90) -> pd.DataFrame | None:
    q = qualify_sql(sql).strip().rstrip(";")
    body = {
        "statement": q,
        "timeout":   timeout_s * 1000,  # ms
        "resultFormat": "json",
        "parameters": {"MULTI_STATEMENT_COUNT": 1},
    }
    try:
       r = requests.post(
    SQL_ENDPOINT,
    headers=_sql_headers(),
    params={"async": "false"},
    json=body,
    timeout=timeout_s + 15,
)
        if r.status_code != 200:
            st.error(f"SQL HTTP {r.status_code} - {r.reason}")
            st.code(r.text[:2000] or "<empty>", language="json")
            return None

        data = r.json()
        res  = data.get("result", {})
        meta = res.get("resultSetMetaData", {})
        row_type = meta.get("rowType", [])
        cols = [c.get("name", f"C{i+1}") for i, c in enumerate(row_type)]

        rows = res.get("data")
        if rows is None:
            rows = res.get("rowset", [])

        values = []
        if isinstance(rows, list) and rows:
            # 형태 A: 각 셀 dict {"value": ...}
            if isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], dict) and "value" in rows[0][0]:
                values = [[cell.get("value") for cell in row] for row in rows]
            # 형태 B: 그냥 값 배열
            elif isinstance(rows[0], list):
                values = rows
            # 형태 C: 객체 배열
            elif isinstance(rows[0], dict):
                return pd.DataFrame(rows)

        return pd.DataFrame(values, columns=cols if cols else None)

    except requests.exceptions.SSLError as e:
        st.error("❌ SQL SSL error — secrets['snowflake']['sql_base'] 값이 '계정로케이터.snowflakecomputing.com' 형식인지 확인.")
        st.code(str(e), language="text")
        return None
    except Exception as e:
        st.error(f"❌ SQL REST error: {e}")
        return None

# ---------------------------
# 텍스트 정리 + 응답 파서
# ---------------------------
def _normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = s.replace("【†", "[").replace("†】", "]")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _pull_from_tool_result(result_obj, citations, set_sql_fn):
    if not isinstance(result_obj, dict):
        return
    j = result_obj.get("json")
    if isinstance(j, str):
        try:
            j = json.loads(j)
        except Exception:
            j = None
    if not isinstance(j, dict):
        return
    for k in ("sql", "generated_sql", "sql_query"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            set_sql_fn(v)
            break
    for s in j.get("searchResults") or j.get("results") or []:
        if isinstance(s, dict):
            citations.append({
                "source_id": s.get("source_id", ""),
                "doc_id":    s.get("doc_id", "") or s.get("id", "")
            })

def parse_json_response(obj: dict):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v):
        if v and isinstance(v, str) and v.strip():
            sql_holder["v"] = v
    def pull(content):
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            return
        for item in content:
            t = item.get("type")
            if t == "text":
                txt = item.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            elif t == "tool_results":
                for r in item.get("tool_results", {}).get("content", []):
                    _pull_from_tool_result(r, citations, set_sql)
    for key in ("output", "message", "response", "data"):
        node = obj.get(key)
        if isinstance(node, dict) and "content" in node:
            pull(node["content"])
    if not text_parts and "content" in obj:
        pull(obj["content"])
    return _normalize_text("".join(text_parts)), sql_holder["v"], citations

# --- SSE 텍스트를 events로 파싱
def parse_sse_text(s: str) -> list:
    events = []
    cur_event, data_lines = None, []
    for line in (s or "").splitlines() + [""]:  # 마지막 flush
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
        elif line.strip() == "" and cur_event is not None:
            data_str = "\n".join(data_lines).strip()
            try:
                data_json = json.loads(data_str) if data_str else {}
            except Exception:
                data_json = {"raw": data_str}
            events.append({"event": cur_event, "data": data_json})
            cur_event, data_lines = None, []
    return events

def parse_events_response(events: list):
    text_parts, citations = [], []
    sql_holder = {"v": None}
    def set_sql(v):
        if v and isinstance(v, str) and v.strip() and not sql_holder["v"]:
            sql_holder["v"] = v
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("event") != "message.delta":
            # 최종 메시지가 message.completed로 올 수도 있으나, 대부분 delta 누적으로 충분
            continue
        payload = ev.get("data", {})
        delta = payload.get("delta", payload)  # 일부 구현은 data 바로 아래에 존재
        content = delta.get("content") or []
        if isinstance(content, dict):
            content = [content]
        for item in content:
            t = item.get("type")
            # {'type':'text','text':...}
            if t == "text":
                txt = item.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            # {'type':'tool_results', ...}
            elif t == "tool_results":
                for r in item.get("tool_results", {}).get("content", []):
                    _pull_from_tool_result(r, citations, set_sql)
            # tool_use는 무시(뒤이어 tool_results가 옴)
    return _normalize_text("".join(text_parts)), sql_holder["v"], citations

def parse_any(resp, content_type=""):
    if resp is None:
        return "", None, []
    # JSON(dict)
    if isinstance(resp, dict):
        return parse_json_response(resp)
    # events(list)
    if isinstance(resp, list):
        return parse_events_response(resp)
    # raw SSE 텍스트
    if isinstance(resp, str) and ("text/event-stream" in content_type or resp.startswith("event:")):
        return parse_events_response(parse_sse_text(resp))
    return "", None, []

# ---------------------------
# Intent 라우팅
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
# Agents 호출 (SSE/JSON 자동)
# ---------------------------
def agent_headers():
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
        sys_text = ("You are a helpful sales conversation assistant. Prefer cortex_search to find relevant call "
                    "transcripts and summarize them. Return a concise natural language summary and include brief citations.")
    elif intent == "SQL":
        sys_text = ("You are a helpful SQL analyst. Prefer cortex_analyst_text_to_sql for metrics/aggregations. "
                    "Also return a short natural language answer summarizing the result.")
    else:
        sys_text = ("You are a helpful assistant. For conversations/calls/transcripts, use cortex_search to retrieve "
                    "and summarize. For metrics/counts/aggregates, use cortex_analyst_text_to_sql to generate SQL. "
                    "Always include a short natural language answer.")

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {"role": "user",   "content": [{"type": "text", "text": user_text}]}
        ],
        "tools": tools,
        "tool_resources": tool_resources,
        # 'stream' 지정 안 함 → 서버 기본(SSE/JSON) 그대로 수용
    }

def call_agents(payload: dict, timeout=90):
    r = requests.post(AGENT_ENDPOINT, headers=agent_headers(), json=payload, timeout=timeout)
    ctype = r.headers.get("Content-Type", "")
    st.write(f"DEBUG: HTTP {r.status_code}, Content-Type={ctype}")
    if r.status_code != 200:
        st.error(f"HTTP {r.status_code} - {r.reason}")
        st.code(r.text[:2000] or "<empty>", language="json")
        return None, ctype
    if "application/json" in ctype:
        try:
            return r.json(), ctype
        except Exception:
            # JSON 실패 시 Raw
            return r.text, ctype
    # SSE or 기타
    return r.text, ctype

# ---------------------------
# Chat state (히스토리 보존)
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
    raw, ctype = call_agents(build_payload(query, max_results=5))

# 파싱(JSON/SSE 자동)
text, sql, citations = parse_any(raw, content_type=ctype)

# ---------------------------
# 폴백(SQL 질문인데 결과 비었을 때)
# ---------------------------
def _extract_rep_name(q: str) -> str | None:
    m = re.search(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", q or "")
    return m.group(1) if m else None

def _fallback_wonlost(q: str) -> tuple[str, pd.DataFrame | None]:
    rep = _extract_rep_name(q) or "Sarah Johnson"
    sql_fb = f"""
    SELECT
      win_status,
      COUNT(*) AS deal_count,
      MIN(close_date) AS start_date,
      MAX(close_date) AS end_date
    FROM SALES_INTELLIGENCE.DATA.SALES_METRICS
    WHERE UPPER(sales_rep) = UPPER('{rep.replace("'", "''")}')
    GROUP BY win_status
    ORDER BY win_status DESC NULLS LAST
    """
    return rep, run_sql_rest(sql_fb, timeout_s=90)

intent = detect_intent(query)
if not text and not sql and intent in ("SQL","Auto") and ("won" in query.lower() or "lost" in query.lower()):
    rep, df_fb = _fallback_wonlost(query)
    if df_fb is not None and not df_fb.empty:
        text = f"(fallback) **{rep}** → won/lost breakdown from SALES_METRICS."
        sql  = "/* fallback query executed */ " + "SELECT ..."

# ---------------------------
# 어시스턴트 메시지 구성 + 첨부 저장
# ---------------------------
assistant_chunks, tables_to_persist, expanders_to_persist = [], [], []

if text:
    assistant_chunks.append(text)

if sql:
    assistant_chunks.append("### Generated SQL\n```sql\n" + sql.strip() + "\n```")
    if AUTO_RUN_SQL:
        df = run_sql_rest(sql, timeout_s=120)
        if df is not None:
            st.write("### Query Result")
            st.dataframe(df, use_container_width=True)
            df_safe = df.astype(str)
            tables_to_persist.append({
                "title": "Query Result",
                "data": df_safe.to_dict(orient="records")
            })
            # win/lost 요약 자동
            cols_lower = [c.lower() for c in df.columns]
            if "win_status" in cols_lower and ("deal_count" in cols_lower or "count" in cols_lower):
                win_col = df.columns[cols_lower.index("win_status")]
                cnt_col = "deal_count" if "deal_count" in cols_lower else df.columns[cols_lower.index("count")]
                try:
                    won  = int(df[df[win_col] == True ][cnt_col].sum())
                    lost = int(df[df[win_col] == False][cnt_col].sum())
                    assistant_chunks.append(f"**Summary → Won: `{won}` / Lost(or not-won): `{lost}`**")
                except Exception:
                    pass

# Citations → 전문 미리보기 즉시 조회(REST) + 히스토리에 저장
if citations:
    ids = [c.get("doc_id","") for c in citations if c.get("doc_id")]
    if ids:
        st.markdown("### Citations")
        st.markdown("**IDs:** " + ", ".join(f"`{i}`" for i in ids))
    for doc_id in ids:
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

# 최종 메시지 저장/표시
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
