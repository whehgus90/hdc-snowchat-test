# app.py
import json
import requests
import streamlit as st
import pandas as pd

# =========================
# App config
# =========================
st.set_page_config(page_title="SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (PAT + REST)")
st.caption("Snowflake Agents + Analyst + Search (Community Cloud)")

# =========================
# Secrets / constants
# =========================
SF = st.secrets["snowflake"]

# REST 호출용
ACCOUNT_BASE = SF["account_base"]  # 예: https://fv93338.ap-northeast-2.aws.snowflakecomputing.com
PAT          = SF["pat"]           # Programmatic Access Token
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# (선택) SQL 실행용 커넥터 자격
SF_USER      = SF.get("user")
SF_PASSWORD  = SF.get("password")
SF_ACCOUNT   = SF.get("account")   # 예: fv93338.ap-northeast-2.aws

# 모델(서울리전 미제공 가능 → Cross-Region 허용 시 사용 가능 모델로 선택)
MODEL_NAME   = st.sidebar.selectbox("Model", ["llama3.3-70b", "mistral-large2"], index=0)

# 리소스 이름(대문자/풀네임 권장)
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"

API_ENDPOINT = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"

# =========================
# Helpers
# =========================
def build_headers():
    return {
        "Authorization": f"Bearer {PAT}",
        "Accept": "application/json",  # JSON 모드(비스트리밍)
        "Content-Type": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
        "X-Snowflake-Role": ROLE,
        "X-Snowflake-Database": DATABASE,
        "X-Snowflake-Schema": SCHEMA,
        "X-Snowflake-Warehouse": WAREHOUSE,
    }

def build_payload(user_text: str, max_results: int = 5):
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
        "tools": [
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}},
            {"tool_spec": {"type": "cortex_search", "name": "search1"}}
        ],
        "tool_resources": {
            "analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
            "search1": {
                "name": CORTEX_SEARCH_SERVICE,
                "max_results": max_results,
                "id_column": "conversation_id"
            }
        }
    }

def _pull_from_tool_result(result_obj, citations, set_sql_fn):
    """tool_results 한 덩어리에서 sql / citations 추출"""
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

    # SQL 키 이름 후보들 모두 스캔
    for k in ("sql", "generated_sql", "sql_query"):
        if isinstance(j.get(k), str) and j[k].strip():
            set_sql_fn(j[k])
            break

    # 검색 인용 결과
    sr = j.get("searchResults") or j.get("results") or []
    if isinstance(sr, list):
        for s in sr:
            if isinstance(s, dict):
                citations.append({
                    "source_id": s.get("source_id", ""),
                    "doc_id": s.get("doc_id", "") or s.get("id", "")
                })

def parse_agent_json(obj):
    """
    JSON 응답에서 텍스트/SQL/인용을 최대한 폭넓게 추출
    return (text, sql, citations)
    """
    text_parts = []
    citations = []
    sql_holder = {"v": None}

    def set_sql(v: str):
        if v and isinstance(v, str) and v.strip():
            sql_holder["v"] = v

    # content 배열에서 text/tool_results를 긁어오는 내부 함수
    def pull_from_content(content):
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            return
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                txt = item.get("text")
                if isinstance(txt, str) and txt:
                    text_parts.append(txt)
            elif t == "tool_results":
                tr = item.get("tool_results", {})
                for r in tr.get("content", []):
                    _pull_from_tool_result(r, citations, set_sql)

    # 1) 흔한 루트 키들 훑기
    for key in ("output", "message", "response", "data"):
        node = obj.get(key)
        if isinstance(node, dict) and "content" in node:
            pull_from_content(node["content"])

    # 2) 루트에 content가 바로 있을 수도
    if not text_parts and "content" in obj:
        pull_from_content(obj["content"])

    # 3) 그래도 없으면 전체 dict 깊게 훑어서 content 후보 찾기(방어적)
    if not text_parts and sql_holder["v"] is None:
        def deep_walk(x):
            if isinstance(x, dict):
                if "content" in x:
                    pull_from_content(x["content"])
                for v in x.values():
                    deep_walk(v)
            elif isinstance(x, list):
                for v in x:
                    deep_walk(v)
        deep_walk(obj)

    text = "\n\n".join(text_parts).strip()
    return text, sql_holder["v"], citations

# =========================
# (선택) 생성된 SQL 실행 지원
# =========================
@st.cache_resource
def get_sql_connection():
    """필요시에만 연결. user/password/account가 모두 있을 때만."""
    import snowflake.connector  # lazy import
    if not (SF_USER and SF_PASSWORD and SF_ACCOUNT):
        return None
    return snowflake.connector.connect(
        user=SF_USER,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,   # 예: fv93338.ap-northeast-2.aws
        warehouse=WAREHOUSE,
        database=DATABASE,
        schema=SCHEMA,
        role=ROLE,
        session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
    )

def run_sql_in_snowflake(sql: str) -> pd.DataFrame | None:
    conn = get_sql_connection()
    if conn is None:
        st.info("🔐 secrets에 user/password/account가 없어 SQL 실행은 건너뜁니다.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"❌ SQL 실행 에러: {e}")
        st.code(sql, language="sql")
        return None

# =========================
# Session state / Sidebar
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    max_results = st.slider("Max search results", 1, 10, 5)
    do_run_sql = st.checkbox("Execute generated SQL", value=False)
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

# =========================
# Chat history
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# Input & Call
# =========================
query = st.chat_input("질문을 입력하세요 (예: How many deals did Sarah Johnson win compared to lost?)")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Calling Snowflake Agents..."):
        payload = build_payload(query, max_results=max_results)
        resp = requests.post(API_ENDPOINT, headers=build_headers(), data=json.dumps(payload))

    # ---- 에러 처리 ----
    if resp.status_code != 200:
        st.error(f"HTTP {resp.status_code} — {resp.reason}")
        try:
            st.code(resp.text[:2000], language="json")
        except Exception:
            st.text(resp.text[:2000])
    else:
        # JSON 바디 파싱
        try:
            body = resp.json()
        except Exception:
            body = None

        if not body:
            st.warning("응답 본문이 비었습니다.")
            st.code(resp.text[:2000], language="json")
        else:
            text, sql, cites = parse_agent_json(body)

            # 1) 답 텍스트
            if text:
                text = text.replace("【†", "[").replace("†】", "]")
                st.session_state.messages.append({"role": "assistant", "content": text})
                with st.chat_message("assistant"):
                    st.markdown(text)
            else:
                st.warning("응답 텍스트가 없어 분석 결과를 표시합니다.")

            # 2) 생성된 SQL
            if sql:
                st.markdown("### Generated SQL")
                st.code(sql, language="sql")

                # 선택 시 실제 실행
                if do_run_sql:
                    df = run_sql_in_snowflake(sql)
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                        if not text:
                            with st.chat_message("assistant"):
                                st.markdown("텍스트 응답은 없어 SQL 결과를 표시했어요.")
            else:
                if not text:
                    st.info("텍스트/SQL 모두 없어 Raw 응답을 표시합니다.")
                    st.code(json.dumps(body, ensure_ascii=False, indent=2)[:2000], language="json")

            # 3) Citations
            if cites:
                st.markdown("### Citations")
                st.dataframe(pd.DataFrame(cites), use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("© 2025 HDC DataLab · Streamlit + Snowflake Cortex Agents (PAT)")
