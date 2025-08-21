# app.py
import json
import requests
import streamlit as st
import pandas as pd

# -------------------------------
# App config
# -------------------------------
st.set_page_config(page_title="SNOW챗봇 (Cortex Agents)", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇 - Cortex Agents (PAT + REST)")
st.caption("Snowflake Agents + Analyst + Search (Community Cloud)")

# -------------------------------
# Secrets / constants
# -------------------------------
SF = st.secrets["snowflake"]
ACCOUNT_BASE = SF["account_base"]                     # e.g. https://fv93338.ap-northeast-2.aws.snowflakecomputing.com
PAT          = SF["pat"]                              # Programmatic Access Token
ROLE         = SF.get("role", "SALES_INTELLIGENCE_RL")
WAREHOUSE    = SF.get("warehouse", "SALES_INTELLIGENCE_WH")
DATABASE     = SF.get("database",  "SALES_INTELLIGENCE")
SCHEMA       = SF.get("schema",    "DATA")

# 모델은 서울 리전에 없을 수 있으니 Cross-Region 허용 후, 미국에서 제공되는 걸로 설정
MODEL_NAME   = st.sidebar.selectbox("Model", ["llama3.3-70b", "mistral-large2"], index=0)

# ✅ 대소문자 유지 + 풀네임
CORTEX_SEARCH_SERVICE = "SALES_INTELLIGENCE.DATA.SALES_CONVERSATION_SEARCH"
SEMANTIC_MODEL_FILE   = "@SALES_INTELLIGENCE.DATA.MODELS/sales_metrics_model.yaml"


API_ENDPOINT = f"{ACCOUNT_BASE}/api/v2/cortex/agent:run"

# -------------------------------
# Helpers
# -------------------------------
def build_headers():
    return {
        "Authorization": f"Bearer {PAT}",
        "Accept": "application/json",  # ✅ 임시: 에러 바디를 JSON으로 받기
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

def parse_streaming_response(events):
    """
    events: requests.Response (SSE), list[dict] (이벤트 배열), dict/json 문자열 등
    return: (text, sql, citations)
    """
    text_chunks = []
    citations = []
    sql_holder = {"value": None}  # 클로저로 SQL 담는 컨테이너

    def set_sql(v: str | None):
        if v and isinstance(v, str) and v.strip():
            sql_holder["value"] = v

    # 1) requests.Response (SSE 스트림)
    if hasattr(events, "iter_lines"):
        for raw in events.iter_lines():
            if not raw:
                continue
            # SSE 포맷: b"data: {...}"
            if raw.startswith(b"data: "):
                payload = raw[6:]
                try:
                    evt_obj = json.loads(payload)
                    handle_event_object(evt_obj, text_chunks, citations, set_sql)
                except Exception:
                    # 에러 이벤트/하트비트 등은 무시
                    pass

    # 2) 이미 파싱된 리스트/딕셔너리
    elif isinstance(events, list):
        for evt_obj in events:
            handle_event_object(evt_obj, text_chunks, citations, set_sql)
    elif isinstance(events, dict):
        handle_event_object(events, text_chunks, citations, set_sql)

    # 3) 문자열(JSON)일 수도 있음
    elif isinstance(events, str):
        try:
            obj = json.loads(events)
            if isinstance(obj, list):
                for evt_obj in obj:
                    handle_event_object(evt_obj, text_chunks, citations, set_sql)
            elif isinstance(obj, dict):
                handle_event_object(obj, text_chunks, citations, set_sql)
        except Exception:
            pass

    return "".join(text_chunks).strip(), sql_holder["value"], citations

def handle_event_object(evt_obj: dict, text_chunks: list[str], citations: list[dict], set_sql_fn):
    """
    한 개의 SSE 이벤트 객체 처리
    """
    ev = evt_obj.get("event")

    # 에러 이벤트는 예외로 올려서 상위에서 표시
    if ev == "error":
        err = evt_obj.get("data", {})
        raise RuntimeError(err.get("message", "Unknown error from SSE"))

    # Delta 타입만 처리 (이름이 약간 다를 수 있어 유연 처리)
    if ev not in ("message.delta", "response.delta", "chunk.delta"):
        return

    data = evt_obj.get("data", {})
    delta = data.get("delta", {})
    contents = delta.get("content", [])

    # content가 dict로 오는 경우도 방어
    if isinstance(contents, dict):
        contents = [contents]

    for item in contents:
        t = item.get("type")
        if t == "text":
            text_chunks.append(item.get("text", ""))

        elif t == "tool_results":
            tr = item.get("tool_results", {})
            for result in tr.get("content", []):
                if result.get("type") == "json":
                    j = result.get("json", {})
                    # json이 문자열로 오는 경우도 있음
                    if isinstance(j, str):
                        try:
                            j = json.loads(j)
                        except Exception:
                            j = {}
                    # SQL 추출
                    if isinstance(j, dict) and j.get("sql"):
                        set_sql_fn(j["sql"])
                    # 검색 인용 문서 추출
                    for s in j.get("searchResults", []) if isinstance(j, dict) else []:
                        citations.append({
                            "source_id": s.get("source_id", ""),
                            "doc_id": s.get("doc_id", "")
                        })

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Settings")
    max_results = st.slider("Max search results", 1, 10, 5)
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------
# Chat history
# -------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------
# Input & Call
# -------------------------------
query = st.chat_input("질문을 입력하세요 (예: How many deals did Sarah Johnson win compared to lost?)")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Calling Snowflake Agents..."):
        payload = build_payload(query, max_results=max_results)
        resp = requests.post(API_ENDPOINT, headers=build_headers(), data=json.dumps(payload))  # ✅ stream 제거

        if resp.status_code != 200:
            st.error(f"HTTP {resp.status_code} — {resp.reason}")
            # 서버가 JSON 에러를 줄 수 있으니 본문 일부 표시
            try:
                st.code(resp.text[:2000], language="json")
            except Exception:
                st.text(resp.text[:2000])
        else:
            text, sql, cites = parse_streaming_response(resp)

            # 답 텍스트
            if text:
                # 특수 괄호 교정(인라인 인용 표기 등)
                text = text.replace("【†", "[").replace("†】", "]")
                st.session_state.messages.append({"role": "assistant", "content": text})
                with st.chat_message("assistant"):
                    st.markdown(text)
            else:
                st.warning("응답 텍스트가 비었습니다.")

            # 생성된 SQL이 있으면 표시 (분석 결과)
            if sql:
                st.markdown("### Generated SQL")
                st.code(sql, language="sql")

            # 검색 인용 결과가 있으면 표시 (문서/대화 출처)
            if cites:
                st.markdown("### Citations")
                cite_df = pd.DataFrame(cites)
                st.dataframe(cite_df)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("© 2025 HDC DataLab · Streamlit + Snowflake Cortex Agents (PAT)")
