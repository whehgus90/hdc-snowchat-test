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

def parse_streaming_response(resp: requests.Response):
    """
    SSE 스트림을 읽어 텍스트/SQL/서치결과(인용)들을 추출.
    """
    text_chunks = []
    sql_text = None
    citations = []

    last_event = None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("event:"):
            last_event = line.split("event:", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue

        data_str = line[5:].strip()
        # 일부 환경에서 keepalive로 "data: [DONE]" 이 올 수 있음
        if data_str == "[DONE]":
            break

        try:
            payload = json.loads(data_str)
        except Exception:
            # 혹시 배열형 이벤트(JSON array)로 올 경우 처리
            try:
                arr = json.loads(data_str if data_str.startswith("[") else f"[{data_str}]")
            except Exception:
                continue
            for evt in arr:
                handle_event_object(evt, text_chunks, citations, lambda s: set_sql(s))
            continue

        # 표준 SSE 경로: last_event 를 보고 처리
        evt_obj = {"event": last_event, "data": payload}
        handle_event_object(evt_obj, text_chunks, citations, lambda s: set_sql(s))

    # 내부 클로저에서 sql_text 갱신
    return ("".join(text_chunks).strip(), sql_text, citations)

    # 내부 도우미들
    def set_sql(s):
        nonlocal sql_text
        sql_text = s if s else sql_text

def handle_event_object(evt_obj, text_chunks, citations, set_sql_fn):
    """
    message.delta 이벤트 안의 delta.content를 풀어 텍스트/툴결과를 추출
    """
    if not evt_obj or evt_obj.get("event") != "message.delta":
        # error/done 등은 여기서 무시 (필요시 화면에 표시 가능)
        return
    data = evt_obj.get("data", {})
    delta = data.get("delta", {})
    for c in delta.get("content", []):
        ctype = c.get("type")
        if ctype == "text":
            text_chunks.append(c.get("text", ""))
        elif ctype == "tool_results":
            tr = c.get("tool_results", {})
            for item in tr.get("content", []):
                if item.get("type") == "json":
                    j = item.get("json", {})
                    # Agents가 반환하는 구조: {"text": "...", "sql": "...", "searchResults":[...]}
                    if "text" in j and isinstance(j["text"], str):
                        text_chunks.append(j["text"])
                    if "sql" in j and isinstance(j["sql"], str) and j["sql"]:
                        set_sql_fn(j["sql"])
                    if "searchResults" in j and isinstance(j["searchResults"], list):
                        for r in j["searchResults"]:
                            citations.append({
                                "source_id": r.get("source_id", ""),
                                "doc_id": r.get("doc_id", "")
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
