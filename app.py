import streamlit as st
import snowflake.connector

st.set_page_config(page_title="SNOW챗봇", page_icon="❄️", layout="wide")
st.title("❄️ SNOW챗봇")
st.markdown("---")

# -------------------------------
# Snowflake 연결 (Secrets 사용)
# -------------------------------
@st.cache_resource
def init_connection():
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],   # 예: fv93338.ap-northeast-2.aws
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role=st.secrets["snowflake"]["role"],
        authenticator="snowflake",
        session_parameters={"CLIENT_SESSION_KEEP_ALIVE": True},
    )

conn = init_connection()

def run_query(query: str):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

# -------------------------------
# 사이드바
# -------------------------------
with st.sidebar:
    st.header("설정")
    lang = st.selectbox("언어 선택", ["한국어", "English"])
    st.slider("응답 길이", 1, 10, 5)

# -------------------------------
# 채팅 영역
# -------------------------------
st.header("💬 채팅")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 과거 메시지
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 입력창
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 간단 라우팅: Sarah Johnson 승/패/보류
    if "Sarah Johnson" in prompt:
        sql = """
        SELECT
          SUM(CASE WHEN win_status = TRUE  THEN 1 ELSE 0 END) AS WON_DEALS,
          SUM(CASE WHEN sales_stage = 'Lost' THEN 1 ELSE 0 END) AS LOST_DEALS,
          SUM(CASE WHEN win_status = FALSE AND sales_stage <> 'Lost' THEN 1 ELSE 0 END) AS PENDING_OR_OPEN
        FROM SALES_INTELLIGENCE.DATA.SALES_METRICS
        WHERE sales_rep = 'Sarah Johnson';
        """
        rows = run_query(sql)
        if rows and len(rows[0]) == 3:
            won, lost, pending = rows[0]
            bot_response = (
                "📊 결과:\n"
                f"- Won(승): {won}\n"
                f"- Lost(패): {lost}\n"
                f"- Pending/Open: {pending}\n"
            )
        else:
            bot_response = "결과가 없습니다. 데이터/권한을 확인해주세요."
    else:
        bot_response = f"'{prompt}'에 대한 Snowflake 쿼리를 아직 정의하지 않았어요."

    st.session_state.messages.append({"role":"assistant","content":bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# 하단 유틸
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 통계")
    st.metric("총 메시지", len(st.session_state.messages))
    st.metric("사용자 메시지", len([m for m in st.session_state.messages if m["role"] == "user"]))
with col2:
    st.success("🔧 도구")
    if st.button("채팅 초기화"):
        st.session_state.messages = []
        st.rerun()
    if st.button("샘플 질문"):
        st.write("💡 예시:")
        st.write("- How many deals did Sarah Johnson win compared to deals she lost?")
with col3:
    st.warning("ℹ️ 정보")
    st.write("**SNOW챗봇 (Community Cloud)**")
    st.write("Streamlit + Snowflake")
st.caption("© 2024 HDC DataLab - SNOW챗봇")
