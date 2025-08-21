# ---------------------------
# Statements v2 실행 (REST)
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
        "timeout": timeout_s * 1000,         # ms
        "resultFormat": "json",              # v2: json 권장
        "parameters": {"MULTI_STATEMENT_COUNT": 1},
    }
    try:
        r = requests.post(
            SQL_ENDPOINT,
            headers=_sql_headers(),
            params={"async": "false"},       # v2는 params로 async=false 전달
            json=body,
            timeout=timeout_s + 15,
        )

        if r.status_code != 200:
            st.error(f"SQL HTTP {r.status_code} - {r.reason}")
            st.code(r.text[:2000] or "<empty>", language="json")
            return None

        data = r.json()  # v2 표준 응답
        res  = data.get("result", data)

        # 메타/컬럼 추출 (snake/camel 혼용 대비)
        meta = (
            res.get("resultSetMetaData")
            or res.get("result_set_meta_data")
            or {}
        )
        row_type = meta.get("rowType") or meta.get("row_type") or []
        cols = [c.get("name", f"C{i+1}") for i, c in enumerate(row_type)]

        # 행 추출 (형태 다양성 대비)
        rows = res.get("data")
        if rows is None:
            rows = res.get("rowset", [])

        values = []
        if isinstance(rows, list) and rows:
            # 형태 A: [[{"value":...}, ...], ...]
            if (
                isinstance(rows[0], list)
                and rows[0]
                and isinstance(rows[0][0], dict)
                and "value" in rows[0][0]
            ):
                values = [[cell.get("value") for cell in row] for row in rows]
            # 형태 B: [[v1, v2, ...], ...]
            elif isinstance(rows[0], list):
                values = rows
            # 형태 C: [{"COL":val, ...}, ...]
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
