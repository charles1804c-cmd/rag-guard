import streamlit as st
from rag import answer
from judge import check_faithfulness

st.set_page_config(page_title="RAG-Guard", page_icon="🛡️")
st.title("🛡️ RAG-Guard: Answers You Can Trust")
st.caption("Ask about the PM-KISAN guidelines. Every answer is fact-checked against the source text.")

q = st.text_input("Your question", placeholder="Who is not eligible for PM-KISAN?")

if q:
    with st.spinner("Searching documents and verifying the answer..."):
        ans, docs = answer(q)
        result = check_faithfulness(ans, docs)

    st.subheader("Answer")
    st.write(ans)

    if result.get("note") == "Model declined to answer":
        st.info("The assistant said the documents don't contain this answer.")
    elif result["flag"]:
        st.error(f"⚠️ Low confidence. Faithfulness score: {result['score']}")
    else:
        st.success(f"✅ Well supported. Faithfulness score: {result['score']}")

    if result.get("claims"):
        with st.expander("Claim-by-claim verification"):
            for c in result["claims"]:
                icon = "✅" if c["supported"] else "❌"
                st.markdown(f"{icon} **{c['claim']}**")
                if c.get("evidence"):
                    st.caption(f"Evidence: {c['evidence']}")

    with st.expander("Source chunks used"):
        for d in docs:
            st.caption(f"{d.metadata.get('source')} (page {d.metadata.get('page')})")
            st.write(d.page_content)