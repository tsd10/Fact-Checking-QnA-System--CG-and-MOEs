import streamlit as st
from utils.router import route_question
from utils.experts import generate_answer
from utils.fact_check import fact_check

st.set_page_config(page_title="QA Assistant", layout="centered")
st.title("🎓 Knowledge QA Assistant")
st.caption("Ask a question. The system routes it to an expert and returns a factual answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if user_input := st.chat_input("Ask your factual question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Routing and generating answer..."):
        domain = route_question(user_input)
        answer = generate_answer(user_input, domain)

        response = f"🧭 **Routed Domain:** `{domain.capitalize()}`\n\n"
        response += f"🤖 **Expert Answer:** {answer}"

        # ✅ Check factuality
        is_factual = fact_check(user_input, answer)
        if is_factual:
            response += "\n\n✅ **Fact Check:** Passed"
        else:
            response += "\n\n❌ **Fact Check:** Might be incorrect"



    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
