from langchain.llms import OpenAI
import os
import streamlit as st
from io import IOBase
from typing import Any, List, Optional, Union
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.schema.language_model import BaseLanguageModel


def create_csv_agent(
    llm: BaseLanguageModel,
    path: Union[str, IOBase, List[Union[str, IOBase]]],
    pandas_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Create csv agent by loading to a dataframe and using pandas agent."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    _kwargs = pandas_kwargs or {}
    if isinstance(path, (str, IOBase)):
        df = pd.read_csv(path, **_kwargs)
    elif isinstance(path, list):
        df = []
        for item in path:
            if not isinstance(item, (str, IOBase)):
                raise ValueError(f"Expected str or file-like object, got {type(path)}")
            df.append(pd.read_csv(item, **_kwargs))
    else:
        raise ValueError(f"Expected str, list, or file-like object, got {type(path)}")
    return create_pandas_dataframe_agent(llm, df, **kwargs)


llm = OpenAI(openai_api_key="")

st.set_page_config(page_title="Intract with CSV 📈")
st.header("Ask your CSV 📈")

def conversational_chat(query, agent):
    result = agent.run(query)
    return result

def main():
    uploaded_csv = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_csv:
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        if 'generated_responses' not in st.session_state:
            st.session_state['generated_responses'] = ["Hello! Ask me anything about " + uploaded_csv.name + " 🤗"]

        if 'user_queries' not in st.session_state:
            st.session_state['user_queries'] = ["Hey! 👋"]

        agent = create_csv_agent(llm, uploaded_csv, verbose=True)

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask your CSV data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

                # Create a placeholder for the progress bar
                progress_bar = st.empty()

                if submit_button and user_input:
                    st.session_state['user_queries'].append(user_input)

                    # Simulate a task with progress (for demonstration)
                    for percent_complete in range(1, 101):
                        progress_bar.progress(percent_complete)  # Update progress in a single line

                    agent_response = conversational_chat(user_input, agent)
                    st.session_state['generated_responses'].append(agent_response)

        if st.session_state['generated_responses']:
            with response_container:
                for i in range(len(st.session_state['generated_responses'])):
                    st.write(st.session_state["user_queries"][i], key=str(i) + '_user', avatar_style="big-smile")
                    st.write(st.session_state["generated_responses"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
