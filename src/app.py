import os
import asyncio
import tiktoken
from modules import *
import streamlit as st
from pathlib import Path
from openai import AsyncOpenAI
from langchain.document_loaders import UnstructuredFileLoader

openai_api_key_env = os.getenv("OPENAI_API_KEY", None)

if "INITIAL_PROMPT" not in st.session_state:
    st.session_state["INITIAL_PROMPT"] = "You are an expert in the field of text analysis and production (writing). Based on the user uploaded text and their instructions, produce an appropriate response."

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

@st.cache_data(show_spinner=False)
def load_file(fn):
    loader = UnstructuredFileLoader(fn)
    return loader.load()

def construct_request_message(
    messages: list[dict],
    module_prompt: str,
):
    return messages + [
        {"role": "system", "content": f"Please adhere to below output format to organize your response. Use the instructions inside the <> brackets as guidance but don't print the instructions themselves in the response. Start output immediately without any upfront explanations:\n\n{module_prompt}"}
    ]


async def main():

    st.set_page_config(
        page_title="TextWizard",
        page_icon="🧙‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ### Layout ###

    settings_container = st.container()
    status_container = st.empty()
    generator_container = st.container()

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=openai_api_key_env if openai_api_key_env else "")
        if not openai_api_key or len(openai_api_key) == 0:
            st.stop()
        
    client = AsyncOpenAI(api_key=openai_api_key)

    with settings_container:
        st.title("TextWizard")

        system_prompt_col, file_upload_col = st.columns(2)

        with system_prompt_col:
            initial_prompt = st.text_area("System Prompt", value=st.session_state["INITIAL_PROMPT"], height=120)
        if initial_prompt != st.session_state["INITIAL_PROMPT"]:
            st.session_state["INITIAL_PROMPT"] = initial_prompt

        with file_upload_col:
            uploaded_file = st.file_uploader("Upload a File (.docx, .pdf)", type=["docx", "pdf"])
        if not uploaded_file:
            st.stop()

        user_prompt_col, extracted_text_col = st.columns(2)

        with user_prompt_col:
            user_prompt = st.text_area("User Instruction", placeholder="Instruct the AI how to process your text", height=100)
            specialty = st.selectbox("Specialty", options=sorted(MODULES.keys()), index=None)
            if specialty:
                sections = st.multiselect("Sections", options=sorted(MODULES[specialty].keys()))

        with extracted_text_col:

            extension = Path(uploaded_file.name).suffix
            temp_fn = Path("/tmp") / f"{uploaded_file.file_id}{extension}"
            temp_fn.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_fn, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner("Loading document..."):
                docs = load_file(str(temp_fn))

            # Remove the temp file folder (including all its contents)
            temp_fn.unlink()

            extracted_text = st.text_area("Review Uploaded Text", value=docs[0].page_content, height=270)
            
    messages = [
        {"role": "system", "content": initial_prompt},
        {"role": "system", "content": f"USER-UPLOADED-TEXT:\n\n{extracted_text}"},
        {"role": "user", "content": user_prompt},
    ]

    total_tokens = 0
    highest_token_use = 0

    with generator_container:
        if sections:
            st.write("**Output Sections Preview**")
            for i, section in enumerate(sections):
                st.markdown(f"{i + 1}. {section}", help=MODULES[specialty][section])
                num_tokens = num_tokens_from_messages(
                    construct_request_message(messages, MODULES[specialty][section])
                )
                if num_tokens > highest_token_use:
                    highest_token_use = num_tokens
                total_tokens += num_tokens
    
    
    max_tokens = 124 * 1000 # 120k assuming gpt-4-turbo with 128k total tokens but 8k left for output
    status_container.success(f"Approximate total tokens: {total_tokens}. Approximate single request tokens: {highest_token_use} or {int(round(100 * highest_token_use / max_tokens, 0))}% of max token count.")
    
    if highest_token_use > max_tokens:
        status_container.error(f"Error: the uploaded file is too large. Combined with the longest section template, it has {highest_token_use} tokens, but the maximum is {max_tokens}.")
        st.stop()

    if total_tokens <= 0:
        status_container.info("Please select at least one section.")
        st.stop()

    submit = st.button("Submit", type="primary")

    if submit:

        with generator_container:

            reply_box = st.empty()
            reply_message = ""

            with reply_box:
                with st.chat_message("assistant", avatar="🧙‍♂️"):
                    st.markdown(f"{reply_message}█")

            for i, section in enumerate(sections):
                request_messages = construct_request_message(messages, MODULES[specialty][section])
            
                async for chunk in await client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=request_messages,
                    temperature=0.1,
                    max_tokens=4000,
                    stream=True,
                ):
                    content = chunk.choices[0].delta.content
                    if content:
                        reply_message += content
                        with reply_box:
                            with st.chat_message("assistant", avatar="🧙‍♂️"):
                                st.markdown(f"{reply_message}█")
            
            with reply_box:
                with st.chat_message("assistant", avatar="🧙‍♂️"):
                    st.markdown(reply_message)

if __name__ == "__main__":
    asyncio.run(main())