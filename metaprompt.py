import anthropic
import os
import pandas as pd
import re
import streamlit as st
# read in anthropic api key from anthropic_api_key.txt
with open('anthropic_api_key.txt', 'r') as file:
    ANTHROPIC_API_KEY = file.read().strip()
MODEL_NAME = "claude-3-opus-20240229"
CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# read in the metaprompt template
with open('metaprompt_template.txt', 'r') as file:
    metaprompt = file.read()


def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

def remove_empty_tags(text):
    return re.sub(r'\n<(\w+)>\s*</\1>\n', '', text, flags=re.DOTALL)

def extract_prompt(metaprompt_response):
    between_tags = extract_between_tags("Instructions", metaprompt_response)[0]
    return between_tags[:1000] + remove_empty_tags(remove_empty_tags(between_tags[1000:]).strip()).strip()

def extract_variables(prompt):
    pattern = r'{([^}]+)}'
    variables = re.findall(pattern, prompt)
    return set(variables)

def remove_inapt_floating_variables(prompt):
    message = CLIENT.messages.create(
        model=MODEL_NAME,
        messages=[{'role': "user", "content": remove_floating_variables_prompt.replace("{$PROMPT}", prompt)}],
        max_tokens=4096,
        temperature=0
    ).content[0].text
    return extract_between_tags("rewritten_prompt", message)[0]

def find_free_floating_variables(prompt):
    variable_usages = re.findall(r'\{\$[A-Z0-9_]+\}', prompt)

    free_floating_variables = []
    for variable in variable_usages:
        preceding_text = prompt[:prompt.index(variable)]
        open_tags = set()

        i = 0
        while i < len(preceding_text):
            if preceding_text[i] == '<':
                if i + 1 < len(preceding_text) and preceding_text[i + 1] == '/':
                    closing_tag = preceding_text[i + 2:].split('>', 1)[0]
                    open_tags.discard(closing_tag)
                    i += len(closing_tag) + 3
                else:
                    opening_tag = preceding_text[i + 1:].split('>', 1)[0]
                    open_tags.add(opening_tag)
                    i += len(opening_tag) + 2
            else:
                i += 1

        if not open_tags:
            free_floating_variables.append(variable)

    return free_floating_variables

st.title("Metaprompt Engineering")

st.markdown(
    """
    <a href="https://github.com/quinfer/prompt_engineering_app.git" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-View%20Source-blue?logo=github" alt="View Source on GitHub">
    </a>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
## Instructions

1. Enter the task for the prompt in the text area below.
2. If you have any variables to include, enter them as a comma-separated list in the input field.
3. Click the "Generate Prompt" button to generate the adjusted prompt.
4. The adjusted prompt will be displayed in the code block.
""")

task = st.text_area("Enter the task for the prompt:", height=200)
variables = st.text_input("Enter variables (comma-separated, optional):")

api_call_counter = 0

if st.button("Generate Prompt"):
    if task:
        prompt = metaprompt.replace("{{TASK}}", task.strip()).strip()
        print(f"Prompt: {prompt!r}")
        message = CLIENT.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": prompt
                }
            ],
            temperature=0
        ).content[0].text

        api_call_counter += 1

        extracted_prompt_template = extract_prompt(message)
        extracted_variables = extract_variables(message)

        if variables:
            input_variables = [var.strip() for var in variables.split(",")]
            missing_variables = set(input_variables) - extracted_variables
            if missing_variables:
                st.warning(f"The following variables are not used in the prompt: {', '.join(missing_variables)}")

        floating_variables = find_free_floating_variables(extracted_prompt_template)
        if len(floating_variables) > 0:
            extracted_prompt_template = remove_inapt_floating_variables(extracted_prompt_template)
            api_call_counter += 1

        st.subheader("Adjusted Prompt:")
        st.code(extracted_prompt_template)

        st.subheader("API Call Counter:")
        st.write(f"Number of Anthropic API calls: {api_call_counter}")
    else:
        st.warning("Please enter a task for the prompt.")
        
# Add a references section
st.subheader("References and Further Reading")
st.markdown(
    """
    - [MetaPrompt: An Effective Method for Prompt Engineering](https://arxiv.org/abs/2106.12672) by Liu et al. (2021)
    - [Prompt Engineering: The Ultimate Guide](https://lilianweng.github.io/posts/2021-08-15-prompt-engineering/) by Lillian Weng
    - [Metaprompt Engineering: A New Approach to Prompt Design](https://openai.com/blog/metaprompt-engineering/) by OpenAI
    - [A Guide to Metaprompt Engineering](https://huggingface.co/docs/transformers/metaprompt_engineering) by Hugging Face
    """
)