from llmlingua import PromptCompressor
import pandas as pd
import streamlit as st
import openai
import re
import os
import tiktoken


# openai.api_key=""
openai.base_url = 'https://one.opengptgod.com/v1'
openai.api_key="sk-"

MAX_CONTEXT_LENGTH = 4000
CHUNK_SIZE = 512
compressed_content=""
chat_history = []

CHUNK_SIZE2 = 3200
def num_tokens(text,model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
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

def compress_novel_content(content,chunk_size=CHUNK_SIZE):
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    compressed_chunks = []
    num = num_tokens(content)
    for chunk in chunks:
        try:
            llm_lingua = PromptCompressor(
            model_name="./llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            )
            summary_response = llm_lingua.compress_prompt(chunk, rate=12000/num , force_tokens=['\n', '?'])
            # print(summary_response)
            compressed_chunks.append(summary_response.get('compressed_prompt'))
        except Exception as e:
            print(f"生成压缩时出错: {e}")
            return

    compressed_content = "".join(compressed_chunks)
    compressed_content = re.sub(r'\n+', ' ', compressed_content)
    compressed_content = re.sub(r'\s+', ' ', compressed_content)
    return compressed_content.strip()


def handle_question(content, question, compressed_content,model="gpt-4-32k-0613"):
    responses=[]
    if content and question:
        if len(content) <= MAX_CONTEXT_LENGTH:
            chat_history.append({"role": "user", "content": f"{content}\n文本: {question}"})
        else:
            if not compressed_content:
                content = compress_novel_content(content)
            chat_history.append({"role": "user", "content": f"压缩：{compressed_content}\n文本: {question}"})

        example_messages = [
            {
                "role": "user",
                "content": content,
            },
        ]
        tokens = num_tokens_from_messages(example_messages,model)
        print('tokens',tokens)
        print('lencontent',len(content))
        chunks = [content[i:i + CHUNK_SIZE2] for i in range(0, tokens, CHUNK_SIZE2)]
        for chunk in chunks:
            print('chunk',chunk)
            try:
                # print(content)
                response = openai.ChatCompletion.create( #model["gpt-4-1106-preview"(128k),"gpt-3.5-turbo-16k"(16k),"gpt-4-turbo"(128k),"gpt-4-32k"]
                model=model,
                messages=[
                    # {"role": "system", "content": question},
                    {"role": "user", "content": f"文本：{chunk}\n 要求：{question}"}
                ]
            )
                print('reponse',response.choices[0].message['content'])
                chat_history.append({"role": "assistant", "content": response.choices[0].message['content']})
                responses.append(response.choices[0].message['content'])

            except Exception as e:
                print(f"API 请求失败: {str(e)}")
        else:
            pass
    
        return content ,responses

with open("1089.txt" , "r" , encoding="utf-8") as r:
    text = r.read()
num = num_tokens(text)
print(num)    
