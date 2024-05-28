from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from tiqu import compress_novel_content
import tiktoken

import re
import unicodedata

def cn_or_en(text):
    subtext = re.sub(r'[,.!0-9]', '', text)
    for char in subtext:
        if 'CJK UNIFIED' in unicodedata.name(char, ''):
            return 'chinese'
        elif 'LATIN' in unicodedata.name(char, ''):
            return 'english'

# 加载文本文件
file_path = './686_en.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

en_or_cn = cn_or_en(text)

def num_tokens(text,model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))




num_token = num_tokens(text)
print(num_token)

if num_token > 16000:
    text = compress_novel_content(text)

# 设置OpenAI API环境变量
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


# 定义新类用于保存句子和ID
class SentenceWithID(BaseModel):
    context: str = Field(description="句子的内容")
    ID: int = Field(description="句子的ID")

class Item(BaseModel):
    商品名称: str = Field(description="商品的名称")
    商品开场: list[SentenceWithID] = Field(description="商品开场的句子和ID，1）疑问句/陈述句表达痛点和使用场景、2）购物体验、3）商家活动&促销、4）使用效果、5）行动号召。")
    商品介绍和卖点描述: list[SentenceWithID] = Field(description="商品介绍和卖点描述的句子和ID，1）商品展示、2）试穿/试用、3）使用过程、4）结果展示")
    商品价格: list[SentenceWithID] = Field(description="商品价格的句子和ID，1）商品优惠/折扣、2）商品价格")
    引导购买: list[SentenceWithID] = Field(description="引导购买的句子和ID，1）引导点击链接/优惠码/关注、 2）购买保障说明/售后")

class ItemList(BaseModel):
    商品列表: list[Item] = Field(description="单个或多个商品的合集")

# 定义输出解析器
parser = PydanticOutputParser(pydantic_object=ItemList)

# 定义模板以匹配所需信息
prompt_with_parser = """
你要将以下内容进行剪辑，剪辑要求：

第一：通读全文提取出描述不同商品的内容，提取“商品名称”。全文可能包含多个商品,不能遗漏。
第二：将每个商品里关于“商品开场”“商品介绍和卖点描述”“商品价格”“引导购买”的字幕的内容作为额外的信息补充到商品二级信息中，将整段话进行记录
第三：记录每句话在原文中的ID，并将这些句子和ID包含在输出中，确保每个ID是唯一的，且句子是完整的句子。
第四：确保每个商品的描述和卖点信息是准确对应的。

{格式信息}

原始字幕的内容在 >>> 和 <<< 之间
>>> {原始的字幕} <<<
"""
prompt_with_parser_en = """
你要将以下内容进行剪辑，剪辑要求：

第一：通读全文提取出描述不同商品的内容，提取“商品名称”。全文可能包含多个商品,不能遗漏。
第二：将每个商品里关于“商品开场”“商品介绍和卖点描述”“商品价格”“引导购买”的字幕的内容作为额外的信息补充到商品二级信息中，将整段话进行记录
第三：记录每句话在原文中的ID，并将这些句子和ID包含在输出中，确保每个ID是唯一的，且句子是完整的句子。
第四：确保每个商品的描述和卖点信息是准确对应的。
第五：请用英文输出结果。

{格式信息}

原始字幕的内容在 >>> 和 <<< 之间
>>> {原始的字幕} <<<
"""
if en_or_cn == 'chinese':
    prompt_template_with_parser = PromptTemplate(
    template=prompt_with_parser,
    input_variables=["原始的字幕"],
    partial_variables={"格式信息": parser.get_format_instructions()},
    output_parser=parser,
)
elif en_or_cn == 'english':
    prompt_template_with_parser = PromptTemplate(
        template=prompt_with_parser_en,
        input_variables=["原始的字幕"],
        partial_variables={"格式信息": parser.get_format_instructions()},
        output_parser=parser,
    )

# 创建LLM链
llm = ChatOpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_KEY"])
chain = LLMChain(llm=llm, prompt=prompt_template_with_parser)

# 执行链
output1 = chain.invoke({"原始的字幕": text})

# 定义新的PromptTemplate以处理第一个链的输出
json_conversion_prompt = """
你需要根据下面的文本将其转换为JSON格式。

文本:
{output}

请将以上文本转换为JSON格式。
"""



json_conversion_template = PromptTemplate(
    template=json_conversion_prompt,
    input_variables=["output"]
)

# 创建第二个LLM链
chain2 = LLMChain(llm=llm, prompt=json_conversion_template)

# 执行第二个链，将第一个链的输出传递给第二个链
output2 = chain2.invoke({"output": output1["text"]})

with open("686_en.json" , "w",encoding='utf-8') as w:
    w.write(output2["text"])
