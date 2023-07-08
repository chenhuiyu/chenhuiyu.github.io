import os
import re
import sys
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set your OpenAI API key
# openai.api_key = 'your-openai-api-key'

@retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(20))
def translate_and_rename_file(src_path):
    # Extract the filename without extension
    base_name = os.path.basename(src_path)
    file_name = os.path.splitext(base_name)[0]

    messages = [
        {
            "role": "system",
            "content": "Translate the file name in to english: "+ file_name,
        }
    ]
    # Translate the filename using ChatGPT
    response = openai.Completion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        max_tokens=30,
        temperature=0,
    )

    en_file_name = response["choices"][0]["message"]["content"]

    # Rename the Chinese file
    zh_path = src_path.replace(file_name, en_file_name + '.zh-CN')

    os.rename(src_path, zh_path)

    # Create the path for the English version
    en_path = zh_path.replace('.zh-CN.md', '.en.md')

    # Read the Chinese blog post
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_content = f.read()

    # Split the content into blocks
    blocks = re.split(r'(```.*?```|\!\[.*?\]\(.*?\))', zh_content, flags=re.DOTALL)

    # Translate each block
    en_blocks = []
    for block in blocks:
        # If the block is a code block or an image, keep it as is
        if block.startswith('```') or block.startswith('!['):
            en_blocks.append(block)
        else:
            # Translate the block using ChatGPT
            messages = [
                {
                    "role": "system",
                    "content": "Translate the document in to english, use markdown format: ",
                },
                {
                    "role": "user",
                    "content": block,
                }
            ]
            # Translate the filename using ChatGPT
            response = openai.Completion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                max_tokens=30,
                temperature=0,
            )

            en_blocks.append(response["choices"][0]["message"]["content"])

    en_content = ''.join(en_blocks)

    # Write the English blog post
    with open(en_path, 'w', encoding='utf-8') as f:
        f.write(en_content)

if __name__ == "__main__":
    # Check if a file path was provided
    folder = '/Users/huiyu.chen/Documents/PersonalFiles/Blog/source/_posts/NLP Insights'
    
    for file in os.listdir(folder):
        if file.endswith('.md'):
            path = os.path.join(folder, file)
            # Translate and rename the file
            translate_and_rename_file(path)
