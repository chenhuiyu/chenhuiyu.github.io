import os
import time
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import openai


# Set your OpenAI API key
# openai.api_key = 'your-openai-api-key'
openai.organization = "org-vKWF67g2KjEjw9xurInSmRjF"
openai.api_key = "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"

class BlogHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Only process Markdown files
        if event.src_path.endswith('.md'):
            # Check if the file is a Chinese blog post
            if '.zh-CN.md' not in event.src_path:
                self.translate_and_rename_file(event.src_path)

    def translate_and_rename_file(self, src_path):
        # Extract the filename without extension
        base_name = os.path.basename(src_path)
        file_name = os.path.splitext(base_name)[0]

        # Translate the filename using ChatGPT
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=file_name,
            max_tokens=10,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        en_file_name = response.choices[0].text.strip()

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
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=block,
                    max_tokens=5000,
                    temperature=0.5,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                en_blocks.append(response.choices[0].text.strip())

        en_content = ''.join(en_blocks)

        # Write the English blog post
        with open(en_path, 'w', encoding='utf-8') as f:
            f.write(en_content)

if __name__ == "__main__":
    path = '.'  # The directory to watch
    event_handler = BlogHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
