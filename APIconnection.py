import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("YOUR_API_KEY"))
with open('PredictText.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

client = genai.Client()
uploaded_file = client.files.upload(file_content=file_content, mime_type='text/plain')

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        uploaded_file,
        "\n\n",
        "List the top 3 most common words that begin with what is in the file in everyday english. No explanation, just the words by themselves.",
    ],
)
print(response.text)
