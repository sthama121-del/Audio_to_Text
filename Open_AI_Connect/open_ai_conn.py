#from dotenv import load_dotenv
#load_dotenv()

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about Texas"
)

print(response.output_text)

