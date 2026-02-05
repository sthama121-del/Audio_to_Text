import base64
import requests
import os
from openai import OpenAI

# 1. Setup the Client
# Make sure your API key is set in your environment or paste it here
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Define the image URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3b/LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg"

# 3. Download the image with a User-Agent header
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    base64_image = base64.b64encode(response.content).decode('utf-8')
    print("Successfully downloaded and encoded the image!")
except Exception as e:
    print(f"Failed to download or encode image: {e}")
    exit()

# 4. Make the API Call
try:
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Who is in this image and what are they doing?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    # 5. Print the result
    print(chat_response.choices[0].message.content)

except Exception as e:
    print(f"An error occurred during the API call: {e}")