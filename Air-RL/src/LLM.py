import requests

url = 'https://cloud.luchentech.com/api/maas/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer 9736f492-2b26-4903-a095-a38a06a825b0'
}
payload = {
    "model": "deepseek_r1",
    "messages": [
      {
        "role": "user",
        "content": "以贴吧老哥的口吻，回击一下deep seek不如ChatGPT的言论"
      }
    ],
    "stream": False,
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.json())