import requests
r = requests.get("https://huggingface.co")
print(r.status_code)
