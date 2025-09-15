import time
import random
import requests
from config import SILICON_API_KEY, SILICON_API_URL

def call_silicon_llm(prompt: str, model: str = "", temperature: float = 0.7, top_p: float = 0.7, max_tokens: int = 1024, delay: float = 2.0, max_retries: int = 3) -> str:
    headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    time.sleep(delay + random.uniform(0, 1))
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(SILICON_API_URL, headers=headers, json=payload, timeout=120)
            if response.status_code == 200 and "choices" in response.json():
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * delay + random.uniform(0, 2)
                    print(f"[INFO] rate limited, retry in {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    print("[ERROR] max retries reached due to rate limit")
                    return "RATE_LIMITED"
            else:
                print("[API ERROR]", response.status_code, response.text[:500])
                return "API_ERROR"
        except Exception as e:
            print("[API EXCEPTION]", e)
            if attempt < max_retries:
                wait_time = delay * 2
                print(f"[INFO] retry in {wait_time}s")
                time.sleep(wait_time)
                continue
            else:
                return "API_EXCEPTION"
    return "API_ERROR"
