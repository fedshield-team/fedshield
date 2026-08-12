from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    max_tokens=300,
    messages=[{"role": "user", "content": "Write one sentence about network security."}]
)

print("FULL RESPONSE OBJECT:")
print(response)
print("\n---")
print("CONTENT:", repr(response.choices[0].message.content))
print("FINISH REASON:", response.choices[0].finish_reason)