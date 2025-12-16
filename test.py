import requests
import json


question = "tôi thèm bún chả , hãy chỉ tôi làm món bún chả "

print(f" Câu hỏi: {question}\n")
print("=" * 60)

print("=" * 60)
response = requests.post(
    "http://localhost:3737/chat",
    json={
        "message": question,
        "session_id": "test_session",
        "stream": True  
    },
    stream=True  
)

if response.ok:
    print("🤖 AI 37: ", end="", flush=True)
    
    
    for line in response.iter_lines():
        if line:
            line_text = line.decode('utf-8')
            
            if line_text.startswith('data: '):
                json_str = line_text[6:]  # Bỏ "data: "
                
                try:
                    data = json.loads(json_str)
                    
                    if 'content' in data:
                        
                        print(data['content'], end="", flush=True)
                    
                    if data.get('done'):
                        print()  
                        break
                        
                except json.JSONDecodeError:
                    pass
else:
    print(f" Lỗi: {response.text}")