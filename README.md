# 🤖 AI 37 Chatbot - RAG System

Chatbot AI thông minh với khả năng tìm kiếm web, tính toán, thực thi code và hiểu hình ảnh, được xây dựng trên Flask và LM Studio.

## ✨ Tính năng

### 🎯 Giai đoạn 1: Core Chatbot
- ✅ Kết nối LM Studio với GPU acceleration
- ✅ Streaming response (hiển thị từng chữ như ChatGPT)
- ✅ Lưu lịch sử hội thoại vào file
- ✅ Tóm tắt tự động khi hội thoại dài

### 🔍 Giai đoạn 2: Advanced Tools
- ✅ **Web Search**: Tìm kiếm thông tin từ Internet (DuckDuckGo)
- ✅ **Calculator**: Tính toán phức tạp (Sympy)
- ✅ **Code Execution**: Chạy Python code an toàn (RestrictedPython)
- ✅ **Multi-turn Clarification**: Hỏi rõ ràng khi cần thiết

### 🖼️ Giai đoạn 3: Vision
- ✅ **Image Understanding**: Hiểu và phân tích hình ảnh (LLaVA Vision Model)

## 🛠️ Công nghệ sử dụng

- **Backend**: Flask, Flask-CORS
- **AI Model**: LM Studio (vistral-7b-chat@q8, llava-v1.5-7b)
- **Web Search**: DuckDuckGo Search (ddgs)
- **Calculator**: SymPy
- **Code Execution**: RestrictedPython
- **Web Scraping**: BeautifulSoup4
- **Weather API**: AccuWeather
- **Search API**: Bing Search API (optional)

## 📋 Yêu cầu hệ thống

- Python 3.8+
- LM Studio (đã cài đặt và chạy local)
- GPU khuyến nghị cho hiệu suất tốt

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/PhanHieudc37/chatbot-rag.git
cd chatbot-rag
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Keys

Tạo file `.env` từ template:

```bash
copy .env.example .env
```

Mở file `.env` và điền API keys của bạn:

```env
# API Keys
ACCUWEATHER_API_KEY=your_accuweather_api_key_here
BING_SEARCH_API_KEY=your_bing_search_api_key_here

# RAG Parameters
RAG_MAX_TOKENS=2000
RAG_TEMPERATURE=0.7
```

**Lấy API Keys:**
- **AccuWeather**: [https://developer.accuweather.com/](https://developer.accuweather.com/)
- **Bing Search**: [Azure Portal](https://portal.azure.com) → Tạo "Bing Search v7" resource

### 4. Khởi động LM Studio

1. Mở LM Studio
2. Load model: `vistral-7b-chat@q8`
3. Start server tại `http://localhost:1234`

### 5. Chạy ứng dụng

**Windows:**
```bash
START.bat
```

**Linux/Mac:**
```bash
python serve_rag.py
```

Truy cập: **http://localhost:3737**

## 📁 Cấu trúc thư mục

```
chatbot-rag/
├── serve_rag.py          # Main Flask application
├── config.py             # Configuration management
├── test.py               # Test scripts
├── requirements.txt      # Python dependencies
├── START.bat            # Windows startup script
├── .env                 # API keys (không commit lên Git)
├── .env.example         # Template cho .env
├── .gitignore           # Git ignore rules
├── templates/
│   └── index.html       # Web UI
├── static/
│   ├── style.css        # Styles
│   └── script.js        # Frontend logic
└── __pycache__/         # Python cache
```

## 🎮 Sử dụng

### Chat cơ bản
Nhập câu hỏi và nhận phản hồi streaming từ AI.

### Web Search
```
Tìm kiếm tin tức mới nhất về AI
```

### Calculator
```
Tính đạo hàm của x^2 + 3x + 2
Giải phương trình x^2 - 4 = 0
```

### Code Execution
```
Chạy code Python: 
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
print(fibonacci(10))
```

### Image Understanding
Upload hình ảnh và hỏi về nội dung trong ảnh.

## ⚙️ Cấu hình

Chỉnh sửa file `config.py` hoặc `.env`:

```python
# LM Studio Configuration
LM_STUDIO_URL = 'http://localhost:1234/v1/chat/completions'
LM_STUDIO_MODEL = 'vistral-7b-chat@q8'

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 3737

# RAG Parameters
RAG_MAX_TOKENS = 2000
RAG_TEMPERATURE = 0.7
```

## 🔒 Bảo mật

- ⚠️ **KHÔNG BAO GIỜ** commit file `.env` lên Git
- File `.env` đã được thêm vào `.gitignore`
- Sử dụng `.env.example` làm template cho team

## 🐛 Troubleshooting

### Lỗi kết nối LM Studio
- Kiểm tra LM Studio đã chạy chưa
- Xác nhận port 1234 đang mở
- Kiểm tra model đã load đúng chưa

### Lỗi API Keys
- Kiểm tra file `.env` tồn tại
- Xác nhận API keys hợp lệ
- Restart server sau khi thay đổi `.env`

## 📝 License

MIT License

## 👨‍💻 Author

**Phan Hiếu**
- GitHub: [@PhanHieudc37](https://github.com/PhanHieudc37)

## 🤝 Contributing

Contributions, issues và feature requests đều được chào đón!

---

⭐ Nếu project này hữu ích, hãy cho một Star nhé!
