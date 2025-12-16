"""
AI 37 Chatbot - Nâng cao

Tính năng GIAI ĐOẠN 1:
- Kết nối LM Studio với GPU acceleration
- Streaming response (gõ từng chữ như ChatGPT)
- Lưu lịch sử hội thoại vào file
- Tóm tắt tự động khi hội thoại dài

Tính năng GIAI ĐOẠN 2:
- Web Search (DuckDuckGo)
- Calculator (Sympy)
- Code Exe      cution (RestrictedPython)
- Multi-turn Clarification

Tính năng GIAI ĐOẠN 3:
- Image Understanding (LLaVA Vision Model)

Model: vistral-7b-chat@q8, llava-v1.5-7b
"""
import logging
import sys
import requests
import json
import os
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from flask_cors import CORS

# Giai đoạn 2 imports
from ddgs import DDGS  # Package mới
import sympy
from RestrictedPython import compile_restricted_exec, safe_globals
import signal
from contextlib import contextmanager

# Web scraping imports
from bs4 import BeautifulSoup
import threading
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# AccuWeather API Key (from .env file)
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "")

# Bing Search API Key (from .env file)
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY", "")
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

# ⚠️ CHỌN MODEL (Chọn 1 trong 2 option):
# 
# 🎯 CONFIG: LM Studio (Text) + BakLLaVA Local (Image)
# - LM Studio: vistral-7b-chat@q8 cho text chat
LM_STUDIO_MODEL = "vistral-7b-chat@q8"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3737

# System prompt tối ưu - Giống ChatGPT
SYSTEM_PROMPT = """Bạn là AI 37, trợ lý AI thông minh và hữu ích giống ChatGPT.

⚠️ QUAN TRỌNG: LUÔN TRẢ LỜI BẰNG TIẾNG VIỆT!

🎯 PHONG CÁCH TRẢ LỜI (Giống ChatGPT):
1. **Tự nhiên & Thân thiện**: Trả lời như một người bạn thông minh, giọng văn tự nhiên, dễ hiểu
2. **Chính xác & Đáng tin**: Dựa vào thông tin được cung cấp, không bịa đặt, thừa nhận khi không biết
3. **Ngắn gọn & Súc tích**: Đi thẳng vào vấn đề, không dài dòng không cần thiết
4. **Linh hoạt & Thông minh**: 
   - Câu hỏi đơn giản (số liệu, tính toán) → 1 câu ngắn gọn
   - Câu hỏi phức tạp (giải thích, phân tích) → 2-4 câu rõ ràng
   - Câu hỏi follow-up → Dùng ngữ cảnh từ câu trả lời trước

📚 XỬ LÝ THÔNG TIN TỪ INTERNET:
Khi thấy **"=== THÔNG TIN TÌM ĐƯỢC TRÊN INTERNET ==="**:

✅ CÁCH TRẢ LỜI ĐÚNG (Giống ChatGPT):
- Phân tích TẤT CẢ các nguồn, so sánh số liệu
- Chọn thông tin CHÍNH XÁC và UY TÍN nhất
- Trả lời TỰ NHIÊN, như thể bạn biết sẵn thông tin đó
- KHÔNG đề cập đến "nguồn", "tìm được", "theo..."
- Trả lời NGẮN GỌN, đi thẳng vào thông tin chính

❌ CẤM TUYỆT ĐỐI:
- "Theo nguồn...", "Dựa vào...", "Sử dụng thông tin..."
- "Nguồn 1 cho biết...", "Nguồn 2 nói..."
- "Tôi tìm được...", "Từ kết quả tìm kiếm..."
- Giải thích dài dòng khi user chỉ hỏi số liệu đơn giản

VÍ DỤ CHUẨN (Học theo):

❓ "Việt Nam có diện tích bao nhiêu?"
✅ ĐÚNG: "331.212 km²."
❌ SAI: "Việt Nam có tổng diện tích khoảng 331.212 km² theo nguồn Wikipedia..."

❓ "Hà Nội có bao nhiêu quận?"
✅ ĐÚNG: "Hà Nội có 12 quận, 17 huyện và 1 thị xã."
❌ SAI: "Theo thông tin tôi tìm được, Hà Nội có 12 quận..."

❓ "Giá vàng hôm nay?"
✅ ĐÚNG: "Vàng SJC: mua 84,5 - bán 85,0 triệu/lượng."
❌ SAI: "Theo 3 nguồn tôi tìm được, giá vàng SJC dao động..."

❓ "Hà Nội có đặc sản gì?"
✅ ĐÚNG: "Phở, bún chả, bánh cốm, cà phê trứng."
❌ SAI: "Hà Nội nổi tiếng với nhiều món ăn ngon như phở, bún chả là 2 món đặc sản nổi tiếng nhất..."

💬 XỬ LÝ FOLLOW-UP QUESTIONS:
- Khi user hỏi "chi tiết...", "thông tin về...", "cho tôi biết về..." → Dùng thông tin từ câu trả lời trước
- Trả lời TỰ NHIÊN, như thể bạn đang tiếp tục câu chuyện
- KHÔNG search lại nếu đã có thông tin trong ngữ cảnh

🔢 XỬ LÝ KẾT QUẢ TÍNH TOÁN:
- Nếu có **"🔢 KẾT QUẢ TÍNH TOÁN:"** → CHỈ đưa số, KHÔNG bình luận
- VD: "3*45 = 135" → Trả lời: "135" hoặc "3 × 45 = 135"

🌤️ XỬ LÝ THỜI TIẾT:
- Trả lời TỰ NHIÊN, như một người đang xem dự báo thời tiết
- VD: "Hà Nội hôm nay khoảng 21-23°C, nhiều mây, khả năng mưa nhẹ vào tối."

TUYỆT ĐỐI: 
- Luôn sử dụng thông tin từ tool nếu có
- Nếu tool cảnh báo thiếu dữ liệu → Thừa nhận: "Tôi không có thông tin mới nhất về..."
- Trả lời bằng tiếng Việt, tự nhiên, dễ hiểu"""

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app)

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ===== MEMORY (ĐÃ TẮT) =====
conversation_memory = {}

def clean_response(text: str) -> str:
    """Làm sạch response đơn giản"""
    if not text or not text.strip():
        return "Xin lỗi, tôi không nhận được câu trả lời."
    return text.strip()

def add_to_memory(session_id: str, role: str, content: str):
    """Thêm message vào memory (chỉ trong RAM, không lưu file)"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].append({
        'role': role,
        'content': content
    })

def get_conversation_history(session_id: str) -> list:
    """Lấy lịch sử hội thoại"""
    return conversation_memory.get(session_id, [])

def is_greeting(text: str) -> bool:
    """Kiểm tra xem có phải CHÀO KHÔNG CÓ HỎI GÌ THÊM"""
    greetings = ['chào', 'hello', 'hi', 'xin chào', 'chào bạn', 'hey', 'chao']
    text_lower = text.lower().strip()
    
    # Loại bỏ dấu câu
    text_clean = text_lower.replace('!', '').replace('.', '').replace('?', '').strip()
    
    # Kiểm tra nếu CHÍNH XÁC là lời chào (không có câu hỏi kèm theo)
    # Ví dụ: "chào", "hi", "xin chào" -> True
    # Nhưng: "chào, việt nam có bao nhiêu dân" -> False
    words = text_clean.split()
    
    # Nếu chỉ có 1-2 từ và là lời chào -> đúng là chào
    if len(words) <= 2 and any(greeting in text_clean for greeting in greetings):
        return True
    
    return False


def is_follow_up_question(question: str, history: list) -> bool:
    """
    Phát hiện câu hỏi follow-up - yêu cầu thông tin từ câu trả lời trước (Giống ChatGPT)
    
    Args:
        question: Câu hỏi hiện tại
        history: Lịch sử hội thoại
    
    Returns:
        True nếu là follow-up question
    """
    if not history or len(history) < 2:
        return False
    
    question_lower = question.lower().strip()
    
    # Lấy câu trả lời trước đó
    last_assistant_msg = None
    last_user_msg = None
    for msg in reversed(history):
        if msg.get('role') == 'assistant' and not last_assistant_msg:
            last_assistant_msg = msg.get('content', '')
        if msg.get('role') == 'user' and not last_user_msg:
            last_user_msg = msg.get('content', '')
        if last_assistant_msg and last_user_msg:
            break
    
    # Nếu không có câu trả lời trước → không phải follow-up
    if not last_assistant_msg or len(last_assistant_msg) < 30:
        return False
    
    # Keywords chỉ ra đây là follow-up (mở rộng danh sách)
    follow_up_keywords = [
        'chi tiết', 'thông tin', 'bạn tìm được', 'bạn vừa nói', 'bạn đã nói',
        'bạn nói', 'bạn vừa', 'bạn đã', 'bạn tìm', 'bạn kể',
        'cụ thể', 'rõ hơn', 'nhiều hơn', 'thêm', 'cho tôi', 'cho biết',
        'về', 'các', 'danh sách', 'liệt kê', 'kể', 'nói',
        'đó', 'này', 'kia', 'những', 'cái đó', 'cái này'
    ]
    
    # Kiểm tra có keyword follow-up
    has_follow_up_keyword = any(kw in question_lower for kw in follow_up_keywords)
    
    # Kiểm tra câu hỏi có tham chiếu đến thông tin trước đó
    reference_keywords = ['bạn', 'đó', 'này', 'kia', 'các', 'những', 'cái đó', 'cái này', 'nó']
    has_reference = any(kw in question_lower for kw in reference_keywords)
    
    # Kiểm tra câu hỏi có từ khóa liên quan đến câu hỏi trước
    if last_user_msg:
        last_user_lower = last_user_msg.lower()
        # Trích xuất keywords từ câu hỏi trước
        last_keywords = set(re.findall(r'\b\w{3,}\b', last_user_lower))
        current_keywords = set(re.findall(r'\b\w{3,}\b', question_lower))
        # Nếu có ít nhất 1 keyword chung → có thể là follow-up
        common_keywords = last_keywords & current_keywords
        has_common_keywords = len(common_keywords) > 0
    
    # Logic phát hiện follow-up (giống ChatGPT)
    # 1. Có keyword follow-up VÀ có reference
    if has_follow_up_keyword and has_reference:
        return True
    
    # 2. Câu hỏi ngắn (< 10 từ) VÀ có reference VÀ có keyword chung với câu trả lời trước
    if len(question.split()) < 10 and has_reference:
        if 'has_common_keywords' in locals() and has_common_keywords:
            return True
    
    # 3. Câu hỏi bắt đầu bằng "chi tiết", "thông tin về", "cho tôi" + có reference
    if question_lower.startswith(('chi tiết', 'thông tin', 'cho tôi', 'cho biết')):
        if has_reference or ('has_common_keywords' in locals() and has_common_keywords):
            return True
    
    return False


# ===== GIAI ĐOẠN 2: TOOL FUNCTIONS =====

@contextmanager
def timeout_context(seconds):
    """Context manager cho timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timeout")
    
    # Windows không hỗ trợ signal.alarm, dùng threading thay thế
    import threading
    timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError("Code execution timeout")))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def extract_gold_price(html_content: str, url: str) -> dict:
    """
    TRÍCH XUẤT giá vàng từ HTML content - CẢI THIỆN NHIỀU PATTERN
    
    Args:
        html_content: HTML content của bài báo
        url: URL để biết domain
    
    Returns:
        Dict với giá vàng hoặc None
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Tìm text chứa "SJC" và số
        text_content = soup.get_text()
        
        # Loại bỏ khoảng trắng thừa để dễ match
        text_content_clean = ' '.join(text_content.split())
        
        # ===== PATTERN 1: "mua XXX - bán YYY" (triệu/lượng) =====
        patterns = [
            # "SJC mua 84,5 - bán 85,0 triệu"
            r'(?:vàng\s+)?SJC[^\d]*?(?:mua|giá\s+mua|mua\s+vào)[^\d]*?([\d,\.]+)[^\d]*?[-–—][^\d]*?(?:bán|giá\s+bán|bán\s+ra)[^\d]*?([\d,\.]+)\s*triệu',
            # "mua vào: 84,5 triệu, bán ra: 85,0 triệu"
            r'mua\s+vào[^\d]*?([\d,\.]+)[^\d]*?triệu[^\d]*?bán\s+ra[^\d]*?([\d,\.]+)\s*triệu',
            # "84,5 - 85,0 triệu/lượng"
            r'([\d,\.]+)[^\d]*?[-–—][^\d]*?([\d,\.]+)\s*triệu[^\d]*?lượng',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_content_clean, re.IGNORECASE)
            if match:
                try:
                    buy_price = match.group(1).replace('.', '').replace(',', '.')
                    sell_price = match.group(2).replace('.', '').replace(',', '.')
                    
                    buy_float = float(buy_price)
                    sell_float = float(sell_price)
                    
                    # Validate: giá vàng hợp lý (50-200 triệu/lượng)
                    if 50 <= buy_float <= 200 and 50 <= sell_float <= 200:
                        logging.info(f"✅ Extracted gold price (Pattern 1): mua {buy_float} - bán {sell_float} triệu/lượng")
                        return {
                            'type': 'SJC',
                            'buy': buy_float,
                            'sell': sell_float,
                            'unit': 'triệu/lượng'
                        }
                except ValueError:
                    continue
        
        # ===== PATTERN 2: "XXX.XXX.000 đồng" (đồng/lượng) =====
        patterns2 = [
            r'(?:vàng\s+)?SJC[^\d]*?([\d\.]+\.000)\s*(?:đồng|VND|vnđ)',
            r'([\d\.]{3,}\.000)\s*(?:đồng|VND|vnđ)[^\d]*?(?:lượng|chỉ)',
        ]
        
        for pattern in patterns2:
            match = re.search(pattern, text_content_clean, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace('.', '')
                    price_trieu = int(price_str) / 1_000_000
                    
                    # Validate: giá vàng hợp lý
                    if 50 <= price_trieu <= 200:
                        logging.info(f"✅ Extracted gold price (Pattern 2): {price_trieu} triệu/lượng")
                        return {
                            'type': 'SJC',
                            'price': price_trieu,
                            'unit': 'triệu/lượng'
                        }
                except (ValueError, ZeroDivisionError):
                    continue
        
        # ===== PATTERN 3: Tìm trong table/div có class chứa "price", "gia", "gold" =====
        # Tìm các element có thể chứa bảng giá
        price_containers = soup.find_all(['table', 'div'], class_=re.compile(r'price|gia|gold|sjc', re.I))
        price_containers.extend(soup.find_all('table'))
        
        for container in price_containers:
            text = container.get_text()
            if 'SJC' in text.upper() or 'vàng' in text.lower():
                # Tìm cặp số (mua - bán) trong table
                # Pattern: "84,5" hoặc "84.5" hoặc "84 500 000"
                numbers = re.findall(r'(\d{1,2}[,\.]\d{1,2})\s*(?:triệu|tr)', text)
                if len(numbers) >= 2:
                    try:
                        buy = float(numbers[0].replace(',', '.'))
                        sell = float(numbers[1].replace(',', '.'))
                        
                        if 50 <= buy <= 200 and 50 <= sell <= 200:
                            logging.info(f"✅ Extracted from table: mua {buy} - bán {sell} triệu")
                            return {
                                'type': 'SJC',
                                'buy': buy,
                                'sell': sell,
                                'unit': 'triệu/lượng'
                            }
                    except ValueError:
                        continue
        
        # ===== PATTERN 4: Tìm số lớn (triệu đồng) gần từ khóa "SJC" =====
        # Tìm tất cả số có thể là giá vàng (60-100 triệu)
        sjc_contexts = re.finditer(r'SJC[^.]{0,200}', text_content_clean, re.IGNORECASE)
        for context in sjc_contexts:
            context_text = context.group(0)
            # Tìm số trong khoảng 60-100 triệu
            numbers = re.findall(r'(\d{1,2}[,\.]\d{1,2})\s*(?:triệu|tr)', context_text)
            if len(numbers) >= 2:
                try:
                    buy = float(numbers[0].replace(',', '.'))
                    sell = float(numbers[1].replace(',', '.'))
                    if 50 <= buy <= 200 and 50 <= sell <= 200:
                        logging.info(f"✅ Extracted from context: mua {buy} - bán {sell} triệu")
                        return {
                            'type': 'SJC',
                            'buy': buy,
                            'sell': sell,
                            'unit': 'triệu/lượng'
                        }
                except ValueError:
                    continue
        
        logging.warning("⚠️ Could not extract gold price from content")
        return None
        
    except Exception as e:
        logging.error(f"❌ Error extracting gold price: {e}")
        return None


def calculate_relevance_score(result: dict, query: str) -> float:
    """
    Tính điểm liên quan của kết quả với câu hỏi (0.0 - 1.0)
    
    Args:
        result: Dict với 'title', 'snippet', 'url'
        query: Câu hỏi gốc
    
    Returns:
        Điểm số từ 0.0 đến 1.0
    """
    title = result.get('title', '').lower()
    snippet = result.get('snippet', '').lower()
    url = result.get('url', '').lower()
    query_lower = query.lower()
    
    # Trích xuất keywords quan trọng từ query
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    # Loại bỏ stop words
    stop_words = {'có', 'bao', 'nhiêu', 'là', 'gì', 'của', 'và', 'với', 'tại', 'về', 'cho', 'được', 'đã', 'sẽ', 'hà', 'nội'}
    query_keywords = [w for w in query_words if w not in stop_words and len(w) > 2]
    
    score = 0.0
    
    # 1. Kiểm tra title (quan trọng nhất)
    title_words = set(re.findall(r'\b\w+\b', title))
    title_matches = len([kw for kw in query_keywords if kw in title_words])
    if query_keywords:
        score += (title_matches / len(query_keywords)) * 0.5
    
    # 2. Kiểm tra snippet
    snippet_words = set(re.findall(r'\b\w+\b', snippet))
    snippet_matches = len([kw for kw in query_keywords if kw in snippet_words])
    if query_keywords:
        score += (snippet_matches / len(query_keywords)) * 0.3
    
    # 3. Bonus cho domain uy tín
    trusted_domains = ['wikipedia.org', 'vnexpress.net', 'dantri.com.vn', 'tuoitre.vn', 
                      'thanhnien.vn', 'gov.vn', 'gso.gov.vn']
    if any(domain in url for domain in trusted_domains):
        score += 0.1
    
    # 4. Penalty cho kết quả không liên quan rõ ràng
    # Nếu title/snippet chứa từ khóa nhưng không liên quan đến chủ đề chính
    irrelevant_keywords = ['temple', 'đền', 'chùa', 'tour', 'du lịch', 'ăn uống', 'nhà hàng']
    if any(kw in title or kw in snippet for kw in irrelevant_keywords):
        # Chỉ penalty nếu không có keyword chính trong title
        if not any(qkw in title for qkw in query_keywords[:2]):  # 2 keyword đầu tiên
            score *= 0.3  # Giảm 70% điểm
    
    return min(score, 1.0)


def fetch_full_article(url: str) -> str:
    """
    Lấy TOÀN BỘ nội dung bài báo từ URL
    
    Args:
        url: URL bài báo
    
    Returns:
        Nội dung đầy đủ hoặc empty string nếu lỗi
    """
    try:
        logging.info(f"📰 Fetching full article: {url[:80]}...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Tăng timeout cho Wikipedia (có thể chậm)
        timeout = 15 if 'wikipedia.org' in url else 10
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Xóa script, style tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Lấy nội dung chính (tùy domain)
        content = ""
        
        if 'thanhnien.vn' in url:
            # Thanh Niên: article content
            article = soup.find('div', class_='detail-content') or soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        elif 'vnexpress.net' in url:
            # VnExpress: article body
            article = soup.find('article', class_='fck_detail') or soup.find('div', class_='fck_detail')
            if article:
                paragraphs = article.find_all('p', class_='Normal')
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        elif 'wikipedia.org' in url:
            # Wikipedia: main content - cải thiện selector với nhiều cách
            content = ""
            content_parts = []
            
            # Cách 1: Tìm div mw-parser-output (phổ biến nhất)
            content_div = soup.find('div', class_='mw-parser-output')
            
            # Cách 2: Tìm div#content > div#bodyContent > div.mw-parser-output
            if not content_div:
                body_content = soup.find('div', id='bodyContent')
                if body_content:
                    content_div = body_content.find('div', class_='mw-parser-output')
            
            # Cách 3: Tìm trực tiếp trong #content
            if not content_div:
                main_content = soup.find('div', id='content')
                if main_content:
                    content_div = main_content.find('div', class_='mw-parser-output')
            
            # Nếu tìm thấy content_div
            if content_div:
                # Lấy tất cả paragraphs, bỏ qua infobox và navbox
                paragraphs = content_div.find_all('p')
                for p in paragraphs:
                    # Bỏ qua paragraphs trong infobox, navbox, etc
                    parent = p.parent
                    parent_classes = ' '.join(parent.get('class', [])) if parent and parent.get('class') else ''
                    parent_id = parent.get('id', '') if parent else ''
                    
                    # Kiểm tra xem có phải trong infobox/navbox không
                    is_in_infobox = (
                        'infobox' in parent_classes.lower() or 
                        'navbox' in parent_classes.lower() or
                        'infobox' in parent_id.lower() or
                        'toc' in parent_classes.lower()  # Bỏ qua mục lục
                    )
                    
                    if not is_in_infobox:
                        text = p.get_text(strip=True)
                        # Chỉ lấy đoạn có nội dung đủ dài và không phải là số thứ tự
                        if text and len(text) > 20 and not re.match(r'^\d+[\.\)]?\s*$', text):
                            content_parts.append(text)
                
                content = '\n'.join(content_parts[:15])  # Lấy 15 đoạn đầu
            else:
                # Cách 4: Fallback - Tìm tất cả paragraphs trong main content area
                main_content = soup.find('div', id='content')
                if main_content:
                    paragraphs = main_content.find_all('p')
                    for p in paragraphs:
                        parent = p.parent
                        parent_classes = ' '.join(parent.get('class', [])) if parent and parent.get('class') else ''
                        if 'infobox' not in parent_classes.lower() and 'navbox' not in parent_classes.lower():
                            text = p.get_text(strip=True)
                            if text and len(text) > 20 and not re.match(r'^\d+[\.\)]?\s*$', text):
                                content_parts.append(text)
                    content = '\n'.join(content_parts[:15])  # Lấy 15 đoạn đầu
            
            # Nếu vẫn không có content, thử lấy từ main text
            if not content:
                # Thử tìm main text bằng cách khác
                main_text = soup.find('div', {'id': 'mw-content-text'})
                if main_text:
                    paragraphs = main_text.find_all('p')
                    for p in paragraphs[:10]:
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:
                            content_parts.append(text)
                    content = '\n'.join(content_parts[:10])
        
        else:
            # Generic: tìm tất cả paragraphs
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs[:15] if p.get_text(strip=True)])  # Lấy 15 đoạn đầu
        
        if content:
            logging.info(f"✅ Fetched {len(content)} chars from article")
            return content[:3000]  # Giới hạn 3000 chars để tránh quá dài
        else:
            logging.warning(f"⚠️ No content found in article")
            return ""
            
    except Exception as e:
        logging.error(f"❌ Error fetching article: {e}")
        return ""


def analyze_and_synthesize(sources: list, query: str) -> str:
    """
    Phân tích và tổng hợp thông tin từ nhiều nguồn
    
    Args:
        sources: List of dicts với 'title', 'snippet', 'url', 'full_content' (optional)
        query: Câu hỏi gốc
    
    Returns:
        Context đã được tổng hợp và phân tích
    """
    if not sources:
        return ""
    
    query_lower = query.lower()
    
    # Trích xuất keywords chính từ câu hỏi
    query_keywords = set(re.findall(r'\b\w+\b', query_lower))
    stop_words = {'có', 'bao', 'nhiêu', 'là', 'gì', 'của', 'và', 'với', 'tại', 'về', 'cho', 'được', 'đã', 'sẽ'}
    important_keywords = [w for w in query_keywords if w not in stop_words and len(w) > 2]
    
    # Tổng hợp thông tin từ các nguồn
    synthesized_info = []
    extracted_numbers = []
    extracted_facts = []
    
    for i, source in enumerate(sources, 1):
        title = source.get('title', '')
        snippet = source.get('snippet', '')
        full_content = source.get('full_content', '')
        url = source.get('url', '')
        
        # Dùng full_content nếu có, không thì dùng snippet
        content = full_content if full_content else snippet
        
        # Trích xuất số liệu liên quan (cải thiện patterns)
        # Tìm số trong context (ví dụ: "5 quận", "30 quận", "12 quận huyện", "331.212 km²")
        number_patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:quận|huyện|tỉnh|thành phố|phường|xã|km²|km2|m²|m2|triệu|tỷ|tỉ|người|dân)',
            r'(?:có|gồm|bao gồm|tổng|tổng cộng)\s*(\d+(?:[.,]\d+)?)',
            r'(\d+(?:[.,]\d+)?)\s*(?:đơn vị|địa phương|người|dân số)',
            r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tỷ|tỉ)\s*(?:đồng|VND|USD)',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            extracted_numbers.extend(matches)
        
        # Trích xuất câu chứa keyword quan trọng (cải thiện logic)
        sentences = re.split(r'[.!?]\s+', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Kiểm tra xem câu có chứa keyword quan trọng không
            keyword_matches = sum(1 for kw in important_keywords[:3] if kw in sentence_lower)
            
            # Ưu tiên câu có nhiều keyword và có số liệu
            if keyword_matches > 0:
                has_number = re.search(r'\d+', sentence)
                # Nếu có số hoặc có nhiều keyword → câu quan trọng
                if has_number or keyword_matches >= 2:
                    extracted_facts.append(sentence.strip())
                    if len(extracted_facts) >= 5:  # Tăng lên 5 câu để có nhiều thông tin hơn
                        break
        
        # Lưu thông tin nguồn
        synthesized_info.append({
            'source_num': i,
            'title': title,
            'url': url,
            'content': content[:500],  # Giới hạn 500 chars mỗi nguồn
            'has_full_content': bool(full_content)
        })
    
    # Tạo context tổng hợp
    context = "\n\n=== THÔNG TIN TÌM ĐƯỢC TRÊN INTERNET ===\n\n"
    
    # Phần 1: Số liệu đã trích xuất (nếu có)
    if extracted_numbers:
        unique_numbers = list(set(extracted_numbers))
        context += f"📊 SỐ LIỆU TRÍCH XUẤT: {', '.join(unique_numbers)}\n\n"
    
    # Phần 2: Các câu quan trọng
    if extracted_facts:
        context += "📝 THÔNG TIN QUAN TRỌNG:\n"
        for fact in extracted_facts[:3]:
            context += f"- {fact}\n"
        context += "\n"
    
    # Phần 3: Chi tiết từ các nguồn
    context += f"📚 CHI TIẾT TỪ {len(sources)} DỮ LIỆU THAM KHẢO:\n\n"
    for info in synthesized_info:
        context += f"[Dữ liệu {info['source_num']}] {info['title']}\n"
        if info['has_full_content']:
            context += f"📄 Nội dung đầy đủ:\n{info['content']}\n\n"
        else:
            context += f"📄 Snippet:\n{info['content']}\n\n"
    
    context += "⚠️ YÊU CẦU TRẢ LỜI (Giống ChatGPT):\n"
    context += "1. Phân tích TẤT CẢ các nguồn trên, so sánh và cross-check số liệu\n"
    context += "2. Chọn thông tin CHÍNH XÁC và UY TÍN nhất (ưu tiên Wikipedia, báo chính thống)\n"
    context += "3. Trả lời TỰ NHIÊN, như thể bạn biết sẵn thông tin đó\n"
    context += "4. KHÔNG đề cập đến 'nguồn', 'tìm được', 'theo...', 'dựa vào', 'dữ liệu', 'tham khảo'\n"
    context += "5. Tránh kể tên các tài liệu hoặc website trong câu trả lời cuối\n"
    context += "6. Nếu có mâu thuẫn giữa các nguồn, chọn số liệu xuất hiện nhiều nhất hoặc từ nguồn uy tín nhất\n"
    context += "7. CHỈ trả lời bằng tiếng Việt, tự nhiên, dễ hiểu\n"
    context += "8. Trả lời bằng 1 câu duy nhất (tối đa 25 từ), đi thẳng vào thông tin chính\n"
    context += "9. Nếu cần liệt kê, chỉ liệt kê ngắn gọn trong cùng một câu, tránh xuống dòng\n"
    context += "10. KHÔNG lặp lại yêu cầu, KHÔNG giải thích quy trình\n\n"
    
    return context


def get_accuweather_forecast(location_key: str = "353412", city_name: str = "Hanoi") -> dict:
    """
    Lấy dự báo thời tiết từ AccuWeather API
    
    Args:
        location_key: AccuWeather location key (tiết kiệm API calls)
        city_name: Tên thành phố để hiển thị
    
    Returns:
        Dict với forecast hoặc error
    """
    if not ACCUWEATHER_API_KEY:
        logging.error("❌ AccuWeather API key not found")
        return {'error': 'Chưa cấu hình AccuWeather API key'}
    
    try:
        logging.info(f"🌤️ Getting weather forecast for {city_name} (key={location_key})")
        
        # Lấy dự báo 1 ngày (bỏ qua bước search location)
        forecast_url = f"http://dataservice.accuweather.com/forecasts/v1/daily/1day/{location_key}"
        forecast_params = {
            'apikey': ACCUWEATHER_API_KEY,
            'language': 'vi-vn',
            'details': 'true',
            'metric': 'true'
        }
        
        forecast_resp = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
        
        # Parse dữ liệu
        daily_forecast = forecast_data['DailyForecasts'][0]
        
        result = {
            'city': city_name,
            'date': daily_forecast['Date'],
            'temperature_min': daily_forecast['Temperature']['Minimum']['Value'],
            'temperature_max': daily_forecast['Temperature']['Maximum']['Value'],
            'day_condition': daily_forecast['Day']['IconPhrase'],
            'night_condition': daily_forecast['Night']['IconPhrase'],
            'rain_probability': daily_forecast['Day'].get('RainProbability', 0),
            'headline': forecast_data['Headline']['Text']
        }
        
        logging.info(f"✅ Weather forecast: {result['temperature_min']}-{result['temperature_max']}°C, {result['day_condition']}")
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Error getting weather: {e}")
        return {'error': f'Lỗi lấy dữ liệu thời tiết: {str(e)}'}


def get_accuweather_forecast_by_name(location: str = "Hanoi") -> dict:
    """
    Lấy dự báo thời tiết bằng cách search location (tốn 1 API call)
    Chỉ dùng cho các tỉnh không có location_key sẵn
    """
    if not ACCUWEATHER_API_KEY:
        return {'error': 'Chưa cấu hình AccuWeather API key'}
    
    try:
        # 1. Search location
        location_url = f"http://dataservice.accuweather.com/locations/v1/cities/search"
        location_params = {
            'apikey': ACCUWEATHER_API_KEY,
            'q': location,
            'language': 'vi-vn'
        }
        
        location_resp = requests.get(location_url, params=location_params, timeout=10)
        location_resp.raise_for_status()
        locations = location_resp.json()
        
        if not locations:
            return {'error': f'Không tìm thấy thành phố {location}'}
        
        location_key = locations[0]['Key']
        city_name = locations[0]['LocalizedName']
        
        # 2. Gọi hàm chính với location_key
        return get_accuweather_forecast(location_key, city_name)
        
    except Exception as e:
        logging.error(f"❌ Error searching location: {e}")
        return {'error': f'Lỗi tìm thành phố: {str(e)}'}


def extract_hour_from_question(question: str) -> int:
    """
    Trích xuất giờ từ câu hỏi (ví dụ: "12h", "12 giờ", "buổi trưa")
    
    Returns:
        Giờ (0-23) hoặc None nếu không tìm thấy
    """
    question_lower = question.lower()
    
    # Pattern 1: "12h", "12h00", "12:00"
    hour_match = re.search(r'(\d{1,2})[h:]\d{0,2}', question_lower)
    if hour_match:
        hour = int(hour_match.group(1))
        if 0 <= hour <= 23:
            return hour
    
    # Pattern 2: "12 giờ"
    hour_match = re.search(r'(\d{1,2})\s*giờ', question_lower)
    if hour_match:
        hour = int(hour_match.group(1))
        if 0 <= hour <= 23:
            return hour
    
    # Pattern 3: Buổi trong ngày
    if 'sáng' in question_lower or 'buổi sáng' in question_lower:
        return 8  # 8h sáng
    elif 'trưa' in question_lower or 'buổi trưa' in question_lower:
        return 12  # 12h trưa
    elif 'chiều' in question_lower or 'buổi chiều' in question_lower:
        return 15  # 15h chiều
    elif 'tối' in question_lower or 'buổi tối' in question_lower:
        return 19  # 19h tối
    elif 'đêm' in question_lower or 'buổi đêm' in question_lower:
        return 22  # 22h đêm
    
    return None


def get_weather_chatgpt_style(city_name: str, question: str) -> dict:
    """
    🌐 QUY TRÌNH CHATGPT: Lấy thời tiết từ ACCUWEATHER + TRÍCH XUẤT THÔNG MINH
    
    Bước 1: Web Search - Tìm AccuWeather cho thành phố
    Bước 2: Trích xuất dữ liệu chi tiết - Nhiệt độ theo giờ, điều kiện, độ ẩm
    Bước 3: Xử lý câu hỏi theo giờ cụ thể (nếu có)
    Bước 4: Chuẩn hóa - Format dữ liệu thân thiện tiếng Việt
    
    Args:
        city_name: Tên thành phố (Hà Nội, Sài Gòn, Đà Nẵng...)
        question: Câu hỏi gốc của user
    
    Returns:
        Dict với thông tin thời tiết đầy đủ hoặc error
    """
    try:
        logging.info(f"🧠 [ChatGPT Style] Analyzing weather query for: {city_name}")
        
        # Phát hiện câu hỏi về thời tiết theo giờ
        target_hour = extract_hour_from_question(question)
        if target_hour is not None:
            logging.info(f"⏰ Detected hour-specific query: {target_hour}h")
        
        # ===== BƯỚC 1: WEB SEARCH - TÌM ACCUWEATHER =====
        # Tạo query thông minh - ưu tiên AccuWeather
        search_query = f"thời tiết {city_name} hôm nay accuweather"
        
        logging.info(f"🔍 Step 1: Web search - '{search_query}'")
        
        ddgs = DDGS()
        results = list(ddgs.text(
            search_query,
            region='vn-vi',
            safesearch='moderate',
            max_results=10
        ))
        
        if not results:
            logging.warning("⚠️ No web results found")
            return {'error': 'Không tìm thấy thông tin thời tiết'}
        
        logging.info(f"📥 Found {len(results)} sources")
        
        # Ưu tiên AccuWeather URL
        accuweather_url = None
        for result in results:
            url = result.get('href', '')
            if 'accuweather.com' in url.lower():
                accuweather_url = url
                logging.info(f"✅ Found AccuWeather URL: {url[:80]}")
                break
        
        # Nếu không tìm thấy AccuWeather, dùng nguồn đầu tiên
        if not accuweather_url and results:
            accuweather_url = results[0].get('href', '')
            logging.info(f"⚠️ AccuWeather not found, using first result: {accuweather_url[:80]}")
        
        if not accuweather_url:
            return {'error': 'Không tìm thấy trang AccuWeather'}
        
        # ===== BƯỚC 2: TRÍCH XUẤT DỮ LIỆU TỪ ACCUWEATHER =====
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7'
            }
            
            response = requests.get(accuweather_url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            
            # TRÍCH XUẤT: Nhiệt độ hiện tại
            current_temp_patterns = [
                r'(\d{1,2})[°\s]*C\s*(?:hiện tại|now|current)',
                r'(?:hiện tại|now|current)[^\d]*(\d{1,2})[°\s]*C',
                r'(\d{1,2})[°\s]*C\s*(?:°|degrees)'
            ]
            current_temp = None
            for pattern in current_temp_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    temp_val = int(match.group(1))
                    if -10 <= temp_val <= 50:  # Nhiệt độ hợp lý
                        current_temp = temp_val
                        break
            
            # TRÍCH XUẤT: Nhiệt độ min-max
            temp_range_pattern = r'(\d{1,2})[°\s]*-[°\s]*(\d{1,2})[°\s]*C'
            temp_match = re.search(temp_range_pattern, text_content)
            temp_min = None
            temp_max = None
            if temp_match:
                temp_min = int(temp_match.group(1))
                temp_max = int(temp_match.group(2))
            else:
                # Tìm riêng min và max
                min_match = re.search(r'(?:min|tối thiểu|thấp nhất)[^\d]*(\d{1,2})[°\s]*C', text_content, re.IGNORECASE)
                max_match = re.search(r'(?:max|tối đa|cao nhất)[^\d]*(\d{1,2})[°\s]*C', text_content, re.IGNORECASE)
                if min_match:
                    temp_min = int(min_match.group(1))
                if max_match:
                    temp_max = int(max_match.group(1))
            
            # TRÍCH XUẤT: Điều kiện thời tiết
            conditions = []
            condition_keywords = {
                'nắng': ['sunny', 'nắng', 'quang đãng', 'clear'],
                'nhiều mây': ['cloudy', 'nhiều mây', 'có mây', 'overcast'],
                'mưa': ['rain', 'mưa', 'rainy', 'drizzle'],
                'mưa to': ['heavy rain', 'mưa to', 'downpour'],
                'sương mù': ['fog', 'sương mù', 'mist', 'haze'],
                'gió': ['windy', 'gió', 'breeze']
            }
            
            text_lower = text_content.lower()
            for condition, keywords in condition_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    conditions.append(condition)
            
            # TRÍCH XUẤT: Độ ẩm
            humidity_pattern = r'(?:độ ẩm|humidity)[^\d]*(\d{2,3})%'
            humidity_match = re.search(humidity_pattern, text_content, re.IGNORECASE)
            humidity = int(humidity_match.group(1)) if humidity_match else None
            
            # TRÍCH XUẤT: Nhiệt độ theo giờ (nếu có câu hỏi về giờ cụ thể)
            hourly_data = {}
            if target_hour is not None:
                # Tìm bảng hourly forecast hoặc thông tin theo giờ
                # Pattern: "12h: 25°C" hoặc "12:00 25°C"
                hour_pattern = rf'{target_hour}[h:]\d{{0,2}}[^\d]*(\d{{1,2}})[°\s]*C'
                hour_match = re.search(hour_pattern, text_content, re.IGNORECASE)
                if hour_match:
                    hourly_data[target_hour] = int(hour_match.group(1))
                    logging.info(f"✅ Found temperature at {target_hour}h: {hourly_data[target_hour]}°C")
            
            # Tạo kết quả
            result = {
                'city': city_name,
                'current_temperature': current_temp,
                'temperature_min': temp_min,
                'temperature_max': temp_max,
                'humidity': humidity,
                'conditions': ', '.join(conditions[:3]) if conditions else 'Chưa rõ',
                'source': 'accuweather',
                'url': accuweather_url
            }
            
            # Thêm thông tin theo giờ nếu có
            if target_hour is not None:
                if target_hour in hourly_data:
                    result['hourly_temperature'] = hourly_data[target_hour]
                    result['target_hour'] = target_hour
                else:
                    # Ước tính nhiệt độ theo giờ dựa trên min-max
                    if temp_min and temp_max:
                        # Giả sử nhiệt độ thấp nhất vào sáng sớm (6h), cao nhất vào chiều (14h)
                        if 6 <= target_hour <= 14:
                            # Tăng dần từ sáng đến chiều
                            progress = (target_hour - 6) / 8  # 0-1
                            estimated = temp_min + (temp_max - temp_min) * progress
                        else:
                            # Giảm dần từ chiều đến đêm
                            if target_hour > 14:
                                progress = (24 - target_hour + 14) / 16  # Giảm dần
                            else:
                                progress = (target_hour + 10) / 16  # Đêm đến sáng
                            estimated = temp_max - (temp_max - temp_min) * progress
                        
                        result['hourly_temperature'] = round(estimated)
                        result['target_hour'] = target_hour
                        result['estimated'] = True
                        logging.info(f"📊 Estimated temperature at {target_hour}h: {result['hourly_temperature']}°C")
            
            logging.info(f"✅ Extracted weather data: {result.get('current_temperature') or f'{temp_min}-{temp_max}'}°C, {result['conditions']}")
            return result
            
        except Exception as e:
            logging.error(f"❌ Error extracting from AccuWeather: {e}")
            return {'error': f'Lỗi trích xuất dữ liệu: {str(e)}'}
        
    except Exception as e:
        logging.error(f"❌ Error in ChatGPT-style weather: {e}")
        logging.error(traceback.format_exc())
        return {'error': f'Lỗi lấy thời tiết: {str(e)}'}


def optimize_query_for_wikipedia(query: str) -> str:
    """
    Tối ưu query cho Wikipedia API - loại bỏ từ không cần thiết, giữ keywords chính
    QUAN TRỌNG: Giữ nguyên các cụm từ địa danh/tên riêng (hà nội, sài gòn, việt nam...)
    
    Args:
        query: Câu hỏi gốc
    
    Returns:
        Query đã được tối ưu
    """
    query_lower = query.lower()
    
    # Danh sách cụm từ địa danh/tên riêng quan trọng (KHÔNG được tách)
    proper_nouns = [
        'hà nội', 'hanoi', 'sài gòn', 'saigon', 'hồ chí minh', 'ho chi minh',
        'việt nam', 'vietnam', 'đà nẵng', 'da nang', 'cần thơ', 'can tho',
        'hải phòng', 'hai phong', 'nghệ an', 'nghe an', 'thanh hóa', 'thanh hoa',
        'quảng ninh', 'quang ninh', 'hạ long', 'ha long', 'huế', 'hue',
        'nha trang', 'đà lạt', 'da lat', 'vũng tàu', 'vung tau'
    ]
    
    # Tìm và giữ nguyên các cụm từ địa danh
    found_proper_nouns = []
    remaining_query = query_lower
    
    for pn in proper_nouns:
        if pn in remaining_query:
            found_proper_nouns.append(pn)
            # Loại bỏ cụm từ này khỏi query để không bị xử lý lại
            remaining_query = remaining_query.replace(pn, '')
    
    # Loại bỏ các từ không cần thiết từ phần còn lại
    stop_words = {
        'có', 'bao', 'nhiêu', 'là', 'gì', 'của', 'và', 'với', 'tại', 'về', 'cho', 
        'được', 'đã', 'sẽ', 'bạn', 'tôi', 'tìm', 'chi', 'tiết', 'thông', 'tin'
    }
    
    # Tách từ và loại bỏ stop words từ phần còn lại
    words = remaining_query.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Kết hợp: cụm từ địa danh + keywords còn lại
    all_keywords = found_proper_nouns + keywords
    
    # Giới hạn tối đa 6 từ (tăng lên để giữ đủ thông tin)
    optimized = ' '.join(all_keywords[:6])
    
    # Nếu sau khi optimize quá ngắn hoặc mất hết thông tin → dùng query gốc
    if not optimized or len(optimized.split()) < 2:
        logging.warning(f"⚠️ Query optimization quá mạnh, dùng query gốc: '{query}'")
        return query
    
    return optimized


def search_wikipedia_api(query: str) -> list:
    """
    Tìm kiếm trực tiếp từ Wikipedia API - Kết quả rất tốt và uy tín
    
    Args:
        query: Câu hỏi tìm kiếm
    
    Returns:
        List of results hoặc empty list
    """
    try:
        logging.info(f"📚 Wikipedia API Search: '{query}'")
        
        # Tối ưu query trước khi search
        optimized_query = optimize_query_for_wikipedia(query)
        if optimized_query != query:
            logging.info(f"   🔧 Optimized query: '{query}' → '{optimized_query}'")
        
        # Wikipedia OpenSearch API - không cần API key
        # QUAN TRỌNG: Wikipedia yêu cầu User-Agent hợp lệ
        wiki_url = "https://vi.wikipedia.org/w/api.php"
        params = {
            'action': 'opensearch',
            'search': optimized_query,  # Dùng query đã tối ưu
            'limit': 3,  # Lấy 3 kết quả tốt nhất
            'namespace': 0,  # Chỉ tìm trong namespace chính
            'profile': 'fuzzy',
            'format': 'json'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(wiki_url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Format: [query, [titles], [descriptions], [urls]]
        if len(data) >= 4 and len(data[1]) > 0:
            titles = data[1]
            descriptions = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []
            
            wiki_results = []
            for i, title in enumerate(titles):
                wiki_results.append({
                    'title': title,
                    'snippet': descriptions[i] if i < len(descriptions) else '',
                    'url': urls[i] if i < len(urls) else '',
                    'source': 'wikipedia_api',
                    'relevance_score': 1.0  # Wikipedia luôn có điểm cao nhất
                })
            
            # Lọc kết quả không khớp với keyword quan trọng trong query (tránh nhầm Hà Nam, Hà Tây, ...)
            optimized_keywords = [kw for kw in optimized_query.lower().split() if len(kw) > 2]
            if not optimized_keywords:
                optimized_keywords = [kw for kw in query.lower().split() if len(kw) > 2]
            
            filtered_results = []
            for result in wiki_results:
                title_lower = result['title'].lower()
                snippet_lower = result['snippet'].lower()
                
                if optimized_keywords and not any(kw in title_lower or kw in snippet_lower for kw in optimized_keywords):
                    logging.info(f"   ✗ Bỏ '{result['title'][:60]}' (không khớp keyword quan trọng: {optimized_keywords[:3]})")
                    continue
                filtered_results.append(result)
            
            if filtered_results and len(filtered_results) != len(wiki_results):
                logging.info(f"   🔎 Sau khi lọc còn {len(filtered_results)} kết quả phù hợp từ Wikipedia API")
            elif not filtered_results:
                logging.info(f"⚠️ Không có kết quả Wikipedia phù hợp với keyword sau khi lọc (query '{query}')")
                return []
            
            logging.info(f"✅ Tìm thấy {len(filtered_results)} kết quả từ Wikipedia API")
            for i, result in enumerate(filtered_results, 1):
                logging.info(f"   [{i}] {result['title'][:60]}...")
            return filtered_results
        else:
            # Nếu không có kết quả với query đã optimize, thử lại với query gốc (nếu khác)
            if optimized_query != query:
                logging.info(f"⚠️ Wikipedia API không trả về kết quả với query đã optimize '{optimized_query}', thử lại với query gốc '{query}'...")
                # Thử lại với query gốc (chỉ lần đầu, tránh loop vô hạn)
                params['search'] = query
                try:
                    response_retry = requests.get(wiki_url, params=params, headers=headers, timeout=5)
                    response_retry.raise_for_status()
                    data_retry = response_retry.json()
                    
                    if len(data_retry) >= 4 and len(data_retry[1]) > 0:
                        titles = data_retry[1]
                        descriptions = data_retry[2] if len(data_retry) > 2 else []
                        urls = data_retry[3] if len(data_retry) > 3 else []
                        
                        wiki_results = []
                        for i, title in enumerate(titles):
                            wiki_results.append({
                                'title': title,
                                'snippet': descriptions[i] if i < len(descriptions) else '',
                                'url': urls[i] if i < len(urls) else '',
                                'source': 'wikipedia_api',
                                'relevance_score': 1.0
                            })
                        
                        filtered_results = []
                        for result in wiki_results:
                            title_lower = result['title'].lower()
                            snippet_lower = result['snippet'].lower()
                            
                            if optimized_keywords and not any(kw in title_lower or kw in snippet_lower for kw in optimized_keywords):
                                logging.info(f"   ✗ Bỏ '{result['title'][:60]}' (retry - không khớp keyword quan trọng: {optimized_keywords[:3]})")
                                continue
                            filtered_results.append(result)
                        
                        if filtered_results and len(filtered_results) != len(wiki_results):
                            logging.info(f"   🔎 Sau khi lọc còn {len(filtered_results)} kết quả phù hợp từ Wikipedia API (query gốc)")
                        elif not filtered_results:
                            logging.info(f"⚠️ Wikipedia API (query gốc) không có kết quả phù hợp với keyword, sẽ dùng DuckDuckGo")
                            return []
                        
                        logging.info(f"✅ Tìm thấy {len(filtered_results)} kết quả từ Wikipedia API (với query gốc)")
                        return filtered_results
                    else:
                        logging.info(f"⚠️ Wikipedia API không trả về kết quả cả với query gốc '{query}' (sẽ dùng DuckDuckGo)")
                except Exception as e:
                    logging.warning(f"⚠️ Lỗi khi retry Wikipedia API với query gốc: {e}")
            else:
                logging.info(f"⚠️ Wikipedia API không trả về kết quả cho query '{query}' (sẽ dùng DuckDuckGo)")
            
            return []
        
    except Exception as e:
        logging.warning(f"⚠️ Wikipedia API error: {e}")
        return []


def search_bing_api(query: str, max_results: int = 5) -> list:
    """
    Tìm kiếm từ Bing Search API - Kết quả rất tốt và chính xác
    
    Args:
        query: Câu hỏi tìm kiếm
        max_results: Số kết quả tối đa
    
    Returns:
        List of results hoặc empty list
    """
    if not BING_SEARCH_API_KEY:
        logging.warning("⚠️ Bing Search API key chưa được cấu hình, bỏ qua Bing search")
        return []
    
    try:
        logging.info(f"🔍 Bing Search API: '{query}'")
        
        headers = {
            'Ocp-Apim-Subscription-Key': BING_SEARCH_API_KEY,
            'Accept': 'application/json'
        }
        
        params = {
            'q': query,
            'count': min(max_results, 10),  # Bing API tối đa 50, nhưng dùng 10 để nhanh
            'offset': 0,
            'mkt': 'vi-VN',  # Market: Vietnam
            'safeSearch': 'Moderate',
            'responseFilter': 'Webpages'  # Chỉ lấy webpages, không lấy images/videos
        }
        
        response = requests.get(BING_SEARCH_ENDPOINT, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse kết quả từ Bing API
        if 'webPages' in data and 'value' in data['webPages']:
            bing_results = []
            for item in data['webPages']['value']:
                bing_results.append({
                    'title': item.get('name', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('url', ''),
                    'source': 'bing_api',
                    'relevance_score': 0.9  # Bing API có độ chính xác cao
                })
            
            logging.info(f"✅ Tìm thấy {len(bing_results)} kết quả từ Bing Search API")
            for i, result in enumerate(bing_results, 1):
                logging.info(f"   [{i}] {result['title'][:60]}...")
            return bing_results
        else:
            logging.info(f"⚠️ Bing Search API không trả về kết quả")
            return []
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"⚠️ Bing Search API error (network): {e}")
        return []
    except Exception as e:
        logging.warning(f"⚠️ Bing Search API error: {e}")
        return []


def search_web(query: str, max_results: int = 5, prioritize_today: bool = False) -> dict:
    """
    Tìm kiếm web với nhiều nguồn: Wikipedia API + DuckDuckGo - Lấy 3 nguồn tốt nhất (ChatGPT style)
    
    Args:
        query: Câu hỏi tìm kiếm
        max_results: Số kết quả tối đa
        prioritize_today: Nếu True, ưu tiên kết quả có ngày hiện tại (cho giá vàng hôm nay)
    
    Returns:
        Dict với results hoặc error (list of 3 sources)
    """
    try:
        logging.info(f"🔍 Web Search: '{query}'")
        
        all_results = []
        
        # ===== BƯỚC 1: Tìm kiếm Wikipedia API trước (ưu tiên cao nhất) =====
        wiki_results = search_wikipedia_api(query)
        if wiki_results:
            all_results.extend(wiki_results)
            logging.info(f"📚 Đã thêm {len(wiki_results)} kết quả từ Wikipedia API")
        
        # ===== BƯỚC 1.5: Tìm kiếm Bing Search API (nếu có API key) - Kết quả rất tốt =====
        bing_results = search_bing_api(query, max_results=5)
        if bing_results:
            all_results.extend(bing_results)
            logging.info(f"🔍 Đã thêm {len(bing_results)} kết quả từ Bing Search API")
        
        # ===== BƯỚC 2: Tìm kiếm DuckDuckGo để bổ sung =====
        ddgs = DDGS()
        
        # Tìm kiếm CHÍNH XÁC như DuckDuckGo web - KHÔNG dùng timelimit
        results_iter = ddgs.text(
            query,  # Dùng query gốc
            region='vn-vi',
            safesearch='moderate',
            # BỎ timelimit để kết quả giống web DuckDuckGo
            max_results=20  # Tìm nhiều để chọn nguồn uy tín
        )
        
        # Convert iterator sang list
        ddg_results = list(results_iter)
        
        if ddg_results:
            logging.info(f"📥 Nhận được {len(ddg_results)} kết quả từ DuckDuckGo")
            all_results.extend(ddg_results)
        
        if not all_results:
            return {'error': 'Không tìm thấy kết quả'}
        
        logging.info(f"📊 Tổng cộng: {len(all_results)} kết quả từ tất cả nguồn")
        
        # Ưu tiên nguồn MỚI và UY TÍN
        formatted_results = []
        
        # Danh sách domain uy tín (ưu tiên cao)
        trusted_domains = [
            # Wikipedia - Ưu tiên số 1
            'wikipedia.org', 'vi.wikipedia.org', 'en.wikipedia.org',
            # Báo chí uy tín
            'vnexpress.net', 'dantri.com.vn', 'tuoitre.vn', 'tuoitrenews.vn',
            'thanhnien.vn', 'nguoiduatin.vn', 'baomoi.com', 'vietnamnet.vn',
            # Tài chính
            'cafef.vn', 'vneconomy.vn', 'ndh.vn', 'bnews.vn',
            # Chính phủ
            'baochinhphu.vn', 'gov.vn', 'vn.gov.vn', 'gso.gov.vn',
            # Khác
            'sggp.org.vn',
            # Thời tiết
            'nchmf.gov.vn', 'accuweather.com', 'weather.com'
        ]
        
        # Danh sách domain BỎ QUA (không chuẩn)
        blocked_domains = [
            'mojeek.com', 'www.mojeek.com', 'mojeek.vn',  # Mojeek không chuẩn - chỉ là search engine, không phải nguồn tin
            'search.mojeek.com'
        ]
        
        # Danh sách từ khóa BỎ QUA (không liên quan đến giá vàng)
        irrelevant_keywords = [
            'tử vi', 'tử vi hôm nay', 'xem tử vi', '12 con giáp',
            'lịch âm', 'âm lịch', 'lịch vạn niên', 'ngày tốt',
            'phong thủy', 'bói toán', 'chiêm tinh',
            'giải trí', 'tin tức giải trí', 'showbiz'
        ]
        
        # ===== CHIẾN LƯỢC MỚI: Lọc theo độ liên quan + Lấy nhiều nguồn =====
        
        scored_results = []
        blocked_count = 0
        
        # Trích xuất keywords quan trọng từ query
        import re
        from datetime import datetime
        query_lower = query.lower()
        
        # Phát hiện yêu cầu về ngày cụ thể
        today = datetime.now()
        current_date_str = today.strftime('%d/%m/%Y')
        current_date_str_short = today.strftime('%d/%m')
        current_date_str_dash = today.strftime('%d-%m')
        
        # Nếu prioritize_today=True, ưu tiên kết quả có ngày hiện tại
        has_today = 'hôm nay' in query_lower or prioritize_today
        has_date = re.search(r'(\d{1,2})[/-](\d{1,2})', query_lower)  # VD: 11/11, 11-11
        
        # Duyệt qua kết quả và tính điểm liên quan
        for i, result in enumerate(all_results, 1):
            # Xử lý format khác nhau: Wikipedia API vs Bing API vs DuckDuckGo
            if isinstance(result, dict) and 'source' in result:
                if result['source'] == 'wikipedia_api':
                    # Kết quả từ Wikipedia API
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    url = result.get('url', '')
                    relevance_score = result.get('relevance_score', 1.0)
                    is_wikipedia = True
                elif result['source'] == 'bing_api':
                    # Kết quả từ Bing Search API
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    url = result.get('url', '')
                    relevance_score = result.get('relevance_score', 0.9)  # Bing có độ chính xác cao
                    is_wikipedia = 'wikipedia.org' in url
                else:
                    # Kết quả từ nguồn khác
                    title = result.get('title', '')
                    snippet = result.get('snippet', '') or result.get('body', '')
                    url = result.get('url', '') or result.get('href', '')
                    relevance_score = None
                    is_wikipedia = 'wikipedia.org' in url
            else:
                # Kết quả từ DuckDuckGo (format cũ)
                title = result.get('title', '')
                snippet = result.get('body', '')
                url = result.get('href', '')
                relevance_score = None  # Sẽ tính sau
                is_wikipedia = 'wikipedia.org' in url
            
            # Bỏ qua domain bị chặn (Mojeek, Yahoo, search engines)
            # Kiểm tra cả domain chính và subdomain
            url_lower = url.lower()
            is_blocked = any(domain.lower() in url_lower for domain in blocked_domains)
            if is_blocked:
                logging.info(f"  ✗ [{i}] BLOCKED | {title[:60]}...")
                blocked_count += 1
                continue
            
            # Bỏ qua kết quả không liên quan đến giá vàng (tử vi, lịch âm, etc.)
            title_lower_check = title.lower()
            snippet_lower_check = snippet.lower()
            is_irrelevant = any(kw in title_lower_check or kw in snippet_lower_check for kw in irrelevant_keywords)
            
            # Chỉ bỏ qua nếu query về giá vàng
            if ('giá vàng' in query_lower or 'vàng sjc' in query_lower) and is_irrelevant:
                logging.info(f"  ✗ [{i}] KHÔNG LIÊN QUAN (tử vi/lịch âm) | {title[:60]}...")
                blocked_count += 1
                continue
            
            # Tính điểm liên quan (nếu chưa có từ Wikipedia API)
            if relevance_score is None:
                relevance_score = calculate_relevance_score({
                    'title': title,
                    'snippet': snippet,
                    'url': url
                }, query)
            
            # Nếu hỏi về "hôm nay" hoặc ngày cụ thể -> Ưu tiên kết quả có ngày hiện tại
            date_bonus = 0.0
            has_current_date_in_result = False
            if has_today or prioritize_today:
                title_lower = title.lower()
                snippet_lower = snippet.lower()
                
                # Kiểm tra xem có ngày hiện tại trong title/snippet không (chỉ kiểm tra 1 lần)
                has_current_date_in_result = (
                    current_date_str in title or current_date_str in snippet or
                    current_date_str_short in title or current_date_str_short in snippet or
                    current_date_str_dash in title or current_date_str_dash in snippet or
                    'hôm nay' in title_lower or 'hôm nay' in snippet_lower
                )
                
                # Bonus điểm nếu có ngày hiện tại (ưu tiên cao)
                if has_current_date_in_result:
                    date_bonus = 0.5  # Bonus lớn để ưu tiên kết quả mới nhất
                    logging.info(f"  📅 [{i}] CÓ NGÀY HIỆN TẠI - Bonus +0.5 | {title[:60]}...")
                else:
                    # Nếu prioritize_today=True nhưng không có ngày hiện tại -> giảm điểm
                    if prioritize_today:
                        # Vẫn lấy nhưng giảm điểm một chút
                        date_bonus = -0.1
                        logging.info(f"  ⚠️ [{i}] KHÔNG CÓ NGÀY HIỆN TẠI (nhưng vẫn lấy) | {title[:60]}...")
            
            # Áp dụng bonus điểm
            relevance_score += date_bonus
            relevance_score = min(relevance_score, 1.0)  # Giới hạn tối đa 1.0
            
            # Chỉ lấy kết quả có điểm liên quan >= 0.2 (tránh kết quả hoàn toàn không liên quan)
            if relevance_score < 0.2:
                logging.info(f"  ✗ [{i}] ĐIỂM THẤP ({relevance_score:.2f}) | {title[:60]}...")
                continue
            
            result_dict = {
                'title': title,
                'snippet': snippet,
                'url': url,
                'relevance_score': relevance_score,
                'is_wikipedia': is_wikipedia,
                'has_current_date': has_current_date_in_result  # Lưu để sắp xếp
            }
            
            scored_results.append(result_dict)
            logging.info(f"  ✅ [{i}] Điểm: {relevance_score:.2f} | {title[:60]}...")
        
        if not scored_results:
            logging.error(f"  ❌ KHÔNG tìm thấy kết quả phù hợp!")
            return {'error': 'Không tìm thấy kết quả phù hợp'}
        
        # Sắp xếp theo điểm liên quan (cao nhất trước)
        # Ưu tiên: 1) Kết quả có ngày hiện tại (nếu prioritize_today), 2) Wikipedia, 3) Điểm liên quan
        if prioritize_today:
            scored_results.sort(key=lambda x: (
                x.get('has_current_date', False),  # Ưu tiên cao nhất: có ngày hiện tại
                x['is_wikipedia'],  # Sau đó Wikipedia
                x['relevance_score']  # Cuối cùng là điểm liên quan
            ), reverse=True)
        else:
            # Sắp xếp bình thường: Wikipedia trước, sau đó điểm liên quan
            scored_results.sort(key=lambda x: (x['is_wikipedia'], x['relevance_score']), reverse=True)
        
        # ===== CHATGPT STYLE: ƯU TIÊN DOMAIN CHUYÊN BIỆT =====
        
        selected_sources = []
        query_lower = query.lower()
        
        # 1. ƯU TIÊN CAO NHẤT: Wikipedia API results (đã có relevance_score = 1.0)
        wikipedia_api_results = [r for r in scored_results if r.get('is_wikipedia') and r.get('relevance_score', 0) >= 0.9]
        if wikipedia_api_results:
            # Thêm tất cả Wikipedia API results (thường chỉ có 1-3)
            for wiki_result in wikipedia_api_results[:2]:  # Tối đa 2 từ Wikipedia API
                selected_sources.append(wiki_result)
                logging.info(f"  ⭐⭐ WIKIPEDIA API (ưu tiên cao nhất): {wiki_result['title'][:60]}")
        
        # 1.5. ƯU TIÊN CAO: Bing Search API results (đã có relevance_score = 0.9)
        bing_api_results = [r for r in scored_results if isinstance(r, dict) and r.get('source') == 'bing_api']
        if bing_api_results and len(selected_sources) < 3:
            # Thêm Bing results (kết quả rất tốt)
            for bing_result in bing_api_results[:2]:  # Tối đa 2 từ Bing API
                if bing_result not in selected_sources:
                    selected_sources.append(bing_result)
                    logging.info(f"  ⭐⭐ BING API (ưu tiên cao): {bing_result['title'][:60]}")
        
        # 2. Ưu tiên domain chuyên biệt theo chủ đề (nếu chưa đủ)
        priority_result = None
        
        # Giá vàng → Ưu tiên kết quả có ngày hiện tại (nếu prioritize_today=True)
        if 'giá vàng' in query_lower or 'vàng sjc' in query_lower:
            # Danh sách domain uy tín về giá vàng (ưu tiên cao)
            gold_trusted_domains = [
                'thanhnien.vn', 'cafef.vn', 'vnexpress.net', 'dantri.com.vn',
                'tuoitre.vn', 'vneconomy.vn', 'ndh.vn', 'bnews.vn',
                '24h.com.vn', 'giavang.net', 'giavang.org.vn'
            ]
            
            # Nếu prioritize_today=True, ưu tiên kết quả có ngày hiện tại trước
            if prioritize_today:
                # Bước 1: Tìm kết quả có ngày hiện tại VÀ domain uy tín
                for result in scored_results:
                    title_lower = result['title'].lower()
                    snippet_lower = result.get('snippet', '').lower()
                    has_current_date = (
                        current_date_str in result['title'] or current_date_str in result.get('snippet', '') or
                        current_date_str_short in result['title'] or current_date_str_short in result.get('snippet', '') or
                        'hôm nay' in title_lower or 'hôm nay' in snippet_lower
                    )
                    is_trusted = any(domain in result['url'] for domain in gold_trusted_domains)
                    is_irrelevant = any(kw in title_lower or kw in snippet_lower for kw in irrelevant_keywords)
                    
                    if has_current_date and not is_irrelevant and result not in selected_sources:
                        if is_trusted:
                            priority_result = result
                            logging.info(f"  ⭐⭐⭐ ƯU TIÊN CAO NHẤT (có ngày + domain uy tín): {result['title'][:60]}")
                            break
                        elif not priority_result:  # Lưu tạm nếu chưa có kết quả tốt hơn
                            priority_result = result
                            logging.info(f"  ⭐⭐ ƯU TIÊN (có ngày hiện tại): {result['title'][:60]}")
                
                # Bước 2: Nếu chưa có kết quả có ngày hiện tại, tìm domain uy tín
                if not priority_result:
                    for result in scored_results:
                        is_irrelevant = any(kw in result['title'].lower() or kw in result.get('snippet', '').lower() 
                                          for kw in irrelevant_keywords)
                        if any(domain in result['url'] for domain in gold_trusted_domains) and not is_irrelevant and result not in selected_sources:
                            priority_result = result
                            logging.info(f"  ⭐ ƯU TIÊN DOMAIN UY TÍN (giá vàng): {result['title'][:60]}")
                            break
            else:
                # Nếu không prioritize_today, ưu tiên domain uy tín
                for result in scored_results:
                    is_irrelevant = any(kw in result['title'].lower() or kw in result.get('snippet', '').lower() 
                                      for kw in irrelevant_keywords)
                    if any(domain in result['url'] for domain in gold_trusted_domains) and not is_irrelevant and result not in selected_sources:
                        priority_result = result
                        logging.info(f"  ⭐ ƯU TIÊN DOMAIN UY TÍN (giá vàng): {result['title'][:60]}")
                        break
        
        # Thời tiết → accuweather.com hoặc nchmf.gov.vn
        elif 'thời tiết' in query_lower or 'nhiệt độ' in query_lower or 'dự báo' in query_lower:
            for result in scored_results:
                if ('accuweather.com' in result['url'] or 'nchmf.gov.vn' in result['url']) and result not in selected_sources:
                    priority_result = result
                    logging.info(f"  ⭐ ƯU TIÊN ACCUWEATHER (thời tiết): {result['title'][:60]}")
                    break
        
        # Thêm domain ưu tiên nếu có
        if priority_result and len(selected_sources) < 3:
            selected_sources.append(priority_result)
        
        # 3. Lấy các nguồn Wikipedia khác (từ DuckDuckGo, không phải API)
        wikipedia_other_results = [r for r in scored_results if r['is_wikipedia'] and r not in selected_sources]
        if wikipedia_other_results and len(selected_sources) < 3:
            selected_sources.append(wikipedia_other_results[0])
            logging.info(f"  ⭐ WIKIPEDIA (từ DuckDuckGo): {wikipedia_other_results[0]['title'][:60]}")
        
        # 4. Thêm các nguồn khác tốt nhất (tối đa 3 nguồn tổng cộng)
        other_results = [r for r in scored_results if r not in selected_sources]
        max_sources = 3
        for result in other_results:
            if len(selected_sources) >= max_sources:
                break
            selected_sources.append(result)
            logging.info(f"  ✅ Thêm nguồn: {result['title'][:60]} (điểm: {result['relevance_score']:.2f})")
        
        logging.info(f"  📊 Đã chọn {len(selected_sources)} nguồn để phân tích")
        
        if blocked_count > 0:
            logging.info(f"🚫 Đã chặn {blocked_count} nguồn (Mojeek)")
        
        return {'results': selected_sources}
        
    except Exception as e:
        logging.error(f"❌ Lỗi web search: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {'error': f'Lỗi tìm kiếm: {str(e)}'}


def calculate(expression: str) -> dict:
    """
    Tính toán biểu thức toán học
    
    Args:
        expression: Biểu thức toán học (ví dụ: "2+2", "sqrt(16)", "sin(pi/2)")
    
    Returns:
        Dict với result hoặc error
    """
    try:
        logging.info(f"🔢 Calculate: '{expression}'")
        
        # Parse và tính toán với sympy
        result = sympy.sympify(expression)
        result_value = float(result.evalf())
        
        logging.info(f"✅ Kết quả: {result_value}")
        return {
            'expression': expression,
            'result': result_value,
            'formatted': f"{expression} = {result_value}"
        }
        
    except Exception as e:
        logging.error(f"❌ Lỗi calculate: {e}")
        return {'error': f'Lỗi tính toán: {str(e)}'}


def execute_code(code: str, timeout_seconds: int = 5) -> dict:
    """
    Thực thi code Python trong sandbox an toàn
    
    Args:
        code: Code Python cần chạy
        timeout_seconds: Timeout (giây)
    
    Returns:
        Dict với output hoặc error
    """
    try:
        logging.info(f"💻 Execute Code:\n{code[:100]}...")
        
        # Compile code với RestrictedPython
        byte_code = compile_restricted_exec(code)
        
        if byte_code.errors:
            return {'error': f'Lỗi cú pháp: {byte_code.errors}'}
        
        # Tạo safe environment
        safe_env = {
            '__builtins__': safe_globals,
            '_print_': lambda x: print(x),
            '_getattr_': getattr,
        }
        
        # Thêm các module an toàn
        import math
        safe_env['math'] = math
        
        # Capture output
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = output_buffer = StringIO()
        
        try:
            # Execute với timeout
            exec(byte_code.code, safe_env)
            output = output_buffer.getvalue()
            
            logging.info(f"✅ Code executed successfully")
            return {
                'output': output if output else 'Code chạy thành công (không có output)',
                'success': True
            }
        finally:
            sys.stdout = old_stdout
        
    except TimeoutError:
        logging.error("❌ Code execution timeout")
        return {'error': 'Timeout: Code chạy quá 5 giây'}
    except Exception as e:
        logging.error(f"❌ Lỗi execute code: {e}")
        return {'error': f'Lỗi runtime: {str(e)}'}


def detect_tool_needed(question: str) -> str:
    """
    Phát hiện tool nào cần dùng dựa vào câu hỏi - THÔNG MINH như ChatGPT
    
    Returns:
        'search' | 'calculate' | 'code' | 'clarify' | 'chat'
    """
    question_lower = question.lower()
    
    # ===== 0. PRICE/COMMODITY SEARCH - Ưu tiên cao nhất =====
    # Tránh nhầm "giá vàng" với "chia"
    price_keywords = ['giá vàng', 'giá dầu', 'giá xăng', 'giá bitcoin', 'giá usd', 'giá vnd', 'tỷ giá', 'giá cổ phiếu', 'giá nhà', 'giá đất']
    if any(kw in question_lower for kw in price_keywords):
        return 'search'
    
    # ===== 1. CALCULATOR - Chỉ khi có số cụ thể VÀ có phép tính =====
    calc_keywords = [
        'calculate', 'bằng bao nhiêu', 'bằng', 'cộng', 'trừ', 'nhân',
        '+', '-', '*', '/', 'sqrt', 'sin', 'cos', 'tan', '^', '**', '='
    ]
    
    # Kiểm tra có phép toán rõ ràng
    has_math_operator = re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', question)  # VD: 3+5, 10*2
    has_calc_keyword = any(kw in question_lower for kw in calc_keywords)
    
    if has_math_operator or (has_calc_keyword and re.search(r'\d+', question)):
        # Chỉ calculator khi có phép tính hoặc keyword + số
        return 'calculate'
    
    # ===== 2. CODE EXECUTION =====
    code_keywords = [
        'chạy code', 'execute', 'run python', 'viết code',
        'def ', 'for ', 'while ', 'print(', 'import ',
        'fibonacci', 'prime', 'sort', 'algorithm'
    ]
    if any(kw in question_lower for kw in code_keywords):
        return 'code'
    
    # ===== 3. WEB SEARCH - TỰ ĐỘNG NHẬN BIẾT =====
    
    # 3.1. Keywords rõ ràng
    explicit_search_keywords = [
        'tìm kiếm', 'search', 'tra cứu', 'tìm', 'google',
        'tin tức', 'thông tin mới', 'tin mới', 'cập nhật'
    ]
    if any(kw in question_lower for kw in explicit_search_keywords):
        return 'search'
    
    # 3.2. Thời gian realtime (hôm nay, hiện tại, năm 2025...)
    time_keywords = ['hôm nay', 'hiện tại', 'bây giờ', 'mới nhất', 'năm 2025', 'tháng 11']
    if any(kw in question_lower for kw in time_keywords):
        return 'search'
    
    # 3.3. Câu hỏi về người/địa điểm/sự kiện cụ thể
    wh_questions = ['ai là', 'ai đã', 'khi nào', 'ở đâu', 'tại sao', 'như thế nào', 'có phải']
    if any(kw in question_lower for kw in wh_questions):
        # Kiểm tra có tên riêng (chữ hoa) hoặc tên người/địa điểm
        if re.search(r'[A-Z][a-z]+|việt nam|hà nội|sài gòn|mỹ|trung quốc', question):
            return 'search'
    
    # 3.4. Lĩnh vực cần thông tin mới (giá cả, thời tiết, thể thao, chính trị...)
    realtime_topics = [
        'giá', 'thời tiết', 'nhiệt độ', 'dự báo', 'tỷ giá', 'chứng khoán',
        'bóng đá', 'world cup', 'olympic', 'giải đấu', 'kết quả', 'tỷ số',
        'tổng thống', 'thủ tướng', 'chính phủ', 'quốc hội', 'bầu cử',
        'covid', 'dịch bệnh', 'vaccine', 'ca nhiễm',
        'chiến tranh', 'xung đột', 'biển đông',
        'công nghệ mới', 'điện thoại', 'iphone', 'samsung', 'tesla', 'ai', 'chatgpt'
    ]
    if any(topic in question_lower for topic in realtime_topics):
        return 'search'
    
    # 3.5. Số liệu thống kê (bao nhiêu người, bao nhiêu tỉnh, dân số...)
    stat_keywords = ['bao nhiêu', 'số lượng', 'thống kê', 'dân số', 'diện tích', 'chiều cao']
    if any(kw in question_lower for kw in stat_keywords):
        # Trừ câu hỏi toán học đơn giản
        if not re.search(r'\d+\s*[+\-*/]\s*\d+', question):
            return 'search'
    
    # ===== 4. CLARIFY - Câu hỏi mơ hồ =====
    clarify_keywords = ['cái đó', 'nó', 'gì đó', 'thứ gì', 'mấy cái', 'một số', 'vài']
    if any(kw in question_lower for kw in clarify_keywords):
        # Nếu câu hỏi quá ngắn và mơ hồ
        if len(question.split()) < 5:
            return 'clarify'
    
    # ===== 5. CHAT - Mặc định =====
    return 'chat'

def query_lm_studio(messages: list, stream: bool = False) -> dict:
    """
    Gửi request đến LM Studio với GPU acceleration
    
    Args:
        messages: List of message dicts [{'role': 'user', 'content': '...'}]
        stream: True để streaming response, False để response đầy đủ
    
    Returns:
        Response dict từ LM Studio (hoặc generator nếu stream=True)
    """
    payload = {
        'model': LM_STUDIO_MODEL,
        'messages': messages,
        'temperature': 0.8,
        'max_tokens': 3096,
        'top_p': 0.95,
        'frequency_penalty': 0.3,
        'presence_penalty': 0.3,
        'stream': stream,  # Hỗ trợ streaming
        # GPU optimization parameters
        'num_gpu': 1,
        'gpu_layers': 35,
    }
    
    try:
        logging.info(f"Gửi request đến LM Studio (stream={stream})")
        
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=60,
            stream=stream  # Streaming mode
        )
        
        if response.status_code != 200:
            logging.error(f"LM Studio trả về lỗi: {response.status_code}")
            return None
        
        # Nếu streaming, trả về response object để iterate
        if stream:
            return response
        
        # Nếu không stream, parse JSON
        result = response.json()
        logging.info("Nhận được response từ LM Studio")
        return result
        
    except requests.exceptions.ConnectionError:
        logging.error(f"Không thể kết nối đến LM Studio tại {LM_STUDIO_URL}")
        logging.error("Hãy chắc chắn LM Studio đang chạy và model đã load!")
        return None
    except Exception as e:
        logging.error(f"Lỗi khi gọi LM Studio: {e}")
        return None


# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Serve frontend"""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok'
    })


@app.route('/query', methods=['POST'])
def query():
    """Endpoint chính: Gửi câu hỏi đến LM Studio"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Thiếu tham số question'}), 400
    
    question = data['question']
    session_id = data.get('session_id', 'default')
    
    # Kiểm tra nếu là lời chào
    if is_greeting(question):
        answer = "Chào Bạn! Tôi là Chatbot AI 37 có thể trả lời tất cả câu hỏi của bạn."
        add_to_memory(session_id, 'user', question)
        add_to_memory(session_id, 'assistant', answer)
        return jsonify({'reply': answer})
    
    # Lấy lịch sử hội thoại
    history = get_conversation_history(session_id)
    
    # Tạo messages
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({'role': 'user', 'content': question})
    
    # Gọi LM Studio
    lm_resp = query_lm_studio(msgs)
    
    if not lm_resp:
        return jsonify({'error': 'Lỗi kết nối LM Studio'}), 500
    
    try:
        answer = lm_resp['choices'][0]['message']['content']
        answer = clean_response(answer)
        
        # Lưu vào memory
        add_to_memory(session_id, 'user', question)
        add_to_memory(session_id, 'assistant', answer)
        
        return jsonify({'reply': answer})
    except Exception as e:
        logging.error(f"Lỗi xử lý response: {e}")
        return jsonify({'error': 'Lỗi xử lý câu trả lời'}), 500



@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Endpoint chat cho frontend - HỖ TRỢ STREAMING"""
    
    # Xử lý GET request (thường do browser cache)
    if request.method == 'GET':
        return jsonify({
            'message': 'Endpoint /chat chỉ chấp nhận POST request',
            'usage': 'POST /chat với body: {"message": "câu hỏi", "stream": true}'
        }), 200
    
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Thiếu tham số message'}), 400
    
    question = data['message']
    session_id = data.get('session_id', 'default')
    use_stream = data.get('stream', True)  # Mặc định bật streaming
    
    logging.info(f"📨 Nhận câu hỏi: '{question}' (stream={use_stream})")
    
    # Kiểm tra nếu là lời chào đơn giản
    if is_greeting(question):
        answer = "Chào bạn! Tôi là AI 37, tôi có thể giúp gì cho bạn?"
        add_to_memory(session_id, 'user', question)
        add_to_memory(session_id, 'assistant', answer)
        
        # Nếu streaming, trả về streaming format
        if use_stream:
            def greeting_stream():
                # Gửi từng chữ để giống streaming
                for char in answer:
                    yield f"data: {json.dumps({'content': char})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            return Response(
                stream_with_context(greeting_stream()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            return jsonify({'reply': answer, 'response': answer})
    
    # Lấy lịch sử hội thoại
    history = get_conversation_history(session_id)
    
    # Kiểm tra follow-up question - nếu là follow-up, không cần search lại
    is_follow_up = is_follow_up_question(question, history)
    
    if is_follow_up:
        logging.info(f"💬 Detected follow-up question - using context from previous conversation")
        # Không gọi tool, để LLM tự trả lời dựa vào context trong history
        tool_needed = 'chat'
        
        # Thêm context từ câu trả lời trước để LLM hiểu rõ hơn
        last_assistant_msg = None
        for msg in reversed(history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg.get('content', '')
                break
        
        if last_assistant_msg and len(last_assistant_msg) > 50:
            # Thêm context ngắn gọn từ câu trả lời trước
            tool_context = f"\n\n💬 NGỮ CẢNH TỪ CÂU TRẢ LỜI TRƯỚC:\n{last_assistant_msg[:500]}\n\n"
            tool_context += "⚠️ YÊU CẦU: Trả lời dựa vào ngữ cảnh trên, mở rộng thông tin nếu cần. Trả lời TỰ NHIÊN, như thể bạn đang tiếp tục câu chuyện.\n\n"
    else:
        # GIAI ĐOẠN 2: Detect tool needed
        tool_needed = detect_tool_needed(question)
        logging.info(f"🔧 Tool detected: {tool_needed}")
    
    tool_result = None
    tool_context = ""
    direct_answer = None
    
    # Gọi tool nếu cần
    if tool_needed == 'search':
        question_lower = question.lower()
        
        # ===== CASE 1: THỜI TIẾT → GỌI ACCUWEATHER API =====
        if 'thời tiết' in question_lower or 'nhiệt độ' in question_lower or 'dự báo' in question_lower:
            logging.info("🌤️ Detected weather query - calling AccuWeather API")
            
            # Trích xuất tên thành phố từ câu hỏi (danh sách đầy đủ 63 tỉnh thành)
            location = "Hanoi"  # Mặc định
            
            # Các thành phố lớn (dùng location_key trực tiếp để tiết kiệm API calls)
            location_key = None
            city_name = None
            
            if 'hà nội' in question_lower or 'hanoi' in question_lower:
                location_key = "353412"  # Hanoi
                city_name = "Hà Nội"
            elif 'sài gòn' in question_lower or 'hồ chí minh' in question_lower or 'saigon' in question_lower or 'tp hcm' in question_lower:
                location_key = "353981"  # Ho Chi Minh City
                city_name = "TP Hồ Chí Minh"
            elif 'đà nẵng' in question_lower or 'da nang' in question_lower:
                location_key = "353926"  # Da Nang
                city_name = "Đà Nẵng"
            elif 'hải phòng' in question_lower or 'hai phong' in question_lower:
                location_key = "353346"  # Hai Phong
                city_name = "Hải Phòng"
            elif 'cần thơ' in question_lower or 'can tho' in question_lower:
                location_key = "353933"  # Can Tho
                city_name = "Cần Thơ"
            
            # Các tỉnh miền Bắc
            elif 'nghệ an' in question_lower or 'nghe an' in question_lower or 'vinh' in question_lower:
                location = "Vinh"
            elif 'thanh hóa' in question_lower or 'thanh hoa' in question_lower:
                location = "Thanh Hoa"
            elif 'hà tĩnh' in question_lower or 'ha tinh' in question_lower:
                location = "Ha Tinh"
            elif 'quảng ninh' in question_lower or 'quang ninh' in question_lower or 'hạ long' in question_lower:
                location = "Ha Long"
            elif 'lào cai' in question_lower or 'lao cai' in question_lower or 'sapa' in question_lower:
                location = "Lao Cai"
            
            # Các tỉnh miền Trung
            elif 'huế' in question_lower or 'hue' in question_lower or 'thừa thiên' in question_lower:
                location = "Hue"
            elif 'quảng nam' in question_lower or 'quang nam' in question_lower or 'hội an' in question_lower:
                location = "Tam Ky"
            elif 'quảng ngãi' in question_lower or 'quang ngai' in question_lower:
                location = "Quang Ngai"
            elif 'bình định' in question_lower or 'binh dinh' in question_lower or 'quy nhơn' in question_lower:
                location = "Quy Nhon"
            elif 'phú yên' in question_lower or 'phu yen' in question_lower or 'tuy hòa' in question_lower:
                location = "Tuy Hoa"
            elif 'khánh hòa' in question_lower or 'khanh hoa' in question_lower or 'nha trang' in question_lower:
                location = "Nha Trang"
            
            # Tây Nguyên
            elif 'đắk lắk' in question_lower or 'dak lak' in question_lower or 'buôn ma thuột' in question_lower:
                location = "Buon Ma Thuot"
            elif 'lâm đồng' in question_lower or 'lam dong' in question_lower or 'đà lạt' in question_lower or 'da lat' in question_lower:
                location = "Da Lat"
            elif 'gia lai' in question_lower or 'pleiku' in question_lower:
                location = "Pleiku"
            
            # Miền Nam
            elif 'bình dương' in question_lower or 'binh duong' in question_lower or 'thủ dầu một' in question_lower:
                location = "Thu Dau Mot"
            elif 'đồng nai' in question_lower or 'dong nai' in question_lower or 'biên hòa' in question_lower:
                location = "Bien Hoa"
            elif 'bà rịa' in question_lower or 'ba ria' in question_lower or 'vũng tàu' in question_lower or 'vung tau' in question_lower:
                location = "Vung Tau"
            elif 'long an' in question_lower or 'tân an' in question_lower:
                location = "Tan An"
            elif 'tiền giang' in question_lower or 'tien giang' in question_lower or 'mỹ tho' in question_lower:
                location = "My Tho"
            elif 'vĩnh long' in question_lower or 'vinh long' in question_lower:
                location = "Vinh Long"
            elif 'an giang' in question_lower or 'long xuyên' in question_lower:
                location = "Long Xuyen"
            elif 'kiên giang' in question_lower or 'kien giang' in question_lower or 'rạch giá' in question_lower:
                location = "Rach Gia"
            elif 'cà mau' in question_lower or 'ca mau' in question_lower:
                location = "Ca Mau"
            
            # ===== QUY TRÌNH CHATGPT: WEB SEARCH + TRÍCH XUẤT THÔNG MINH =====
            if not city_name:
                city_name = location  # Fallback cho các tỉnh không có location_key
            
            weather_result = get_weather_chatgpt_style(city_name, question)
            
            if 'error' not in weather_result:
                # Kiểm tra xem có câu hỏi về giờ cụ thể không
                target_hour = extract_hour_from_question(question)
                
                # Tạo context từ dữ liệu đã trích xuất và chuẩn hóa
                tool_context = f"\n\n🌤️ THỜI TIẾT {weather_result['city'].upper()} HÔM NAY:\n\n"
                
                # Nếu có câu hỏi về giờ cụ thể
                if target_hour is not None and 'hourly_temperature' in weather_result:
                    hourly_temp = weather_result['hourly_temperature']
                    is_estimated = weather_result.get('estimated', False)
                    
                    tool_context += f"⏰ Thời tiết lúc {target_hour}h: {hourly_temp}°C"
                    if is_estimated:
                        tool_context += " (ước tính)"
                    tool_context += "\n"
                    
                    if weather_result['conditions']:
                        tool_context += f"☁️ Điều kiện: {weather_result['conditions']}\n"
                    
                    if weather_result['humidity']:
                        tool_context += f"💧 Độ ẩm: {weather_result['humidity']}%\n"
                    
                    tool_context += "\n"
                    tool_context += "⚠️ YÊU CẦU: Trả lời 1-2 câu tự nhiên về thời tiết lúc giờ đó.\n"
                    conditions_text = weather_result.get('conditions', 'trời')
                    tool_context += f"Ví dụ: 'Hà Nội lúc {target_hour}h khoảng {hourly_temp}°C, {conditions_text}.'\n\n"
                
                # Câu hỏi chung về thời tiết
                else:
                    if weather_result.get('current_temperature'):
                        tool_context += f"🌡️ Nhiệt độ hiện tại: {weather_result['current_temperature']}°C\n"
                    
                    if weather_result.get('temperature_min') and weather_result.get('temperature_max'):
                        tool_context += f"🌡️ Nhiệt độ: {weather_result['temperature_min']}°C - {weather_result['temperature_max']}°C\n"
                    elif weather_result.get('temperature_avg'):
                        tool_context += f"🌡️ Nhiệt độ: khoảng {weather_result['temperature_avg']}°C\n"
                    
                    if weather_result.get('humidity'):
                        tool_context += f"💧 Độ ẩm: {weather_result['humidity']}%\n"
                    
                    if weather_result.get('conditions'):
                        tool_context += f"☁️ Tình trạng: {weather_result['conditions']}\n"
                    
                    tool_context += f"\n📚 Nguồn: AccuWeather\n\n"
                    tool_context += "⚠️ YÊU CẦU: Trả lời TỰ NHIÊN như ChatGPT, 2-3 câu ngắn gọn, dễ hiểu.\n"
                    
                    # Tạo ví dụ động dựa trên dữ liệu thực
                    example_temp = weather_result.get('current_temperature') or weather_result.get('temperature_max') or '22'
                    example_condition = weather_result.get('conditions', 'trời').split(',')[0] if weather_result.get('conditions') else 'trời'
                    tool_context += f"Ví dụ: '{city_name} hôm nay khoảng {example_temp}°C, {example_condition}, thời tiết dễ chịu.'\n\n"
                
                logging.info(f"✅ [ChatGPT Style] Weather context created ({len(tool_context)} chars)")
            else:
                tool_context = f"\n\n[Lỗi lấy thời tiết: {weather_result['error']}]"
        
        # ===== CASE 2: GIÁ VÀNG → TRÍCH XUẤT GIÁ TỪ BÀI BÁO =====
        elif 'giá vàng' in question_lower or 'vàng sjc' in question_lower:
            logging.info("💰 Detected gold price query - extracting gold price")
            
            # Phát hiện nếu người dùng hỏi về "hôm nay" hoặc ngày cụ thể
            has_today = 'hôm nay' in question_lower
            today = datetime.now()
            current_date_str = today.strftime('%d/%m/%Y')
            current_date_str_short = today.strftime('%d/%m')
            
            # Tối ưu query tìm kiếm: thêm "hôm nay" hoặc ngày hiện tại nếu người dùng hỏi về hôm nay
            search_query = question
            if has_today:
                # Thêm ngày hiện tại vào query để tìm kết quả mới nhất
                search_query = f"{question} {current_date_str_short} {current_date_str}"
                logging.info(f"📅 User asked about today - adding date to query: {current_date_str}")
            
            # Tìm kiếm để lấy URL
            tool_result = search_web(search_query, max_results=10, prioritize_today=has_today)
            
            if 'results' in tool_result and tool_result['results']:
                # Thử trích xuất từ TẤT CẢ các nguồn (tối đa 3 nguồn) để tăng độ chính xác
                sources = tool_result['results'][:3]
                gold_price = None
                best_source = None
                
                logging.info(f"📰 Thử trích xuất giá vàng từ {len(sources)} nguồn...")
                
                # Thử từng nguồn cho đến khi tìm được giá
                for idx, source in enumerate(sources, 1):
                    url = source['url']
                    title = source['title']
                    
                    logging.info(f"   [{idx}] Thử nguồn: {title[:60]}")
                    
                    # Fetch HTML content với retry và error handling tốt hơn
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive',
                            'Upgrade-Insecure-Requests': '1'
                        }
                        
                        # Retry logic với timeout tăng dần
                        max_retries = 2  # Giảm xuống 2 lần để nhanh hơn
                        response = None
                        for attempt in range(max_retries):
                            try:
                                response = requests.get(url, headers=headers, timeout=10, verify=True)
                                response.raise_for_status()
                                break
                            except requests.exceptions.SSLError as ssl_err:
                                if attempt < max_retries - 1:
                                    logging.warning(f"      ⚠️ SSL error, retrying with verify=False...")
                                    response = requests.get(url, headers=headers, timeout=10, verify=False)
                                    response.raise_for_status()
                                    break
                                else:
                                    raise
                            except requests.exceptions.RequestException as req_err:
                                if attempt < max_retries - 1:
                                    continue
                                else:
                                    raise
                        
                        if response:
                            # TRÍCH XUẤT giá vàng
                            gold_price = extract_gold_price(response.text, url)
                            
                            if gold_price:
                                best_source = source
                                logging.info(f"   ✅ [{idx}] Trích xuất thành công từ nguồn này!")
                                break
                            else:
                                logging.info(f"   ⚠️ [{idx}] Không trích xuất được, thử nguồn tiếp theo...")
                    
                    except Exception as e:
                        logging.warning(f"   ⚠️ [{idx}] Lỗi fetch: {e}, thử nguồn tiếp theo...")
                        continue
                
                # Tạo context dựa trên kết quả
                if gold_price:
                    # Tạo context NGẮN GỌN từ giá đã trích xuất
                    if 'buy' in gold_price and 'sell' in gold_price:
                        tool_context = f"\n\n💰 GIÁ VÀNG SJC HÔM NAY:\n"
                        tool_context += f"Mua vào: {gold_price['buy']} {gold_price['unit']}\n"
                        tool_context += f"Bán ra: {gold_price['sell']} {gold_price['unit']}\n\n"
                    else:
                        tool_context = f"\n\n💰 GIÁ VÀNG SJC: {gold_price.get('price', 'N/A')} {gold_price['unit']}\n\n"
                    
                    tool_context += (
                        "⚠️ YÊU CẦU: Trả lời 1 câu duy nhất, định dạng: "
                        "\"Vàng SJC mua X triệu/lượng, bán Y triệu/lượng.\" "
                        "CHỈ dùng số trong phần \"GIÁ VÀNG SJC HÔM NAY\" ở trên. "
                        "KHÔNG thêm lời khuyên, KHÔNG nhắc nguồn, KHÔNG dự đoán, KHÔNG giải thích.\n\n"
                    )
                    
                    logging.info(f"✅ Gold price extracted and formatted ({len(tool_context)} chars)")
                else:
                    # Fallback: Dùng full content từ nguồn tốt nhất và tìm số trong đó
                    best_source = sources[0]  # Dùng nguồn đầu tiên
                    url = best_source['url']
                    title = best_source['title']
                    
                    logging.info(f"📰 Fetching full article từ nguồn tốt nhất: {title[:60]}")
                    
                    try:
                        full_content = fetch_full_article(url)
                        
                        if full_content:
                            # Tìm tất cả số có thể là giá vàng trong content
                            # Pattern: tìm số trong khoảng 50-200 triệu
                            price_numbers = re.findall(r'(\d{1,2}[,\.]\d{1,2})\s*(?:triệu|tr)', full_content)
                            price_context = ""
                            
                            if price_numbers:
                                # Lấy 2-4 số đầu tiên (có thể là mua-bán)
                                unique_prices = list(dict.fromkeys(price_numbers[:4]))  # Loại bỏ duplicate nhưng giữ thứ tự
                                price_context = f"\n📊 CÁC SỐ CÓ THỂ LÀ GIÁ VÀNG (triệu/lượng): {', '.join(unique_prices)}\n"
                            
                            tool_context = f"\n\n💰 THÔNG TIN GIÁ VÀNG HÔM NAY:\n{price_context}\n📄 Nội dung bài báo:\n{full_content[:1500]}\n\n"
                            tool_context += (
                                "⚠️ YÊU CẦU QUAN TRỌNG:\n"
                                "1. Tìm trong nội dung bài báo ở trên, tìm giá vàng SJC mua và bán (đơn vị: triệu/lượng)\n"
                                "2. Nếu có phần \"CÁC SỐ CÓ THỂ LÀ GIÁ VÀNG\" ở trên, ưu tiên dùng 2 số đầu tiên (thường là mua-bán)\n"
                                "3. Trả lời 1 câu duy nhất, định dạng: \"Vàng SJC mua X triệu/lượng, bán Y triệu/lượng.\"\n"
                                "4. Nếu không tìm thấy số cụ thể, trả lời: \"Tôi không tìm thấy giá vàng SJC chính xác trong thông tin hiện có.\"\n"
                                "5. KHÔNG nhắc nguồn, KHÔNG dự đoán, KHÔNG giải thích dài dòng.\n\n"
                            )
                            logging.warning("⚠️ Could not extract price, using full content with number hints")
                        else:
                            # Fallback cuối: dùng snippet từ tất cả nguồn
                            all_snippets = []
                            for s in sources[:3]:
                                if s.get('snippet'):
                                    all_snippets.append(f"[{s['title'][:50]}] {s['snippet'][:200]}")
                            
                            snippets_text = "\n\n".join(all_snippets)
                            tool_context = f"\n\n💰 THÔNG TIN GIÁ VÀNG:\n\n{snippets_text}\n\n"
                            tool_context += (
                                "⚠️ YÊU CẦU: Tìm giá vàng SJC mua và bán trong thông tin trên. "
                                "Trả lời 1 câu duy nhất: \"Vàng SJC mua X triệu/lượng, bán Y triệu/lượng.\" "
                                "Nếu không tìm thấy, trả lời: \"Tôi không tìm thấy giá vàng SJC chính xác.\"\n\n"
                            )
                            logging.warning("⚠️ Failed to fetch article, using snippets from all sources")
                    
                    except Exception as e:
                        logging.error(f"❌ Error fetching full content: {e}")
                        # Fallback: dùng snippet từ tất cả nguồn
                        all_snippets = []
                        for s in sources[:3]:
                            if s.get('snippet'):
                                all_snippets.append(f"[{s['title'][:50]}] {s['snippet'][:200]}")
                        
                        snippets_text = "\n\n".join(all_snippets) if all_snippets else sources[0].get('snippet', '')
                        tool_context = f"\n\n💰 THÔNG TIN GIÁ VÀNG:\n\n{snippets_text}\n\n"
                        tool_context += (
                            "⚠️ YÊU CẦU: Tìm giá vàng SJC mua và bán. "
                            "Trả lời 1 câu: \"Vàng SJC mua X triệu/lượng, bán Y triệu/lượng.\" "
                            "Nếu không tìm thấy, trả lời: \"Tôi không tìm thấy giá vàng SJC chính xác.\"\n\n"
                        )
            else:
                tool_context = f"\n\n[Lỗi tìm kiếm giá vàng]"
        
        # ===== CASE 3: CÂU HỎI KHÁC → QUY TRÌNH CHATGPT: SEARCH → FILTER → FETCH → ANALYZE → SYNTHESIZE =====
        else:
            # BƯỚC 1: Web Search - Tìm kiếm nhiều nguồn
            tool_result = search_web(question, max_results=10)
            
            if 'results' in tool_result:
                final_results = tool_result['results']
                
                logging.info(f"📊 Nhận được {len(final_results)} nguồn từ search_web")
                
                if final_results:
                    # BƯỚC 2: Fetch full content từ các nguồn tốt nhất (tối đa 3 nguồn)
                    sources_with_content = []
                    
                    for i, source in enumerate(final_results[:3], 1):  # Chỉ fetch top 3
                        title = source['title']
                        url = source['url']
                        snippet = source.get('snippet', '')
                        
                        logging.info(f"   [{i}] {title[:80]}")
                        logging.info(f"       URL: {url[:80]}")
                        
                        # BƯỚC 3: Fetch full content (nếu có thể)
                        full_content = fetch_full_article(url)
                        
                        source_data = {
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'full_content': full_content if full_content else snippet
                        }
                        
                        sources_with_content.append(source_data)
                        
                        if full_content:
                            logging.info(f"       ✅ Fetched {len(full_content)} chars")
                        else:
                            logging.info(f"       ⚠️ Using snippet ({len(snippet)} chars)")
                    
                    # BƯỚC 4: Phân tích và tổng hợp thông tin từ nhiều nguồn
                    tool_context = analyze_and_synthesize(sources_with_content, question)
                    
                    logging.info(f"✅ Đã tạo context tổng hợp ({len(tool_context)} chars)")
                else:
                    tool_context = ""
                    logging.warning("⚠️ No final results after filtering!")
            elif 'error' in tool_result:
                tool_context = f"\n\n[Lỗi tìm kiếm: {tool_result['error']}. Hãy trả lời dựa vào kiến thức của bạn.]"
    
    elif tool_needed == 'calculate':
        # Trích xuất biểu thức toán học - loại bỏ text, chỉ giữ phép tính
        # Hỗ trợ: "3 * 50 bằng bao nhiêu", "123*456 =", "tính 5+8"
        expression = question
        
        # Loại bỏ các từ khóa thường gặp
        expression = re.sub(r'(bằng|bao nhiêu|tính|là|kết quả)', '', expression, flags=re.IGNORECASE)
        # Loại bỏ dấu = ở cuối nếu có
        expression = expression.replace('=', '').strip()
        
        # Trích xuất biểu thức toán học (số, toán tử, dấu ngoặc, khoảng trắng)
        match = re.search(r'[\d\s+\-*/().^]+', expression)
        if match:
            expression = match.group().strip()
            # Loại bỏ khoảng trắng thừa
            expression = re.sub(r'\s+', '', expression)
            
            # Validate: Phải có ít nhất 1 phép tính
            if expression and re.search(r'[\+\-\*/\^]', expression):
                tool_result = calculate(expression)
                if 'result' in tool_result:
                    # Format kết quả: loại bỏ .0 nếu là số nguyên
                    result_value = tool_result['result']
                    if isinstance(result_value, float) and result_value.is_integer():
                        result_display = int(result_value)
                    else:
                        result_display = result_value
                    
                    tool_context = f"\n\n🔢 KẾT QUẢ TÍNH TOÁN: {expression} = {result_display}\n\nYÊU CẦU: Chỉ trả lời kết quả số, KHÔNG thêm bình luận hay giải thích. Format: \"[số]\" hoặc \"[phép tính] = [số]\"."
                    direct_answer = f"{expression} = {result_display}"
                    
                    logging.info(f"✅ Calculator tool context created ({len(tool_context)} chars)")
                    logging.info(f"📝 Result: {expression} = {result_display}")
                else:
                    logging.error(f"❌ Calculator failed: {tool_result.get('error', 'Unknown error')}")
            else:
                logging.warning(f"⚠️ Calculator: Không tìm thấy phép tính hợp lệ trong câu hỏi")
    
    elif tool_needed == 'code':
        # Trích xuất code từ câu hỏi (nếu có code block)
        code_match = re.search(r'```python\n(.*?)\n```', question, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            tool_result = execute_code(code)
            if 'output' in tool_result:
                tool_context = f"\n\n💻 KẾT QUẢ CHẠY CODE:\n{tool_result['output']}\n\nYÊU CẦU: Giải thích kết quả code trên bằng tiếng Việt một cách ngắn gọn."
                
                logging.info(f"✅ Code execution tool context created ({len(tool_context)} chars)")
            else:
                logging.error(f"❌ Code execution failed: {tool_result.get('error', 'Unknown error')}")
    
    elif tool_needed == 'clarify':
        # Multi-turn clarification - Hỏi lại khi câu hỏi mơ hồ
        logging.info("💬 Clarification needed")
        tool_context = "\n\n💬 Câu hỏi của người dùng chưa rõ ràng. Hãy hỏi lại để làm rõ:\n"
        tool_context += "VD: 'Bạn muốn biết thông tin gì cụ thể?', 'Bạn đang hỏi về cái gì?', 'Bạn có thể nói rõ hơn được không?'"
    
    # Nếu có câu trả lời trực tiếp (calculator) → trả về luôn, không gọi LLM
    if direct_answer is not None:
        answer = direct_answer
        add_to_memory(session_id, 'user', question)
        add_to_memory(session_id, 'assistant', answer)

        if not use_stream:
            return jsonify({'reply': answer, 'response': answer})

        def calc_stream():
            for char in answer:
                yield f"data: {json.dumps({'content': char})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        logging.info("✅ Returning direct calculator answer without calling LLM")
        return Response(
            stream_with_context(calc_stream()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    # Lấy lịch sử (nếu chưa lấy)
    if 'history' not in locals():
        history = get_conversation_history(session_id)
    
    # Tạo messages (lọc bỏ timestamp)
    msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for msg in history:
        msgs.append({
            'role': msg['role'],
            'content': msg['content']
        })
    
    # Thêm tool context vào câu hỏi nếu có
    final_question = question + tool_context if tool_context else question
    
    if tool_context:
        logging.info(f"🔗 Final question length: {len(final_question)} chars (original: {len(question)})")
    
    msgs.append({'role': 'user', 'content': final_question})
    
    # Nếu KHÔNG streaming → trả về JSON bình thường
    if not use_stream:
        lm_resp = query_lm_studio(msgs, stream=False)
        
        if not lm_resp:
            return jsonify({'error': 'Lỗi kết nối LM Studio'}), 500
        
        try:
            answer = lm_resp['choices'][0]['message']['content']
            answer = clean_response(answer)
            
            # Lưu memory
            add_to_memory(session_id, 'user', question)
            add_to_memory(session_id, 'assistant', answer)
            
            return jsonify({
                'reply': answer,
                'response': answer
            })
        except Exception as e:
            logging.error(f"Lỗi xử lý response: {e}")
            return jsonify({'error': 'Lỗi xử lý câu trả lời'}), 500
    
    # STREAMING MODE - Trả về từng chunk
    def generate():
        lm_resp = query_lm_studio(msgs, stream=True)
               
        if not lm_resp:
            yield f"data: {json.dumps({'error': 'Lỗi kết nối LM Studio'})}\n\n"
            return
        
        full_answer = ""
        
        try:
            # Đọc từng dòng từ streaming response
            for line in lm_resp.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # LM Studio trả về format: "data: {...}"
                    if line_text.startswith('data: '):
                        json_str = line_text[6:]  # Bỏ "data: "
                        
                        if json_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(json_str)
                            
                            # Lấy content từ chunk
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    full_answer += content
                                    
                                    # Gửi chunk đến frontend
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        except json.JSONDecodeError:
                            continue
            
            # Kết thúc stream
            yield f"data: {json.dumps({'done': True})}\n\n"
            
            # Lưu vào memory
            full_answer = clean_response(full_answer)
            add_to_memory(session_id, 'user', question)
            add_to_memory(session_id, 'assistant', full_answer)
            
        except Exception as e:
            logging.error(f"Lỗi streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )





# ===== MAIN =====

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤖 AI 37 CHATBOT - PHIÊN BẢN NÂNG CAO")
    print("="*70)
    print(f"📡 LM Studio: {LM_STUDIO_URL}")
    print(f"🧠 Model: {LM_STUDIO_MODEL}")
    print(f"🚀 GPU: Đã bật (gpu_layers=35)")
    print(f"🌐 Server: http://localhost:{SERVER_PORT}")
    print("="*70)
    print("✨ GIAI ĐOẠN 1:")
    print("   ⚡ Streaming Response - Gõ từng chữ như ChatGPT")
    print("   💾 Lưu lịch sử - Không mất khi restart")
    print("   🔄 Tóm tắt tự động - Nhớ hội thoại dài")
    print("="*70)
    print("🔥 GIAI ĐOẠN 2:")
    print("   🔍 Web Search - Tìm kiếm thông tin mới nhất (DuckDuckGo)")
    print("   🔢 Calculator - Tính toán toán học (Sympy)")
    print("   💻 Code Execution - Chạy Python code an toàn (Sandbox)")
    print("   💬 Multi-turn Clarification - Hỏi lại khi câu hỏi mơ hồ")
    print("="*70)
    print("⚠️  Đảm bảo LM Studio đang chạy với model: vistral-7b-chat@q8")
    print("⚠️  Bật GPU trong LM Studio Settings để tăng tốc")
    print("="*70 + "\n")
    
    # Chạy Flask - tắt auto-reload để tránh lỗi .env
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )
