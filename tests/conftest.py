import os

# Các module khởi tạo client Groq/Gemini/OpenRouter/Tavily ngay khi import -> cần key (giá trị giả là đủ, test không gọi API thật)
os.environ.setdefault('GROQ_API_KEY', 'test-key')
os.environ.setdefault('OPENROUTER_API_KEY', 'test-key')
os.environ.setdefault('TAVILY_API_KEY', 'test-key')
os.environ.setdefault('GOOGLE_API_KEY', 'test-key')
os.environ.setdefault('NVIDIA_API_KEY', 'test-key')
os.environ.setdefault('ROUTER9_API_KEY', 'test-key')
# Khóa mã hóa số điện thoại, email (pii.py) chỉ dùng cho test
os.environ.setdefault('PII_KEY', 'nhcmcb23MRz2go8gsqNBogicArJ_a_DtJseIPrtUFmY=')
# Không gửi trace của test lên LangSmith (đặt trước khi import phone_sales vì load_dotenv không ghi đè biến đã có)
os.environ['LANGSMITH_TRACING'] = 'false'
