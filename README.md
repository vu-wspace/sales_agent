# Phone Sales Agent

Trợ lý chat tiếng Việt cho cửa hàng bán lẻ iPhone, Samsung Galaxy và phụ kiện, xây dựng với LangChain/LangGraph. Dự án demo phục vụ học tập.

## Tính năng

- Tư vấn điện thoại theo ngân sách và nhu cầu; báo giá, màu, dung lượng, tình trạng hàng từng phiên bản.
- Tìm phụ kiện tương thích (ốp lưng, cường lực, sạc, cáp, S Pen…).
- Trả lời chính sách đổi trả, giao hàng, trả góp, bảo hành bằng RAG.
- Nhớ trạng thái hội thoại bán hàng (nhu cầu, máy đã giới thiệu, lo ngại, cảm xúc) và quản lý giỏ hàng.
- Chuyển nhân viên khi khách khiếu nại, đơn có vấn đề hoặc khách bực bội.
- Stream trạng thái và câu trả lời theo thời gian thực.
- Tự chuyển sang LLM dự phòng khi model chính lỗi hoặc hết quota.

| Thành phần | Công nghệ |
|---|---|
| Agent | LangGraph `StateGraph` (`graph.py`) hoặc LangChain `create_agent` (`agent.py`) |
| LLM | OpenRouter (chính) → NVIDIA → Gemini → Groq (dự phòng) |
| Dữ liệu sản phẩm | PostgreSQL 18: 23 điện thoại, 134 phụ kiện, 377 phiên bản |
| Chính sách (RAG) | Qdrant + embedding `bkai-foundation-models/vietnamese-bi-encoder` |
| Web | FastAPI + trang chat HTML một file |
| Triển khai | Docker Compose |

## Bắt đầu nhanh

Yêu cầu: Docker. Khi phát triển trên máy cần thêm Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # điền API key
docker compose up -d --build    # Postgres (tự nạp dữ liệu) + Qdrant + web app (tự nạp chính sách)
```

| Địa chỉ | Nội dung |
|---|---|
| http://localhost:8000 | Trang chat |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:6333/dashboard | Dashboard Qdrant |

Lần build đầu mất khoảng 8 phút, chủ yếu để tải model embedding (~540 MB). Các lần sau chỉ build lại vài giây nếu không đổi thư viện.

### Chạy trên máy

Postgres và Qdrant chạy bằng Docker, code chạy trên máy:

```bash
uv sync
docker compose up -d postgres qdrant
uv run python -m phone_sales.vector_db           # nạp chính sách vào Qdrant
uv run uvicorn phone_sales.api:app --port 8001   # web app (8000 đã dùng cho container api)
uv run langgraph dev                             # LangGraph Studio: agent_main, phone_graph
```

Dùng trong code:

```python
from langgraph.checkpoint.memory import InMemorySaver
from phone_sales.graph import build_graph

graph = build_graph(checkpointer=InMemorySaver())
config = {'configurable': {'thread_id': 'khach-1'}}
graph.invoke({'messages': [{'role': 'user', 'content': 'tầm 30 triệu muốn máy chụp ảnh đẹp'}]}, config)
res = graph.invoke({'messages': [{'role': 'user', 'content': 'chốt galaxy s26 256gb màu đen'}]}, config)
print(res['messages'][-1].text, res['stage'], res['cart'])
```

### Lệnh Docker thường dùng

```bash
docker compose logs -f api                                # xem log
docker compose up -d --build api                          # build lại sau khi sửa code hoặc chính sách
docker compose run --rm db-seed                           # nạp lại dữ liệu sản phẩm (không mất volume)
docker compose exec api python -m phone_sales.vector_db   # nạp lại chính sách
docker compose down                                       # dừng, giữ dữ liệu
docker compose down -v                                    # dừng và xóa dữ liệu
```

| Container | Cổng | Vai trò |
|---|---|---|
| `phone_db` | 5433 | PostgreSQL, database `phone` |
| `qdrant` | 6333, 6334 | Vector DB |
| `phone_sales_api` | 8000 | Web app |

## Cấu hình

Biến môi trường đặt trong `.env` (mẫu ở `.env.example`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OPENROUTER_API_KEY` | (bắt buộc) | LLM chính ([openrouter.ai](https://openrouter.ai/keys)) |
| `NVIDIA_API_KEY` | (bắt buộc) | LLM dự phòng ([build.nvidia.com](https://build.nvidia.com)) |
| `GOOGLE_API_KEY` | (bắt buộc) | Gemini: LLM dự phòng và judge của eval ([AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | (bắt buộc) | LLM dự phòng ([console.groq.com](https://console.groq.com)) |
| `TAVILY_API_KEY` | (bắt buộc) | Tool `web_search` ([tavily.com](https://tavily.com)) |
| `POSTGRES_PASSWORD` | `postgres` | Mật khẩu Postgres, chỉ có hiệu lực khi tạo volume lần đầu |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/phone` | Database khi chạy trên máy |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant khi chạy trên máy |
| `AGENT_IMPL` | `graph` | Web app dùng `graph.py` (có trạng thái, giỏ hàng) hay `agent.py` |
| `AUTO_INGEST` | `true` | Tự nạp chính sách khi Qdrant rỗng hoặc đã cũ |
| `LANGSMITH_*` | tắt | Trace LangSmith (tùy chọn) |

Trong Docker, compose tự đặt `DATABASE_URL` và `QDRANT_URL` trỏ tới các container.

## API

| Method | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/` | Trang chat |
| `POST` | `/chat` | Gửi tin nhắn, nhận câu trả lời đầy đủ |
| `POST` | `/chat/stream` | Như `/chat`, trả Server-Sent Events |
| `GET` | `/health` | Trạng thái Qdrant và số đoạn chính sách |

### `POST /chat`

```json
// request: bỏ trống thread_id để bắt đầu hội thoại mới
{"message": "ok chốt galaxy s26 256gb màu đen", "thread_id": "5f0c…"}

// response
{"thread_id": "5f0c…", "answer": "Dạ em đã thêm Galaxy S26 256GB Đen Classic vào giỏ…",
 "tools": ["add_to_cart"], "stage": "closing",
 "cart": [{"variant_id": "SM-S942BZKCXXV", "name": "Galaxy S26", "storage": "256GB",
           "color": "Đen Classic", "price": 29990001, "quantity": 1}],
 "handoff": false}
```

- `message` dài 1–1000 ký tự, `thread_id` tối đa 100 ký tự.
- `stage`, `cart`, `handoff` chỉ có giá trị khi `AGENT_IMPL=graph`.
- Mã lỗi: `422` input sai, `429` mọi LLM đều vượt giới hạn, `502` mọi LLM đều lỗi.

### `POST /chat/stream`

Request giống `/chat`. Mỗi sự kiện là một dòng `data: {json}` với `type`:

| `type` | Nội dung |
|---|---|
| `start` | `thread_id` của hội thoại |
| `status` | Dòng trạng thái, vd "Đang xem giá và tồn kho…" |
| `token` | Một đoạn chữ của câu trả lời |
| `reset` | Bỏ phần chữ đã stream (model quyết định gọi tool) |
| `done` | Kết quả cuối, giống response của `/chat` |
| `error` | `status` (429/502/500) và `detail` |

Kết quả tool và nội dung suy nghĩ của model không bao giờ được gửi cho client.

## Kiến trúc

```mermaid
flowchart TB
    browser["Trình duyệt"] -- "POST /chat/stream" --> api["FastAPI<br/>api.py"]
    api -- "state theo thread_id" --> agent["Agent<br/>graph.py / agent.py"]
    agent <--> chain["Chuỗi LLM dự phòng<br/>fallback.py"]
    agent -- "gọi tool" --> tools["Tools<br/>tools.py"]
    tools --> postgres[("Postgres<br/>sản phẩm, phiên bản, tương thích")]
    tools --> qdrant[("Qdrant<br/>chính sách")]
    tools --> tavily(["Tavily"])
```

Luồng một lượt chat trong `graph.py`:

```mermaid
flowchart LR
    start((START)) --> track_user --> model
    model -- "có tool_call" --> tools --> track_tools --> model
    model -- "trả lời xong" --> track_reply --> finish((END))
```

1. `track_user` đọc tin nhắn của khách: nhu cầu, lo ngại, cảm xúc, có cần chuyển nhân viên không.
2. `model` gọi LLM với system prompt, tóm tắt trạng thái hội thoại và lịch sử đã cắt gọn (4 lượt gần nhất).
3. `tools` chạy tool; `track_tools` cập nhật nhu cầu từ tham số tool và tính lại bước bán hàng.
4. `track_reply` ghi lại các sản phẩm thật sự được giới thiệu trong câu trả lời.

`agent.py` dựng bằng `create_agent`, dùng chung LLM, tool tra cứu và prompt nhưng không có trạng thái hội thoại và giỏ hàng.

### Trạng thái hội thoại

`AgentState` (`conversation.py`) được lưu theo `thread_id` và cập nhật bằng luật từ khóa tiếng Việt (không gọi thêm LLM):

| Trường | Ý nghĩa |
|---|---|
| `stage` | Bước bán hàng: `discovery` → `recommending` → `comparing` → `closing`, hoặc `handoff` |
| `slots` | Nhu cầu: ngân sách, hãng, mục đích, dung lượng, màu |
| `shown_products` | Sản phẩm đã giới thiệu |
| `cart` | Giỏ hàng |
| `objections` | Lo ngại của khách (giá, pin, nhiệt độ, camera…) |
| `sentiment_history` | Cảm xúc từng lượt |
| `handoff` | Cần chuyển nhân viên |

Mỗi bước có hướng dẫn riêng được đưa vào system prompt, vd `closing`: xác nhận giỏ và tổng tiền, hỏi cách nhận hàng và thanh toán.

### Tools

| Tool | Nguồn | Dùng khi |
|---|---|---|
| `search_products` | Postgres | Tìm máy theo dòng, hãng, ngân sách |
| `get_product_details` | Postgres | Giá, tồn kho từng phiên bản, thông số của 1 máy |
| `find_accessories` | Postgres | Phụ kiện tương thích, 5 món mỗi trang |
| `add_to_cart`, `remove_from_cart` | Postgres + state | Khách chốt mua hoặc đổi ý (chỉ `graph.py`) |
| `search_policy` | Qdrant | Đổi trả, giao hàng, trả góp, bảo hành, lỗi máy |
| `web_search` | Tavily | Đánh giá, tin tức ngoài dữ liệu nội bộ |

- Tên máy được chuẩn hóa (bỏ dấu, viết tắt như `ip`, `promax`, `17pro`). Khi nhiều sản phẩm khớp, tool trả danh sách để model hỏi lại khách.
- Tồn kho chỉ trả `còn hàng` / `sắp hết` / `hết hàng`, không trả số lượng.
- Mọi tool tự bắt lỗi hệ thống và báo cho model, để lượt chat không bị hỏng.

### RAG chính sách

- 5 file trong `data_schema/policies/` được tách theo mục `##`/`###` (mục FAQ tách theo từng cặp hỏi/đáp), tổng cộng 66 đoạn, mỗi đoạn tối đa 900 ký tự.
- `search_policy` lấy 3 đoạn gần nhất, bỏ đoạn có điểm dưới 0,25 hoặc kém đoạn tốt nhất quá 0,08 (`config.py`), để câu hỏi lạc đề không nhận chính sách không liên quan.
- Kết quả được gom theo file, mỗi file kèm hướng dẫn trả lời riêng (`POLICY_PROMPTS` trong `prompts.py`). Khi thêm file chính sách mới phải thêm prompt tương ứng (có test kiểm tra).
- Khi khởi động, web app so ID các đoạn trong Qdrant với các file `.md`; nếu khác thì nạp lại.

### Chuỗi LLM dự phòng

| Thứ tự | Model |
|---|---|
| Chính | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` |
| Dự phòng 1 | NVIDIA `nvidia/nemotron-3.5-super-vl-120b-a12b` (bật thinking) |
| Dự phòng 2 | `gemini-3.1-flash-lite` |
| Dự phòng 3 | `gemini-3.5-flash-lite` |
| Dự phòng 4 | Groq `qwen/qwen3.8-27b` |

`FallbackMiddleware` thử lần lượt các model tới khi có model trả lời. Model gặp lỗi lặp lại sẽ tạm bị bỏ qua: hết quota ngày hoặc lỗi cấu hình (sai key, sai tên model) 1 giờ, giới hạn theo phút 1 phút, lỗi chứng chỉ SSL 10 phút. Lỗi tạm thời (mất mạng, 5xx) chỉ chuyển model cho lần gọi đó. Mỗi lần chuyển model đều ghi log (`docker compose logs api | grep LLM`).

Gói free của OpenRouter giới hạn khoảng 50 request/ngày (reset lúc 7:00 sáng giờ Việt Nam); mỗi câu hỏi tốn 2–3 request.

## Dữ liệu

Database `phone` (schema `data_schema/phone.sql`, dữ liệu `data_schema/output/phone_seed.sql`):

| Bảng | Nội dung |
|---|---|
| `products` | Điện thoại và phụ kiện: tên, hãng, mô tả, thông số (JSON) |
| `variants` | Phiên bản theo màu và dung lượng: SKU, giá (VND), tồn kho |
| `compatibility` | Cặp phụ kiện – điện thoại tương thích |

Dữ liệu được cào từ apple.com/vn và samsung.com/vn bằng `data_schema/scrape_phone.py`. Lưu ý:

- Tồn kho là số giả lập.
- Một số giá Samsung là ước tính; tool ghi "(giá tham khảo)" cho các SKU này.
- Chưa có thông số chi tiết cho dòng Galaxy S/Z.
- Chính sách của cửa hàng (lãi suất, phí, đối tác) là giả lập; phần bảo hành của hãng là tóm tắt tham khảo.

| Cập nhật | Cách làm |
|---|---|
| Sản phẩm | `cd data_schema && python scrape_phone.py --refresh`, rồi `docker compose run --rm db-seed` |
| Chính sách | Sửa file trong `data_schema/policies/`, rồi `docker compose up -d --build api` |

## Cấu trúc thư mục

```
sales_agent/
├── data_schema/
│   ├── phone.sql, scrape_phone.py
│   ├── output/                 # Dữ liệu đã cào và file seed
│   └── policies/               # Chính sách (.md), nạp vào Qdrant
├── src/phone_sales/
│   ├── config.py               # Cấu hình, ngưỡng RAG
│   ├── llm.py, fallback.py     # LLM và chuỗi dự phòng
│   ├── prompts.py              # System prompt, hướng dẫn trả lời chính sách
│   ├── tools.py                # Các tool
│   ├── conversation.py         # Trạng thái hội thoại và luật cập nhật
│   ├── graph.py                # Agent StateGraph (mặc định)
│   ├── agent.py                # Agent create_agent
│   ├── history.py              # Cắt gọn lịch sử trước khi gửi LLM
│   ├── data_prep.py            # Tách chính sách thành đoạn
│   ├── vector_db.py            # Embedding và Qdrant
│   ├── api.py                  # FastAPI
│   └── static/index.html       # Giao diện chat
├── tests/                      # Unit test
├── evals/                      # Bộ câu hỏi và script đánh giá
├── certs/                      # CA nội bộ cho mạng có SSL inspection (không commit)
├── Dockerfile, docker-compose.yml
├── langgraph.json
└── pyproject.toml, uv.lock, requirements.txt
```

## Kiểm thử và đánh giá

### Test

```bash
uv run pytest
```

118 test, không gọi LLM, mạng hay database thật (dùng model và dữ liệu giả). Khi thêm thư viện bằng `uv add`, cập nhật `requirements.txt` cho khớp (`tests/test_requirements.py` kiểm tra).

### Eval

Eval gọi LLM, Postgres, Qdrant, Tavily thật trên 37 câu hỏi cố định (`evals/dataset.jsonl`) và chấm điểm bằng Gemini. Chuỗi dự phòng bị tắt để điểm phản ánh đúng model chính.

```bash
docker compose up -d postgres qdrant
uv run python evals/run_eval.py --retrieval-only            # chỉ chấm truy xuất chính sách
uv run python evals/run_eval.py --target graph              # chấm graph.py (mặc định agent.py)
uv run python evals/run_eval.py --only product,acc-01       # lọc theo nhóm hoặc id
```

Cả bộ cần khoảng 90 request, vượt quota ngày của OpenRouter gói free, nên cần chia làm nhiều lần.

Kết quả gần nhất (`graph.py`, 30/10/2026). Truy xuất chính sách: hit@1 89%, hit@3 100%, MRR 0,95.

| Nhóm | Số câu | Gọi đúng tool | Đủ ý | Hành vi | Dựa trên dữ liệu | Điểm TB (1–5) |
|---|---|---|---|---|---|---|
| `policy` | 19 | 19/19 | 15/15 | – | 19/19 | 4,89 |
| `product` | 7 | 6/6 | 6/6 | – | 7/7 | 4,57 |
| `accessory` | 3 | 2/3 | 3/3 | – | 3/3 | 4,67 |
| `escalation` | 3 | 1/2 | – | 3/3 | 3/3 | 4,67 |
| `security` | 2 | 1/1 | – | 1/1 | 2/2 | 5,00 |
| `web` | 1 | – | – | – | 1/1 | 5,00 |
| `out_of_scope` | 2 | 2/2 | – | 2/2 | 2/2 | 5,00 |
| **Tổng** | **37** | **31/33** | **24/24** | **6/6** | **37/37** | **4,81** |

Độ trễ trung bình 15,8 giây mỗi câu do model chính là model suy luận. Kết quả chi tiết lưu ở `evals/results/` (không commit).

## Hạn chế

- State hội thoại nằm trong RAM (`InMemorySaver`): mất khi restart, chỉ chạy được 1 worker.
- Giỏ hàng chưa thành đơn hàng thật.
- Luật nhận diện từ khóa không hiểu mọi cách nói (viết không dấu, mỉa mai, viết tắt lạ).
- API chưa có xác thực và giới hạn request.
- Bộ eval nhỏ, chưa có câu hỏi nhiều lượt; judge là LLM nên có thể chấm sai.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| "Lỗi hệ thống khi tra cứu kho hàng" | Kiểm tra Postgres đang chạy; trên máy dùng cổng **5433** |
| Giá, sản phẩm chưa đổi sau khi cào lại | `docker compose run --rm db-seed` |
| `password authentication failed` sau khi đổi mật khẩu | Mật khẩu chỉ đặt lúc tạo volume: `docker compose down -v` rồi chạy lại |
| `search_policy` không tìm thấy gì | `uv run python -m phone_sales.vector_db` |
| `/chat` trả 429 hoặc 502 | Mọi LLM đều hết quota hoặc lỗi; xem `docker compose logs api \| grep LLM` |
| `/health` trả 503 | Kiểm tra Qdrant và `QDRANT_URL` |
| Sửa code nhưng web app không đổi | `docker compose up -d --build api` |
| `uvicorn` báo cổng 8000 đang dùng | Container `phone_sales_api` đang chạy; dùng `--port 8001` |
| Trong Docker báo `CERTIFICATE_VERIFY_FAILED` | Xem [Mạng có SSL inspection](#mạng-có-ssl-inspection) |

### Mạng có SSL inspection

Nếu mạng dùng proxy cấp lại chứng chỉ HTTPS bằng CA nội bộ, các LLM trong container sẽ lỗi chứng chỉ (chạy trên máy thì không, nhờ `truststore`). Xuất CA vào `certs/` rồi build lại:

```powershell
# Đổi 'CN=ten-CA*' thành tên CA nội bộ
$c = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -like 'CN=ten-CA*' } | Select-Object -First 1
"-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks') + "`n-----END CERTIFICATE-----" | Set-Content -Encoding ascii certs\internal-ca.crt
docker compose up -d --build api
```

### Windows

- **`OSError: [WinError 1114] ... c10.dll`** (thường do antivirus nạp bản `msvcp140.dll` cũ): chạy lệnh sau mỗi khi tạo lại `.venv`:

  ```bash
  uv run python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[-1], '_msvcp140_preload.pth').write_text(\"import sys, ctypes; sys.platform == 'win32' and ctypes.WinDLL('msvcp140.dll')\n\")"
  ```

- **`UnicodeEncodeError`** khi in tiếng Việt: đặt `PYTHONUTF8=1`.
- **curl trên Git Bash** có thể gửi sai mã hóa tiếng Việt: dùng PowerShell `Invoke-RestMethod` hoặc Swagger UI.
