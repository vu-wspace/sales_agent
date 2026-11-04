# Phone Sales Agent

Trợ lý chat tiếng Việt cho cửa hàng bán lẻ iPhone, Samsung Galaxy và phụ kiện, xây dựng với LangChain/LangGraph. Dự án demo phục vụ học tập.

## Tính năng

- Tư vấn điện thoại theo ngân sách và nhu cầu; báo giá, màu, dung lượng, tình trạng hàng từng phiên bản.
- Tìm phụ kiện tương thích (ốp lưng, cường lực, sạc, cáp, S Pen…).
- Trả lời chính sách đổi trả, giao hàng, trả góp, bảo hành bằng RAG.
- Nhớ trạng thái hội thoại bán hàng (nhu cầu, máy đã giới thiệu, lo ngại, cảm xúc) và quản lý giỏ hàng.
- Bán thêm: gợi ý dung lượng lớn hơn khi nhu cầu cần (kèm số tiền chênh), máy nhỉnh hơn ngân sách; khi khách chốt mua thì mời sạc chính hãng nếu máy không kèm sạc, rồi ốp lưng, cường lực.
- Đặt hàng ngay trong khung chat: khách xác nhận thì hiện form (tên, số điện thoại, địa chỉ), đơn lưu vào Postgres, tóm tắt đơn lấy từ database.
- Phiên chat: nút "Cuộc trò chuyện mới" kết thúc cuộc cũ (xóa lịch sử, giỏ hàng) và mở cuộc mới; khách đã đặt hàng trên trình duyệt đó được chào lại theo tên, agent biết đơn gần nhất.
- Gửi email xác nhận đơn hàng (HTML + chữ thuần) khi khách có nhập email; khi phát triển, email được Mailpit giữ lại, không gửi ra ngoài.
- Chuẩn hóa tin nhắn (Unicode, dấu kiểu cũ, viết tắt "ko", "dc", "20tr"…) và kiểm tra câu trả lời không bịa số liệu (giá, thông số, chính sách).
- Chuyển nhân viên khi khách khiếu nại, đơn có vấn đề hoặc khách bực bội.
- Stream trạng thái và câu trả lời theo thời gian thực.
- Tự chuyển sang LLM dự phòng khi model chính lỗi hoặc hết quota.

| Thành phần | Công nghệ |
|---|---|
| Agent | LangGraph `StateGraph` (`graph.py`) hoặc LangChain `create_agent` (`agent.py`) |
| LLM | 9Router (chính, chạy trên máy) → OpenRouter → NVIDIA → Gemini → Groq (dự phòng) |
| Dữ liệu sản phẩm | PostgreSQL 18: 23 điện thoại, 134 phụ kiện, 377 phiên bản |
| Chính sách (RAG) | Qdrant + embedding `bkai-foundation-models/vietnamese-bi-encoder` |
| Web | FastAPI + trang chat HTML một file |
| Email | fastapi-mail (SMTP) + template Jinja2; Mailpit khi phát triển |
| Triển khai | Docker Compose |

## Bắt đầu nhanh

Yêu cầu: Docker, các API key trong [Cấu hình](#cấu-hình) và 9Router chạy trên máy (cổng 20128; không có thì agent dùng model dự phòng, nhưng mỗi lần gọi phải chờ lỗi kết nối trước). Khi phát triển trên máy cần thêm Python 3.12.15 và [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env            # điền API key
docker compose up -d --build    # Postgres (tự nạp dữ liệu) + Qdrant + Mailpit + web app (tự nạp chính sách)
```

| Địa chỉ | Nội dung |
|---|---|
| http://localhost:8000 | Trang chat |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:6333/dashboard | Dashboard Qdrant |
| http://localhost:8025 | Mailpit: email xác nhận đơn hàng (không gửi ra Internet) |

Lần build đầu mất khoảng 8 phút, chủ yếu để tải model embedding (~540 MB). Các lần sau chỉ build lại vài giây nếu không đổi thư viện.

### Chạy trên máy

Postgres, Qdrant, Mailpit chạy bằng Docker, code chạy trên máy:

```bash
uv sync
docker compose up -d postgres qdrant mailpit
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
docker compose run --rm db-seed                           # nạp lại dữ liệu sản phẩm (giữ đơn hàng, không mất volume)
docker compose exec api python -m phone_sales.vector_db   # nạp lại chính sách
docker compose down                                       # dừng, giữ dữ liệu
docker compose down -v                                    # dừng và xóa dữ liệu
docker builder prune                                      # xóa build cache cũ (mỗi lần build lại tích thêm vài GB)
```

| Container | Cổng | Vai trò |
|---|---|---|
| `phone_db` | 5433 | PostgreSQL, database `phone` |
| `qdrant` | 6333, 6334 | Vector DB |
| `phone_sales_api` | 8000 | Web app |
| `phone_mailpit` | 1025, 8025 | Bắt email khi phát triển: SMTP cổng 1025, xem email tại http://localhost:8025 |

## Cấu hình

Biến môi trường đặt trong `.env` (mẫu ở `.env.example`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ROUTER9_BASE_URL` | `http://localhost:20128/v1` | 9Router, LLM chính (API kiểu OpenAI, chạy trên máy) |
| `ROUTER9_API_KEY` | (bắt buộc) | Key tạo trong 9Router |
| `ROUTER9_MODEL` | `super_duper_free_combo` | Tên model hoặc combo trên 9Router |
| `OPENROUTER_API_KEY` | (bắt buộc) | LLM dự phòng ([openrouter.ai](https://openrouter.ai/keys)) |
| `NVIDIA_API_KEY` | (bắt buộc) | LLM dự phòng ([build.nvidia.com](https://build.nvidia.com)) |
| `GOOGLE_API_KEY` | (bắt buộc) | Gemini: LLM dự phòng và judge của eval ([AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | (bắt buộc) | LLM dự phòng ([console.groq.com](https://console.groq.com)) |
| `TAVILY_API_KEY` | (bắt buộc) | Tool `web_search` ([tavily.com](https://tavily.com)) |
| `POSTGRES_PASSWORD` | `postgres` | Mật khẩu Postgres, chỉ có hiệu lực khi tạo volume lần đầu |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/phone` | Database khi chạy trên máy |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant khi chạy trên máy |
| `AGENT_IMPL` | `graph` | Web app dùng `graph.py` (có trạng thái, giỏ hàng) hay `agent.py` |
| `AUTO_INGEST` | `true` | Tự nạp chính sách khi Qdrant rỗng hoặc đã cũ |
| `PII_KEY` | (bắt buộc để đặt hàng) | Khóa Fernet mã hóa số điện thoại, email của khách trong database, xem [Thông tin cá nhân](#thông-tin-cá-nhân-pii). Tạo: `uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. **Sao lưu khóa**: mất khóa là không đọc lại được dữ liệu đã mã hóa |
| `SMTP_HOST` | (trống: không gửi email) | Máy chủ SMTP gửi email xác nhận đơn. Chạy trên máy với Mailpit: `localhost` |
| `SMTP_PORT` | `1025` | Cổng SMTP (Mailpit 1025; Gmail 587) |
| `SMTP_USER`, `SMTP_PASSWORD` | trống | Đăng nhập SMTP (để trống với Mailpit) |
| `SMTP_STARTTLS`, `SMTP_SSL` | `false` | STARTTLS (cổng 587) hoặc SSL (cổng 465) |
| `MAIL_FROM` | `PhoneShop Demo <no-reply@phoneshop.example>` | Người gửi; phần tên cũng là tên cửa hàng trong email. Không dùng tên miền `.local` (thư viện kiểm tra email từ chối) |
| `LANGSMITH_*` | tắt | Trace LangSmith (tùy chọn) |

Trong Docker, compose tự đặt `DATABASE_URL`, `QDRANT_URL`, `SMTP_HOST=mailpit` trỏ tới các container và `ROUTER9_BASE_URL` trỏ tới `host.docker.internal` (9Router chạy trên máy host).

## API

| Method | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/` | Trang chat |
| `POST` | `/session` | Kết thúc cuộc trò chuyện cũ, mở cuộc mới; chào lại khách quen |
| `POST` | `/chat` | Gửi tin nhắn, nhận câu trả lời đầy đủ |
| `POST` | `/chat/stream` | Như `/chat`, trả Server-Sent Events |
| `POST` | `/order` | Gửi form đặt hàng, tạo đơn từ giỏ hàng của hội thoại |
| `POST` | `/order/quote` | Form hỏi giá: tạm tính, giảm giá theo mã khuyến mãi hoặc khuyến mãi tự động, tổng tiền |
| `POST` | `/order/cancel` | Khách bấm "Hủy đặt hàng" trên form |
| `GET` | `/health` | Trạng thái Qdrant và số đoạn chính sách |

### `POST /session`

```jsonc
// request: cả 2 trường đều không bắt buộc
{"end_thread_id": "5f0c…",                       // cuộc trò chuyện cũ: xóa lịch sử, giỏ hàng khỏi bộ nhớ
 "customer_token": "KH0B3F1C02.0f0…"}            // mã khách quen, lấy từ response của POST /order
// response: khách quen
{"thread_id": "9a1d…", "returning": true, "customer_token": "KH0B3F1C02.0f0…",
 "greeting": "Dạ, em chào anh/chị Nguyễn Văn A, rất vui được gặp lại anh/chị nha. Lần trước anh/chị đã đặt iPhone 17 512GB Đen. …"}
// response: khách mới, mã sai hoặc khách chưa có đơn
{"thread_id": "9a1d…", "returning": false, "customer_token": null, "greeting": "Dạ, em chào anh/chị. Bên em tư vấn …"}
```

- Trang chat gọi khi mở trang và khi bấm "Cuộc trò chuyện mới". Lời chào là template, không gọi LLM.
- `customer_token` = mã khách + chữ ký HMAC (khóa sinh từ `PII_KEY`), trình duyệt giữ trong `localStorage`. Có chữ ký nên không đổi được sang mã khách khác để xem tên, đơn của người khác. Mã sai hoặc khách không còn đơn thì trả `null`, trang chat xóa mã đã lưu. Database lỗi thì vẫn mở phiên, chào như khách mới, giữ mã.
- Khách quen: lời chào và hồ sơ (`orders.customer_profile`: tên, số đơn, đơn gần nhất gồm mã, ngày, tên món) được ghi vào thread mới (`customer` trong trạng thái hội thoại). Không có số điện thoại, email, địa chỉ trong hồ sơ vì hồ sơ được gửi cho LLM.
- `thread_id` của `/chat` vẫn có thể bỏ trống như trước (gọi API trực tiếp không cần `/session`).

### `POST /chat`

```jsonc
// request: bỏ trống thread_id để bắt đầu hội thoại mới
{"message": "ok chốt galaxy s26 256gb màu đen", "thread_id": "5f0c…"}

// response
{"thread_id": "5f0c…", "answer": "Dạ em đã thêm Galaxy S26 256GB Đen Classic vào giỏ…",
 "tools": ["add_to_cart"], "stage": "closing",
 "cart": [{"variant_id": "SM-S942BZKCXXV", "name": "Galaxy S26", "storage": "256GB",
           "color": "Đen Classic", "price": 29990001, "quantity": 1}],
 "handoff": false, "order_form": false}
```

- `message` dài 1–1000 ký tự, `thread_id` tối đa 100 ký tự.
- `stage`, `cart`, `handoff`, `order_form` chỉ có giá trị khi `AGENT_IMPL=graph`. `order_form: true` là lượt agent gọi `show_order_form`: giao diện hiện form đặt hàng dưới câu trả lời.
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

### `POST /order`

```jsonc
// request: thread_id của hội thoại có giỏ hàng; giỏ hàng lấy từ state trên server, không nhận từ trình duyệt
{"thread_id": "5f0c…", "name": "Nguyễn Văn A", "phone": "0912 345 678",
 "address": "123 Lê Hồng Phong, Quy Nhơn, Gia Lai", "email": "nguyenvana@gmail.com", "promo_code": "GIAM500K",
 "fulfillment_type": "delivery", "payment": "cod"}

// response
{"thread_id": "5f0c…", "order_id": "DH3F9A1C2B",
 "order": {"order_id": "DH3F9A1C2B", "name": "Nguyễn Văn A", "phone": "0912 *** 678", "email": "ng***@gmail.com",
           "address": "…",
           "fulfillment": "Giao tận nơi", "payment": "Thanh toán khi nhận hàng",
           "items": [{"name": "Galaxy S26", "variant": "256GB · Đen Classic", "qty": 1, "unit_price": 29990001}],
           "total": 29990001, "note": "Nhân viên cửa hàng sẽ gọi điện xác nhận đơn trước khi giao."},
 "thanks": "Em cảm ơn anh Nguyễn Văn A…",
 "answer": "Đơn hàng **#DH3F9A1C2B** đã được tạo.
Khách hàng: Nguyễn Văn A
…

Em cảm ơn anh Nguyễn Văn A…"}
```

- `order` là dữ liệu đơn đọc lại từ database, trang chat hiện thành thẻ xác nhận (mã đơn, người nhận, sản phẩm, tổng tiền); `thanks` hiện thành bong bóng riêng. `answer` là bản chữ của cả hai, được ghi vào lịch sử hội thoại.
- Kiểm tra giống form: tên ≥ 2 ký tự, số điện thoại `0`/`+84` + 9 số (bỏ dấu cách, chấm, gạch), địa chỉ ≥ 10 ký tự; `email` không bắt buộc (bỏ trống hoặc `null`), có nhập thì phải đúng dạng `x@y.z`; `promo_code` không bắt buộc (không phân biệt hoa thường). Hiện chỉ nhận `delivery` và `cod`.
- `order` có thêm `subtotal`, `discount`, `promo` (mã đã áp dụng); không có giảm giá thì `discount` = 0.
- Số điện thoại, email trong `order` và `answer` đã che (xem [Thông tin cá nhân](#thông-tin-cá-nhân-pii)).
- Có `email` thì sau khi trả response, server gửi email xác nhận tới địa chỉ đó (xem [Email xác nhận đơn hàng](#email-xác-nhận-đơn-hàng)); response không chờ email gửi xong.
- `customer_token`: mã khách đã ký, trang chat lưu lại để phiên sau chào lại khách (xem [`POST /session`](#post-session)).
- Mã lỗi: `422` thông tin sai, `409` giỏ trống, hết hàng hoặc mã khuyến mãi không dùng được (kèm lý do), `503` không ghi được database. Có lỗi thì không gửi email.

### `POST /order/quote`

```jsonc
// request: mã khách nhập ở form (null = chỉ áp khuyến mãi tự động)
{"thread_id": "5f0c…", "promo_code": "SAMSUNG5"}
// response
{"subtotal": 29990001, "discount": 1499500, "total": 28490501, "promo": "SAMSUNG5",
 "promo_text": "Tuần lễ Samsung: giảm 5% (tối đa 2.000.000đ) cho sản phẩm Samsung, đến hết 30/11/2026 (mã SAMSUNG5)"}
```

Tính từ giỏ hàng của hội thoại trên server, không trừ tồn kho hay lượt dùng mã. Chưa biết khách nên chưa kiểm tra số lần khách đã dùng mã; `POST /order` kiểm tra lại đầy đủ. Mã không dùng được trả `409` kèm lý do (không tồn tại, hết hạn, hết lượt, chưa đủ đơn tối thiểu, không áp dụng cho sản phẩm trong giỏ).

### `POST /order/cancel`

```jsonc
// request
{"thread_id": "5f0c…"}
// response: câu cố định CANCEL_MESSAGE trong orders.py, không gọi LLM
{"thread_id": "5f0c…", "answer": "Thật tiếc, em đã hủy yêu cầu đặt hàng của anh/chị. Anh/chị có cần em hỗ trợ hay tư vấn gì thêm nữa không?"}
```

Không tạo đơn, không đụng database. Ghi vào lịch sử hội thoại "Hủy đặt hàng" + `answer`, và làm trống giỏ hàng để agent không tiếp tục mời chốt đơn hay bán kèm cho giỏ đã hủy (khách muốn mua lại thì chọn máy lại). Thiếu `thread_id` trả `422`.

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
    api -- "POST /order" --> orders["Đơn hàng<br/>orders.py"] --> postgres
    api -. "BackgroundTasks" .-> mailer["Email xác nhận<br/>mailer.py"] -- "SMTP" --> mailpit(["Mailpit / nhà cung cấp email"])
```

Luồng một lượt chat trong `graph.py`:

```mermaid
flowchart LR
    start((START)) --> track_user --> model
    model -- "có tool_call" --> tools --> track_tools --> model
    model -- "trả lời xong" --> track_reply --> finish((END))
```

1. `track_user` đọc tin nhắn của khách: nhu cầu, lo ngại, cảm xúc, có cần chuyển nhân viên không. Từ khóa chuyển nhân viên (`HANDOFF`) khớp cả bản bỏ dấu (`_has_loose`: "gap quan ly", "CHUYEN NV"): gặp nhân viên/quản lý/người phụ trách, gọi staff, khiếu nại, giao sai/trễ/thiếu, bị trừ tiền 2 lần, sai màu, máy đã hỏng; không tính "quản lý pin/bộ nhớ", "nhắn nhân viên gọi xác nhận".
2. `model` gọi LLM với system prompt, tóm tắt trạng thái hội thoại và lịch sử đã cắt gọn (4 lượt gần nhất).
3. `tools` chạy tool; `track_tools` cập nhật nhu cầu từ tham số tool và tính lại bước bán hàng. Model được gọi nhiều tool song song trong 1 bước (9Router và OpenRouter bật `parallel_tool_calls=True`, model dự phòng khác mặc định cũng song song), vd tra giá 2 máy khi so sánh, thêm máy và củ sạc cùng lúc; các khóa state mà tool ghi (`cart`, `order_form`, `offered_alternatives`) đều có reducer nên nhận được nhiều cập nhật cùng bước.
4. `track_reply` ghi lại các sản phẩm thật sự được giới thiệu trong câu trả lời. Câu trả lời báo đã/sẽ chuyển nhân viên, quản lý (`announces_handoff`, vd prompt quy tắc 12 với máy lỗi cần đổi trả) thì bật `handoff` thật, để lời báo không thành lời hứa suông khi từ khóa của khách không khớp. Không tính câu hỏi, lời mời, điều kiện, phủ định ("anh/chị có muốn em chuyển cho nhân viên không?", "nếu cần em sẽ chuyển…", "em chưa chuyển…") và việc bình thường của đơn ("chuyển đơn cho nhân viên xác nhận", "bộ phận kho"). Chạy trên câu trả lời của 2 lần simulator: 9 câu trước đây bị tính nhầm giờ không còn, thêm 2 câu "ghi nhận … và chuyển cho nhân viên" trước đây bị bỏ sót.

`agent.py` dựng bằng `create_agent`, dùng chung tool tra cứu và prompt nhưng không có trạng thái hội thoại và giỏ hàng; model chính vẫn là OpenRouter (`llm.open_llm`), không qua 9Router.

### Đặt hàng

1. Khách nêu tên máy mà chưa nói phiên bản ("Tui muốn mua iPhone 17", "cho tui coi Galaxy S26"): agent gọi `get_product_details`, liệt kê mỗi dung lượng 1 dòng (giá, màu, còn hàng) và hỏi khách chọn phiên bản. Khách nói đủ máy, dung lượng, màu thì sang bước 2 luôn.
2. Lượt khách chọn máy ("chốt S26 256GB đen"): agent `add_to_cart` rồi bán kèm 3 lượt (xem [Bán thêm](#bán-thêm-up-sell-cross-sell)): mời củ sạc (máy không kèm) → mời ốp lưng, cường lực → xác nhận đơn, mỗi lượt 1 loại. `show_order_form` từ chối mở form trong lượt vừa thêm hàng vào giỏ và khi bán kèm chưa xong.
3. Khách xác nhận ("xác nhận đặt hàng", "chốt đơn"): agent gọi `show_order_form`, giao diện hiện form (`index.html`, theo thiết kế `static/order_form.html`) với giỏ hàng và tổng tiền.
4. Khách gửi form → `POST /order` (`orders.py`), trong 1 transaction:
   - trừ tồn kho, thiếu hàng thì hủy cả đơn;
   - khách cũ nhận theo số điện thoại (tra bằng `phone_hash`; cập nhật tên, địa chỉ, email nếu có nhập), khách mới thì tạo `customers` (mã `KH…`);
   - email luôn lưu vào đơn (`orders.recipient_email`); chỉ lưu vào `customers.email` khi email chưa thuộc khách khác (`uq_customers_email_hash` không cho 2 khách trùng email), để đơn không bị lỗi;
   - số điện thoại, email lưu dạng mã hóa (xem [Thông tin cá nhân](#thông-tin-cá-nhân-pii));
   - tính giảm giá (`promotions.quote`, khóa dòng khuyến mãi bằng `FOR UPDATE`): mã khách nhập, hoặc khuyến mãi tự động giảm nhiều nhất; mã không dùng được thì không ghi gì cả;
   - ghi `orders` (mã `DH…`, `subtotal`, `discount_amount`, `promo_id`, `total` = tạm tính − giảm giá) và `order_items`; có khuyến mãi thì ghi `promotion_usages` và tăng `used_count`.
5. Câu trả lời = tóm tắt đơn viết bằng template, đọc lại từ database (khách hàng, địa chỉ, sản phẩm, tổng tiền; số điện thoại, email đã che) + 1–2 câu cảm ơn do LLM viết. LLM chỉ nhận tên khách, không viết số liệu; LLM lỗi thì dùng câu cảm ơn cố định. Trang chat hiện tóm tắt thành thẻ xác nhận (`.receipt`: mã đơn, thông tin nhận hàng, sản phẩm, tổng tiền), lời cảm ơn là bong bóng riêng.
6. Đơn được ghi vào lịch sử hội thoại, giỏ hàng làm trống; gửi form lần nữa nhận `409`.
7. Gửi email xác nhận (xem [Email xác nhận đơn hàng](#email-xác-nhận-đơn-hàng)) sau khi đã trả kết quả cho khách.
8. Khách bấm **Hủy đặt hàng** trên form thay vì gửi → `POST /order/cancel`: bỏ form, hiện câu "Thật tiếc, em đã hủy yêu cầu đặt hàng… Anh/chị có cần em hỗ trợ hay tư vấn gì thêm nữa không?", giỏ hàng làm trống, không tạo đơn.

Form có ô **Mã khuyến mãi** và nút **Áp dụng** (Enter trong ô cũng là áp dụng, không gửi cả form). Mở form là hỏi giá ngay (`POST /order/quote`) để hiện khuyến mãi tự động (nếu có). Thẻ xác nhận đơn có thêm dòng tạm tính và giảm giá.

| Trạng thái nút | Hiển thị |
|---|---|
| Chưa nhập mã | "Áp dụng"; bấm khi ô trống thì báo "Nhập mã khuyến mãi" |
| Đang kiểm tra | "Kiểm tra…", nút mờ, không bấm được |
| Mã không dùng được | Hiện lý do dưới ô (không tồn tại, hết hạn, hết lượt…), giá quay về như không có mã |
| Đã áp dụng | Ô mã khóa lại, tô màu nhấn; nút thành "Bỏ mã"; hiện dòng giảm giá, tổng mới và mô tả khuyến mãi |
| Bấm "Bỏ mã" | Xóa mã, mở lại ô nhập, giá quay về như không có mã (vẫn giữ khuyến mãi tự động nếu có) |

Ô mã và nút cùng chiều cao; nút rộng cố định 104px để đổi chữ không làm ô mã co giãn. Mã đã áp dụng chỉ là xem trước: `POST /order` kiểm tra lại đầy đủ khi đặt hàng.

### Email xác nhận đơn hàng

`POST /order` tạo đơn xong thì lên lịch `mailer.send_order_email` bằng `BackgroundTasks`: khách nhận kết quả ngay, email gửi sau (fastapi-mail, SMTP bất đồng bộ). Tiêu đề "Xác nhận đơn hàng #DH… - PhoneShop Demo" (tên cửa hàng lấy từ `MAIL_FROM`).

- Đọc lại đơn từ database (`orders.get_order`, dữ liệu qua `orders.view`), không dùng dữ liệu form: email luôn khớp đơn đã lưu (tên món, giảm giá, tổng tiền), giống thẻ xác nhận trên khung chat.
- Template `src/phone_sales/templates/order_confirmation.html` (Jinja2, tự escape tên, địa chỉ khách nhập): CSS viết inline và bố cục bảng vì Gmail, Outlook bỏ qua thẻ `<style>`. Kèm bản chữ thuần (`render_text`) cho trình đọc email không hiển thị HTML. Template nằm ngoài `static/` để không bị phục vụ công khai.
- Kết quả ghi vào `orders.email_status`: `pending` (đang chờ) → `sent`, `failed` (lỗi SMTP, chỉ ghi mã đơn vào log), hoặc `skipped` (khách không nhập email, chưa đặt `SMTP_HOST`). Gửi lỗi không ảnh hưởng đơn đã tạo.
- Nội dung: lời chào theo tên người nhận, bảng sản phẩm (tên, phiên bản, số lượng, thành tiền), tạm tính và giảm giá (nếu có), tổng cộng, cách nhận hàng, địa chỉ (chỉ khi giao tận nơi), thanh toán, ghi chú nhân viên gọi xác nhận. Không đưa số điện thoại, email của khách vào nội dung.
- Đây là email **xác nhận đơn hàng**, không phải hóa đơn GTGT: hóa đơn điện tử phải phát hành qua nhà cung cấp dịch vụ hóa đơn (Nghị định 123/2020/NĐ-CP).

Thử trên máy: `docker compose up -d`, đặt một đơn có điền email, mở http://localhost:8025 để xem bản HTML và chữ thuần. Xem trạng thái gửi: `docker exec phone_db psql -U postgres -d phone -c "SELECT order_id, email_status FROM orders ORDER BY created_at DESC LIMIT 5"`. Chạy thật thì đổi `SMTP_*` sang Gmail (App Password) hoặc dịch vụ email giao dịch (Amazon SES, SendGrid, Resend), code giữ nguyên.

### Khuyến mãi

Bảng `promotions` (`promotions.py`):

| Cột | Ý nghĩa |
|---|---|
| `code` | Mã khách nhập (không phân biệt hoa thường). `NULL` = khuyến mãi tự động, không cần mã |
| `discount_type`, `value`, `max_discount` | `percent` (giảm `value`%, tối đa `max_discount`) hoặc `fixed` (giảm `value` đồng); không giảm quá phần giỏ thuộc phạm vi |
| `scope`, `scope_value` | `all` cả đơn; `brand` theo hãng (`Apple`, `Samsung`); `product` theo `product_id`; `category` theo loại (`phone`, `accessory`). Giảm trên phần giỏ thuộc phạm vi; scope khác thì không áp dụng |
| `min_order` | Đơn tối thiểu, so với tạm tính cả đơn |
| `starts_at`, `ends_at`, `is_active` | Thời gian chạy, bật/tắt |
| `usage_limit`, `used_count`, `per_customer_limit` | Tổng số lượt, số lượt đã dùng, số lần mỗi khách (đếm trong `promotion_usages`; khách nhận theo số điện thoại) |

- Agent đọc khuyến mãi bằng tool `get_promotions` (lọc theo sản phẩm khách đang xem) khi khách hỏi khuyến mãi và ở bước 1 khi khách do dự; chỉ nêu khuyến mãi có trong kết quả, không có thì không nhắc.
- Khuyến mãi mẫu để thử: `data_schema/promotions_sample.sql` (GIAM500K, SAMSUNG5, giảm 10% phụ kiện tự động, HETHAN đã hết hạn; dữ liệu demo). Nạp: `docker exec -i phone_db psql -U postgres -d phone < data_schema/promotions_sample.sql`.

### Trạng thái hội thoại

`AgentState` (`conversation.py`) được lưu theo `thread_id` và cập nhật bằng luật từ khóa tiếng Việt (không gọi thêm LLM):

| Trường | Ý nghĩa |
|---|---|
| `stage` | Bước bán hàng: `discovery` → `recommending` → `comparing` → `closing`, hoặc `handoff` |
| `slots` | Nhu cầu: ngân sách, hãng, mục đích, dung lượng, màu |
| `shown_products` | Sản phẩm đã giới thiệu |
| `cart` | Giỏ hàng (xem [Nhiều tool cập nhật state cùng lúc](#nhiều-tool-cập-nhật-state-cùng-lúc)) |
| `objections` | Lo ngại của khách (giá, pin, nhiệt độ, camera…) |
| `sentiment_history` | Cảm xúc từng lượt |
| `handoff` | Cần chuyển nhân viên |
| `cross_sell` | Các bước bán kèm đã làm: `charger`, `accessories`, `confirm`; đặt hàng hoặc hủy đơn thì làm lại từ đầu |
| `declined` | Tin nhắn mới nhất của khách là từ chối ("không cần", "khỏi"…) |
| `order_form` | Lượt này hiện form đặt hàng |
| `orders` | Mã các đơn đã đặt trong hội thoại |
| `objection`, `objection_type`, `objection_step` | Phản đối lượt này (`price`, `competitor`, `hesitation`), loại đang xử lý và đã xử lý bao nhiêu lượt |
| `offered_alternatives` | Máy rẻ hơn đã gợi ý (không gợi ý lại) |
| `customer` | Khách quen (`POST /session`): tên, số đơn, đơn gần nhất. `customer_hint` dặn agent gọi khách theo tên, không chào lại từ đầu, không tự nhắc chi tiết đơn cũ; khách hỏi phụ kiện mà không nói máy thì hỏi có phải cho máy đã mua không |
| `wanted` | Phụ kiện khách đã tự yêu cầu mua `{loại: tin nhắn}` (xem [Bán thêm](#bán-thêm-up-sell-cross-sell)); khách bỏ món khỏi giỏ thì quên loại đó, đặt/hủy đơn thì làm lại |

Mỗi bước có hướng dẫn riêng được đưa vào system prompt, vd `closing`: xác nhận giỏ và tổng tiền, hỏi khách có chốt đơn không, khách xác nhận thì mở form đặt hàng; không tự nêu phí giao hàng hay kỳ hạn trả góp.

#### Nhiều tool cập nhật state cùng lúc

Model có thể gọi nhiều tool song song trong 1 bước (vd "lấy S26 với củ sạc 60W" → 2 lần `add_to_cart`). Khóa state khai báo thường (`cart: list[dict]`) chỉ nhận 1 giá trị mỗi bước, 2 tool cùng ghi thì LangGraph báo `InvalidUpdateError: At key 'cart': Can receive only one value per step`. Khóa khai báo `Annotated[kiểu, reducer]` thì LangGraph gộp lần lượt: `giá trị = reducer(giá trị cũ, giá trị mới)`.

Chỉ thêm `Annotated` là hết lỗi, nhưng reducer phải đúng nghĩa của dữ liệu. Với giỏ hàng, reducer "lấy giá trị mới nhất" sẽ làm mất món mà không báo lỗi: 2 tool cùng đọc giỏ trống ở đầu bước, tool 1 trả cả giỏ `[máy]`, tool 2 trả cả giỏ `[củ sạc]`, kết quả chỉ còn `[củ sạc]`. Vì vậy tool giỏ hàng trả **thao tác** thay vì cả giỏ, và `cart_update` áp lần lượt: `[]` → `[máy]` → `[máy, củ sạc]`.

| Khóa | Reducer | Giá trị mới |
|---|---|---|
| `cart` | `cart_update` | `{'add': món}` thêm (trùng phiên bản thì cộng số lượng), `{'remove': [variant_id]}` bỏ; `list` thay cả giỏ (`api.py` ghi `[]` sau khi đặt/hủy đơn) |
| `order_form` | `last` | Lấy giá trị mới nhất: chỉ là đúng/sai, gọi `show_order_form` 2 lần vẫn cùng kết quả |
| `offered_alternatives`, `shown_products`, `objections` | `append_unique` | Nối thêm, bỏ trùng |
| `sentiment_history`, `orders` | `append` | Nối thêm, giữ trùng (mỗi lượt / mỗi đơn 1 phần tử) |
| `messages` | `add_messages` | Nối thêm tin nhắn (của LangGraph) |
| `slots` | `merge_slots` | Gộp nhu cầu, giá trị mới ghi đè, danh sách thì cộng dồn |
| `cross_sell` | `steps` | Nối bước bán kèm; `None` thì làm lại từ đầu |

Khóa không có reducer (`stage`, `handoff`, `declined`, `objection`…) chỉ do node của graph ghi (mỗi bước 1 lần), tool không ghi. Thêm tool mới ghi vào state thì khóa đó phải có reducer. 9Router và OpenRouter bật `parallel_tool_calls=True` (`graph.py`), model dự phòng khác mặc định cũng gọi song song. Test: `test_graph_parallel_add_to_cart_keeps_both`, `test_cart_update_applies_each_change` (`tests/test_conversation.py`).

### Bán thêm (up-sell, cross-sell)

Tool trả sẵn dữ liệu để model gợi ý bằng số liệu thật, không tự tính hay đoán:

| Loại | Dữ liệu | Khi nào model gợi ý |
|---|---|---|
| Dung lượng lớn hơn | `get_product_details` có lọc dung lượng thì kèm dòng `Dung lượng lớn hơn còn hàng: 512GB từ 45.499.000đ (thêm 6.500.000đ)…` | Khách cần nhiều bộ nhớ (quay video, chơi game…) mà chọn bản thấp |
| Máy nhỉnh hơn ngân sách | `search_products` kèm tối đa 2 máy vượt ngân sách không quá 10%, ghi số tiền vượt | Máy đó hợp nhu cầu hơn rõ rệt; chỉ nhắc 1 máy |
| Phụ kiện | `find_accessories` | Sau khi khách cho điện thoại vào giỏ, theo các bước bên dưới |

Mỗi câu trả lời tối đa 1 gợi ý bán thêm.

**Bán kèm phụ kiện** (`cross_sell_step`, `cross_sell_hint` trong `conversation.py`): tối đa 3 bước, mỗi lượt chat mời 1 loại, không mời khi khách đang bực. Cuối mỗi lượt `track_reply` ghi bước vừa làm vào `cross_sell`, lượt sau sang bước kế tiếp.

| Bước | Khi nào | Model làm gì |
|---|---|---|
| 1. `charger` | Lượt khách chọn máy, máy không kèm củ sạc và giỏ chưa có củ sạc | Chỉ mời củ sạc, chưa mời ốp lưng, cường lực. Hộp đã có cáp: nói rõ, mời 2–3 củ sạc chính hãng không kèm cáp, từ rẻ tới đắt. Hộp không có cáp: 1 củ sạc + 1 cáp |
| 2. `accessories` | Khách lấy củ sạc hoặc không (máy đã kèm củ sạc thì đây là bước đầu, ngay lượt chọn máy) | Thêm củ sạc nếu khách chọn, mời 2–3 món ốp lưng, cường lực chính hãng (không nhắc lại củ sạc). Khách đã tự lấy ốp/cường lực thì bỏ qua bước này |
| 3. `confirm` | Khách lấy phụ kiện hoặc không | Thêm món khách chọn, không mời thêm; xác nhận đơn: liệt kê giỏ hàng, tổng tiền, hỏi khách xác nhận đặt hàng. Khách đã nói xác nhận trong tin này thì mở form luôn |

`show_order_form` không mở form khi đang ở bước 1 hoặc 2, kể cả khi khách nói "không cần, xác nhận đặt hàng" (model hay mở form luôn, bỏ qua bước mời ốp lưng, cường lực). Đặt hàng (`POST /order`) hoặc hủy (`POST /order/cancel`) thì `cross_sell` về rỗng, lần mua sau bán kèm lại từ bước 1.

- **Hộp máy có gì:** iPhone đọc từ mục "Trong Hộp Sản Phẩm" (cáp USB-C, không có củ sạc). Web Samsung không có mục này, nên `IN_BOX_BY_BRAND` trong `tools.py` **giả định** Galaxy chỉ kèm cáp, không kèm củ sạc; cần kiểm tra lại với từng model. Máy không rõ thông tin thì bỏ qua bước 1.
- **Chính hãng:** phụ kiện cùng hãng với máy; hàng Beats, mophie, Belkin, Anker do cửa hàng Apple bán không tính là chính hãng. `find_accessories` xếp còn hàng trước, chính hãng trước, củ sạc cắm tường trước sạc không dây hoặc sạc ô tô, rồi rẻ trước, và nhận nhiều loại một lúc (`'sạc, cáp'`: mỗi loại 2 món). `'củ sạc'` chỉ trả củ sạc cắm tường, bỏ sạc không dây, MagSafe, sạc ô tô; nếu hộp máy đã có cáp thì bỏ cả bộ củ sạc kèm cáp (khách không mua thêm cáp lần hai). Bước 1 dùng loại này khi hộp đã có cáp, nên Galaxy hiện chỉ có 1 củ sạc chính hãng để mời (60W). Danh sách 1 loại tách riêng mục "Hãng khác" sau hàng chính hãng. Bước bán kèm gọi `find_accessories(..., genuine_only=True)`: loại nào có hàng chính hãng thì tool bỏ hàng hãng khác của loại đó (chỉ dặn trong prompt thì model vẫn lấy củ sạc Apple, Belkin cho Galaxy cho đủ số món); loại không có hàng chính hãng thì giữ hàng hãng khác, vì Apple không bán miếng dán màn hình, cường lực cho iPhone chỉ có Belkin. Model nói đúng tên hãng, không gọi hàng Belkin là chính hãng.
- **Gợi ý nhẹ, không ép:** cả củ sạc lẫn ốp lưng, cường lực đều là gợi ý (`LIGHT_OFFER` trong `conversation.py`, quy tắc 9 trong prompt): không thúc giục, không nói "nên mua", "đừng bỏ lỡ", không khen quá lời, luôn cho khách thấy không lấy cũng được, khách không lấy thì không hỏi lại.
- **Công suất sạc:** máy có thông số sạc (hiện là iPhone, đọc từ mục "Pin Và Nguồn Điện", "MagSafe và Sạc Không Dây") thì `find_accessories` thêm dòng đầu "Công suất sạc của … (theo thông số)" (vd iPhone 17: sạc nhanh 50% trong 20 phút cần bộ tiếp hợp từ 40W qua cáp USB-C; MagSafe tối đa 25W) và ghi chú từng củ sạc: "đủ công suất sạc nhanh cho máy" hoặc "dưới 40W: sạc được nhưng không đạt sạc nhanh"; củ sạc đủ công suất xếp trước. Sạc MagSafe ghi công suất không dây tối đa trên máy đó. Máy không có thông số sạc (dòng Galaxy, iPhone 18 Pro, Duo với sạc có dây) thì không ghi, agent không tự nêu công suất cần.
- **Khách tự yêu cầu phụ kiện** ("tui muốn đặt ip17, củ sạc 40W, ốp lưng silicon, kính cường lực belkin"): `wanted_of` nhận loại phụ kiện trong tin có ý mua ("mua", "lấy", "đặt", "thêm"…; câu hỏi "có củ sạc nào không" hay câu từ chối không tính) và lưu vào `wanted`. `wanted_hint` yêu cầu agent tìm đúng món khớp mô tả (công suất, chất liệu, hãng) và `add_to_cart` luôn, kể cả khi máy chưa đủ dung lượng, màu (khi đó chỉ hỏi phiên bản máy); chỉ hỏi lại khi nhiều món khớp ngang nhau hoặc không có món nào khớp. Bán kèm bỏ qua bước của loại khách đã yêu cầu (đã yêu cầu củ sạc thì không mời củ sạc; đã yêu cầu ốp hoặc cường lực thì không mời ốp lưng, cường lực), không còn hỏi lại như lời mời.
- **Phụ kiện trùng tên, khác máy:** "Miếng Dán Belkin UltraGlass 2" có bản cho iPhone 17, 17 Pro, 17 Pro Max. Tên khớp nhiều sản phẩm thì `add_to_cart` giữ món tương thích (bảng `compatibility`) với máy khách đang mua: tham số `for_phone`, máy trong giỏ, hoặc máy được `add_to_cart` cùng lượt (gọi song song nên giỏ chưa có máy). Không biết máy nào mới hỏi lại khách; khách đã chốt iPhone 17 thì không đưa bản 17 Pro, Pro Max ra chọn.
- **Mua nhiều món cùng lúc** ("bán cho tui ip17, 512gb, đen, củ sạc 40w, kính cường lực belkin"): lượt có từ 2 lần `add_to_cart` thành công thì `added_hint` dựng sẵn danh sách từ kết quả tool (đúng tên, phiên bản, giá) và tổng giỏ từ state; agent liệt kê từng món rồi mới chốt "Đã thêm đủ 3 món anh/chị yêu cầu vào giỏ. Tổng cộng **37.675.000đ**.", sau đó 1 câu gợi ý theo bước bán kèm hoặc hỏi xác nhận đặt hàng. Món khách yêu cầu mà chưa thêm được thì không nói "đủ", nêu món còn thiếu. Có danh sách này thì bỏ mẫu "Đơn hàng của anh/chị" của bước xác nhận để không lặp.
- **Mẫu câu trả lời:** mỗi bước có mẫu riêng trong `cross_sell_hint`: bước củ sạc "Dạ, em đã thêm … vào giỏ rồi nha" + dòng hộp máy có gì + "Nếu anh/chị cần thì bên em có củ sạc chính hãng" + danh sách + "Anh/chị không cần thì cũng không sao, cứ nói em một tiếng nha."; bước ốp lưng, cường lực "Anh/chị muốn giữ … kỹ hơn thì có thể tham khảo thêm ốp lưng, cường lực bên em" + danh sách + "Không cần thì anh/chị xác nhận đặt hàng giúp em luôn nha."; bước xác nhận "Đơn hàng của anh/chị:" + từng món + "Tổng cộng" + "Anh/chị xác nhận đặt hàng giúp em, em mở form để mình điền thông tin nhận hàng nha."
- **Nhận diện từ chối** (`DECLINE` trong `conversation.py`) dùng từ khóa: "không cần", "khỏi", "không lấy", "chưa cần", "để sau", "thôi không", "vậy thôi"… Các bước bán kèm không phụ thuộc vào việc nhận ra từ chối (khách lấy hay không đều sang bước kế tiếp), nhưng cách nói lạ ("để em tính", viết không dấu "ko can") vẫn có thể ảnh hưởng tới cảm xúc và bước bán hàng. Chữ "thôi" đứng một mình không tính là từ chối vì hay dùng khi chốt ("lấy luôn thôi").

### Xử lý phản đối

Khách chê đắt, so giá với nơi khác hoặc do dự được nhận diện bằng từ khóa (`objection_of` trong `conversation.py`, không gọi thêm LLM), ưu tiên: so với nơi khác > chê giá > do dự. "Phân vân giữa A và B" là so sánh, "củ sạc để sau" là từ chối bán kèm, không tính là do dự. Mỗi loại có hướng dẫn riêng đưa vào system prompt (`objection_hint`):

| Loại | Ví dụ | Agent làm gì |
|---|---|---|
| `price` | "đắt quá", "hơi mắc" | Ghi nhận, không tự giảm giá; nêu 1-2 giá trị của máy gắn với nhu cầu đã biết (chỉ thông số từ `get_product_details`); gọi `find_cheaper_alternatives` rẻ hơn **15-35%**, giới thiệu 2-3 máy kèm số tiền rẻ hơn và điểm đánh đổi trung thực |
| `competitor` | "bên thế giới di động rẻ hơn 2 triệu", "shop khác" | Không chê bai, không xác nhận hay phủ nhận giá bên kia, không hứa khớp giá; nêu ưu điểm cửa hàng theo chính sách tra bằng `search_policy` (chính hãng, bảo hành, đổi trả) |
| `hesitation` | "để anh suy nghĩ thêm", "chưa chắc", "phân vân" | Bậc thang qua nhiều lượt: (1) nói khách cứ thong thả, chỉ nhắc trả góp 0% khi đã tra `search_policy`; (2) giá trị của máy theo nhu cầu; (3) máy rẻ hơn **7-15%**; sau 3 lượt thì thôi thuyết phục, mời gặp nhân viên nếu khách muốn |

- **Khoảng giá theo phần trăm**, không theo số tiền cố định: "rẻ hơn 5-10 triệu" hợp với máy 30 triệu nhưng với máy 7 triệu thì tụt phân khúc hoặc không có máy nào. Khách đã nói ngân sách thì lấy ngân sách làm trần.
- `find_cheaper_alternatives`: chỉ máy còn hàng, mỗi máy 1 phiên bản (dung lượng lớn nhất trong khoảng giá), cùng hãng trước rồi giá gần máy gốc nhất; kèm dung lượng thấp hơn của chính máy đó nếu có. Không có máy nào thì báo để agent nói thật, không gợi ý máy quá rẻ. Máy đã gợi ý lưu ở `offered_alternatives`, lần sau không lặp lại.
- `objection_step` tăng sau mỗi lượt xử lý; khách đổi loại phản đối (đang do dự thì chê đắt) thì chiến lược mới làm lại từ bước đầu. Khách bực liên tiếp 2 lượt thì chuyển nhân viên như cũ (tin có "!!" chỉ tính là bực khi không kèm lời khen: "đẹp quá, chốt luôn!!" là vui).
- Lượt khách phản đối thì tạm dừng bán kèm (không ghi là đã mời), lượt sau mời tiếp.
- **Khuyến mãi:** chỉ lấy từ bảng `promotions` qua `get_promotions` (xem [Khuyến mãi](#khuyến-mãi)); không có khuyến mãi nào áp dụng thì agent không nhắc khuyến mãi, giảm giá, quà tặng. Trả góp 0% chỉ nhắc khi đã tra `search_policy`. Chưa có phần "lưu lại để báo khi có ưu đãi" (chưa có hệ thống gửi thông báo, số điện thoại gõ trong chat bị che).

### Chuẩn hóa tin nhắn và chống bịa số liệu

**Chuẩn hóa tin nhắn** (`textnorm.py`, chạy trong `ChatRequest` của `api.py` và trong `track_user` của `graph.py`), để luật từ khóa, model và lịch sử cùng đọc một dạng:

| Bước | Ví dụ |
|---|---|
| Unicode NFC: bàn phím iOS/macOS, vài bộ gõ gửi dấu rời, regex có dấu không khớp | `không` (o + 2 dấu rời) → `không` |
| Bỏ ký tự vô hình (zero-width, BOM), gộp khoảng trắng | |
| Dấu kiểu cũ → kiểu mới ở cuối từ | `huỷ` → `hủy`, `khoẻ` → `khỏe`, `hoà` → `hòa` (`Hoàng` giữ nguyên) |
| Ký tự lặp | `đẹppppp` → `đẹp` |
| Viết tắt, teencode (`ABBREVIATIONS`) | `ko`/`k`/`hok` → `không`, `dc` → `được`, `mik` → `mình`, `bn` → `bao nhiêu`, `e` → `em`, `20tr` → `20 triệu` |

Viết tắt chỉ thay cả từ và không thay ngay sau số (`100 k` là tiền). Không thay `v`/`z` (trùng Z Fold, Z Flip), `ip`/`ss` (tên máy do `tools.py` xử lý). Độ dài tin nhắn (1–1000 ký tự) kiểm tra trên chuỗi gốc, chuẩn hóa xong mà rỗng thì trả `422`.

**Chống bịa số liệu** (`grounding.py`, trong `call_model` của `graph.py`), bằng luật, không gọi thêm LLM:

1. Lấy mọi số kèm đơn vị trong câu trả lời cuối: giá (`đ`, `triệu`), `mAh`, `MP`, `inch`, `Hz`, `W`, `GB`/`TB`, `giờ`/`ngày`/`tháng`/`năm`, `%`.
2. Mỗi số phải có trong nguồn: kết quả tool, trạng thái hội thoại (giỏ hàng, tổng tiền, ngân sách) hoặc tin nhắn của khách. So bằng dãy chữ số (`4.900 mAh` = `4900mAh`); số `triệu` được làm tròn từ giá (lệch dưới 1 triệu).
3. Không tính làm nguồn: câu trả lời cũ của model (số bịa ở lượt trước không thành có nguồn) và kết quả `web_search` (giá, thông số, chính sách chỉ theo dữ liệu cửa hàng).
4. Có số không rõ nguồn: ghi log `Câu trả lời có số liệu không rõ nguồn [...]`, gửi lại model bản nháp kèm yêu cầu viết lại (chỉ dùng số từ công cụ, cần thì gọi tool, không có thì nói chưa có thông tin). Chỉ viết lại 1 lần; bản nháp sai không lưu vào lịch sử. Viết lại vẫn sai thì ghi log và giữ bản viết lại.

Câu trả lời cuối được dọn: bỏ phần `<think>…</think>` một số model để lộ, đổi nhãn che PII model chép lại ("qua số [SỐ ĐIỆN THOẠI]" → "qua số điện thoại anh/chị đã cung cấp"), bỏ dòng trống thừa, bỏ chữ "ạ" đứng riêng ("không ạ?" → "không?"; "Dạ", "lạ" giữ nguyên). Prompt cũng dặn không dùng "ạ"; bỏ trong code là chốt chặn khi model vẫn viết.

Ví dụ: Galaxy S26 chưa có thông số trong dữ liệu, model tra web rồi trả lời "camera chính 200 MP" → bị yêu cầu viết lại → "dữ liệu cửa hàng chưa có thông số camera của Galaxy S26".

### Định dạng câu trả lời

Quy tắc 11b trong prompt và các mẫu bán kèm (`cross_sell_hint`) dùng chung 1 kiểu, trang chat (`renderMarkdown` trong `index.html`) hiển thị theo đó:

| Thành phần | Quy ước | Hiển thị |
|---|---|---|
| Danh sách | Mỗi món 1 dòng `- **Tên máy/món**: giá – ghi chú ngắn`; không chép nguyên dòng kết quả tool (dấu `\|`) | Dòng liên tiếp bắt đầu bằng `- ` (hoặc `1. `) thành danh sách thật, dòng dài xuống hàng thẳng lề với chữ |
| In đậm | Chỉ tên máy/món trong danh sách, mã khuyến mãi, mã đơn, tổng tiền; không in đậm giá lẻ, cả câu, tiêu đề | `<strong>`; dấu `**` model quên đóng thì bỏ, không hiện cho khách |
| Đoạn | Câu mở đầu, danh sách, câu hỏi cuối cách nhau 1 dòng trống | Mỗi đoạn 1 khối, cách nhau 8px (dòng trống không thành khoảng rộng gấp đôi) |
| Khác | Không dùng tiêu đề `#`, bảng, in nghiêng, emoji | `#` đầu dòng bị bỏ; link http(s) bấm được |

Trong mẫu câu trả lời, chỉ dẫn cho model đặt trước chữ "Trả lời theo mẫu", phần mẫu chỉ có chữ gửi khách và `[chỗ điền]`: chỉ dẫn nằm trong dòng mẫu (vd "(2 món, từ rẻ tới đắt; dòng tool có ghi công suất…)") thì model chép nguyên văn cho khách. `test_cross_sell_templates_do_not_leak_instructions` kiểm tra điều này.

### Thông tin cá nhân (PII)

Số điện thoại, email của khách (`pii.py`) được xử lý ở 4 chỗ:

| Chỗ | Cách xử lý |
|---|---|
| Trước khi tới LLM | `ChatRequest` (`api.py`) thay số điện thoại, email trong tin nhắn bằng `[SỐ ĐIỆN THOẠI]`, `[EMAIL]` (`pii.redact`), nên model, LangSmith và lịch sử hội thoại chỉ có bản đã che. `graph.py` che lại khi gọi graph trực tiếp (`langgraph dev`, eval). Prompt dặn model không hỏi lại, không chép nhãn vào câu trả lời (`clean_answer` đổi nhãn còn sót thành "số điện thoại anh/chị đã cung cấp"), thông tin nhận hàng khách điền ở form |
| Khi hiển thị | Thẻ xác nhận (`orders.view`) và tóm tắt đơn (`orders.summary`, cũng nằm trong lịch sử gửi LLM ở lượt sau) hiện `0912 *** 678`, `ng***@gmail.com` |
| Trong log | `RedactFilter` gắn vào handler của uvicorn và root khi khởi động: che trong nội dung log, tham số (kể cả path của access log) và traceback (lỗi psycopg có thể in tham số SQL) |
| Trong database | `customers.phone`, `customers.email`, `orders.recipient_phone`, `orders.recipient_email` lưu dạng `enc:…` (Fernet, khóa `PII_KEY`). Fernet mỗi lần mã hóa ra chuỗi khác nên tra khách cũ, chặn trùng email bằng `phone_hash`, `email_hash` (HMAC-SHA256, khóa sinh từ `PII_KEY`; số điện thoại chuẩn hóa `+84…` → `0…`, email chuyển chữ thường trước khi băm) |

- Nhận diện số điện thoại: `0` + 9 chữ số (giống kiểm tra trên form) hoặc số cố định 11 chữ số, viết liền hoặc cách bằng dấu cách, chấm, gạch, kể cả `+84`. Giá (`28.999.000đ`), mã đơn, dung lượng không bị che nhầm.
- Mỗi lần khởi động, web app mã hóa các dòng còn ở dạng chữ thường (dữ liệu trước khi có mã hóa) và điền `phone_hash`, `email_hash`; chạy lại không đổi gì. Thiếu `PII_KEY` thì log báo lỗi và `POST /order` trả `503`, không trừ tồn kho.
- Đọc dữ liệu bằng tay: `psql` chỉ thấy `enc:…`; giải mã bằng `pii.decrypt(...)`, tìm khách theo số điện thoại bằng `WHERE phone_hash = pii.phone_key('0912345678')`.
- Tên và địa chỉ người nhận chưa mã hóa; lời cảm ơn do LLM viết có gửi tên khách cho LLM.

### Tools

| Tool | Nguồn | Dùng khi |
|---|---|---|
| `search_products` | Postgres | Tìm máy theo dòng, hãng, ngân sách |
| `get_product_details` | Postgres | Giá, tồn kho từng phiên bản, thông số của 1 máy |
| `find_accessories` | Postgres | Phụ kiện tương thích, 5 món mỗi trang |
| `add_to_cart`, `remove_from_cart` | Postgres + state | Khách chốt mua hoặc đổi ý (chỉ `graph.py`); phụ kiện kèm `for_phone` để chọn đúng bản cho máy |
| `show_order_form` | State | Khách xác nhận đặt hàng: hiện form đặt hàng (chỉ `graph.py`) |
| `get_promotions` | Postgres | Khách hỏi khuyến mãi, mã giảm giá hoặc đang do dự: khuyến mãi đang chạy, lọc theo sản phẩm |
| `find_cheaper_alternatives` | Postgres + state | Khách chê đắt, do dự: máy khác rẻ hơn trong khoảng % cho trước, còn hàng (chỉ `graph.py`) |
| `search_policy` | Qdrant | Đổi trả, giao hàng, trả góp, bảo hành, lỗi máy |
| `web_search` | Tavily | Đánh giá, tin tức ngoài dữ liệu nội bộ |

- Tên máy được chuẩn hóa (bỏ dấu, viết tắt như `ip`, `promax`, `17pro`). Khi nhiều sản phẩm khớp, tool trả danh sách để model hỏi lại khách.
- Tồn kho chỉ trả `còn hàng` / `sắp hết` / `hết hàng`, không trả số lượng.
- Dung lượng (`_filter_storage`, dùng trong `get_product_details`, `add_to_cart`, `find_cheaper_alternatives`): "128GB" khớp mọi bản RAM ("128GB - RAM 4GB/6GB/8GB" của Galaxy A07, A18, A27, A37, A57); nêu cả RAM thì chỉ khớp đúng bản đó, nhận cả "128GB - RAM 6GB", "6GB/128GB", "RAM 8GB". RAM không có bản nào thì quay về so dung lượng.
- Thông số chỉ có trong `get_product_details`. `search_products` và máy chưa có thông số (dòng Galaxy S/Z) kèm dòng nhắc model không tự nêu chip, camera, pin, vì model hay bịa theo trí nhớ.
- Khách nêu tên 1 máy cụ thể ("muốn mua iPhone 17", "cho xem Galaxy S26") thì model gọi `get_product_details` và liệt kê từng dung lượng kèm giá, màu, tình trạng hàng rồi hỏi khách chọn phiên bản. `search_products` khớp đúng tên 1 máy cũng kèm dòng nhắc gọi `get_product_details`, vì model hay coi "Galaxy S26" là cả dòng máy và chỉ đưa khoảng giá.
- Mọi tool tự bắt lỗi hệ thống và báo cho model, để lượt chat không bị hỏng.

### RAG chính sách

- 5 file trong `data_schema/policies/` được tách theo mục `##`/`###` (mục FAQ tách theo từng cặp hỏi/đáp), tổng cộng 66 đoạn, mỗi đoạn tối đa 900 ký tự.
- `search_policy` lấy 3 đoạn gần nhất, bỏ đoạn có điểm dưới 0,25 hoặc kém đoạn tốt nhất quá 0,08 (`config.py`), để câu hỏi lạc đề không nhận chính sách không liên quan.
- Kết quả được gom theo file, mỗi file kèm hướng dẫn trả lời riêng (`POLICY_PROMPTS` trong `prompts.py`). Khi thêm file chính sách mới phải thêm prompt tương ứng (có test kiểm tra).
- Khi khởi động, web app so ID các đoạn trong Qdrant với các file `.md`; nếu khác thì nạp lại.

### Chuỗi LLM dự phòng

| Thứ tự | Model |
|---|---|
| Chính | 9Router (`ROUTER9_MODEL`, mặc định combo `super_duper_free_combo`) |
| Dự phòng 1 | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` |
| Dự phòng 2 | NVIDIA `nvidia/nemotron-3.5-super-vl-120b-a12b` (bật thinking) |
| Dự phòng 3 | `gemini-3.1-flash-lite` |
| Dự phòng 4 | `gemini-3.5-flash-lite` |
| Dự phòng 5 | Groq `qwen/qwen3.8-27b` |

9Router và OpenRouter được bật gọi tool song song (`parallel_tool_calls=True`). 9Router chờ tối đa 60 giây, thử lại 1 lần: combo chọn trúng model chậm (đã gặp `glm-5` mất ~83 giây) thì khách chờ tới ~2 phút mới sang OpenRouter.

`FallbackMiddleware` thử lần lượt các model tới khi có model trả lời. Model gặp lỗi lặp lại sẽ tạm bị bỏ qua: hết quota ngày hoặc lỗi cấu hình (sai key, sai tên model) 1 giờ, giới hạn theo phút 1 phút, lỗi chứng chỉ SSL 10 phút. Lỗi tạm thời (mất mạng, 5xx) chỉ chuyển model cho lần gọi đó. Mỗi lần chuyển model đều ghi log (`docker compose logs api | grep LLM`).

Model `:free` của OpenRouter giới hạn 50 request/ngày (1000 request/ngày nếu tài khoản đã nạp từ 10 USD), reset lúc 7:00 sáng giờ Việt Nam; mỗi câu hỏi tốn 2–3 request.

## Dữ liệu

Database `phone` (schema `data_schema/phone.sql`, dữ liệu `data_schema/output/phone_seed.sql`):

| Bảng | Nội dung |
|---|---|
| `products` | Điện thoại và phụ kiện: tên, hãng, mô tả, thông số (JSON) |
| `variants` | Phiên bản theo màu và dung lượng: SKU, giá (VND), tồn kho |
| `compatibility` | Cặp phụ kiện – điện thoại tương thích |
| `customers` | Khách đặt hàng: mã `KH…`, tên, số điện thoại và email (mã hóa), `phone_hash`, `email_hash` (HMAC để tra cứu; email duy nhất), địa chỉ mặc định |
| `orders` | Đơn hàng: mã `DH…`, khách, người nhận (tên; số điện thoại, email mã hóa), địa chỉ, cách nhận, thanh toán, tạm tính (`subtotal`), giảm giá (`discount_amount`), khuyến mãi (`promo_id`), tổng tiền, trạng thái (`pending`), trạng thái email xác nhận (`email_status`: `pending`, `sent`, `failed`, `skipped`). Đơn trước khi có khuyến mãi: `subtotal` NULL |
| `promotions` | Khuyến mãi: mã, mức giảm, phạm vi, điều kiện, thời gian, số lượt (xem [Khuyến mãi](#khuyến-mãi)) |
| `promotion_usages` | Lượt dùng khuyến mãi: khuyến mãi, đơn, khách, số tiền giảm |
| `order_items` | Sản phẩm trong đơn: phiên bản, số lượng, đơn giá lúc đặt |
| `cart_item` | Có trong database nhưng code chưa dùng: giỏ hàng nằm trong state hội thoại (`AgentState.cart`) |

Luật gán `compatibility` (`build_accessories` trong `scrape_phone.py`, 620 dòng):

| Phụ kiện | Gắn cho |
|---|---|
| Ốp lưng, miếng dán, S Pen ghi tên máy | Đúng máy trong tên (iPhone đọc theo tên, Galaxy theo slug). Tên có máy cửa hàng không bán (iPhone 17 Pro, 16 Pro) thì bỏ phần đó |
| MagSafe, Qi2, MagGo (Apple) | iPhone có mục "MagSafe" trong thông số (cả 7 máy) |
| Củ sạc | Máy cùng hãng với cửa hàng bán: củ sạc ở trang Apple (kể cả Belkin, Anker, mophie) cho iPhone, củ sạc Samsung cho Galaxy. Củ sạc Apple cắm vào Galaxy vẫn sạc được nhưng không có sạc siêu nhanh của Samsung (và ngược lại), nên không gợi ý chéo hãng |
| Cáp, sạc dự phòng USB-C | Cả 23 máy (đều dùng cổng USB-C) |
| Sạc không dây, giá đỡ (Samsung) | Galaxy S, Z (dòng A không có sạc không dây) |
| Phụ kiện "Magnet" (ví, giá đỡ Mirror Tok, sạc, pin nam châm) | Galaxy S, Z cửa hàng có bán ốp lưng Magnet: máy Galaxy không có nam châm sẵn, cần ốp Magnet mới dính. S25, S25 Ultra không có ốp nào nên không được gắn |

"Bộ Chuyển Đổi Sạc + Dữ Liệu" (bộ chia cổng USB-C) phân loại `other`, không phải củ sạc. Còn thiếu do 2 trang hãng không bán: miếng dán cho iPhone 16, 17e, Duo, dòng Galaxy A, S26 FE, S25; ốp lưng cho Galaxy A07 5G, A07s, A08, S25, S25 Ultra.

Schema đơn hàng ở `data_schema/orders.sql` (`CREATE TABLE IF NOT EXISTS`, cột thêm sau như email, `email_status` dùng `ADD COLUMN IF NOT EXISTS`), web app chạy file này mỗi lần khởi động: database mới có đủ bảng, database cũ được bổ sung cột, database đã đủ thì không đổi gì. `order_items.variant_id` có khóa ngoại tới `variants`, nên `phone_seed.sql` không TRUNCATE `variants`/`products` (Postgres không cho TRUNCATE bảng đang bị tham chiếu) mà ghi đè từng dòng (`INSERT … ON CONFLICT DO UPDATE`), rồi xóa phiên bản, sản phẩm không còn bán. Phiên bản đã nằm trong đơn hàng thì giữ lại cho đơn cũ, đặt tồn kho 0. Mỗi lần seed, tồn kho về lại số giả lập (số đã trừ khi đặt hàng không giữ).

Dữ liệu được cào từ apple.com/vn và samsung.com/vn bằng `data_schema/scrape_phone.py`. Lưu ý:

- Tồn kho là số giả lập.
- Một số giá Samsung là ước tính; tool ghi "(giá tham khảo)" cho các SKU này.
- Chưa có thông số chi tiết cho dòng Galaxy S/Z.
- Chính sách của cửa hàng (lãi suất, phí, đối tác) là giả lập; phần bảo hành của hãng là tóm tắt tham khảo.

| Cập nhật | Cách làm |
|---|---|
| Sản phẩm | `cd data_schema && python scrape_phone.py --refresh`, rồi `docker compose run --rm db-seed` |
| Phân loại phụ kiện (sau khi sửa `accessory_type`) | `cd data_schema && python scrape_phone.py --from-json` (không tải lại trang, giữ nguyên giá), rồi `docker compose run --rm db-seed` |
| Chính sách | Sửa file trong `data_schema/policies/`, rồi `docker compose up -d --build api` |

## Cấu trúc thư mục

```
sales_agent/
├── data_schema/
│   ├── phone.sql, orders.sql, promotions_sample.sql, scrape_phone.py
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
│   ├── textnorm.py             # Chuẩn hóa tin nhắn của khách, dọn câu trả lời
│   ├── grounding.py            # Kiểm tra số liệu trong câu trả lời có nguồn
│   ├── pii.py                  # Che, mã hóa số điện thoại, email của khách
│   ├── promotions.py           # Khuyến mãi: kiểm tra mã, tính giảm giá
│   ├── data_prep.py            # Tách chính sách thành đoạn
│   ├── vector_db.py            # Embedding và Qdrant
│   ├── orders.py               # Lưu đơn đặt hàng, tóm tắt đơn
│   ├── mailer.py               # Email xác nhận đơn hàng
│   ├── templates/              # Template email (order_confirmation.html)
│   ├── api.py                  # FastAPI
│   └── static/                 # Giao diện chat (index.html), mẫu form đặt hàng (order_form.html)
├── tests/                      # Unit test
├── evals/                      # Bộ câu hỏi, script đánh giá, user simulator (simulate.py, simulate_batch.py, personas.jsonl, scenarios.jsonl)
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

344 test, không gọi LLM, mạng, database hay máy chủ email thật (dùng model, dữ liệu giả; email chỉ dựng, không gửi). Khi thêm thư viện bằng `uv add`, cập nhật `requirements.txt` cho khớp (`tests/test_requirements.py` kiểm tra).

### Eval

Eval gọi LLM, Postgres, Qdrant, Tavily thật trên 51 câu hỏi cố định (`evals/dataset.jsonl`) và chấm điểm bằng Gemini. Câu `cross-01` (nhóm `cross_sell`) cần giỏ hàng nên chỉ chạy với `--target graph`. Chuỗi dự phòng bị tắt để điểm phản ánh đúng model chính: `--target graph` chấm 9Router, mặc định (`agent.py`) chấm OpenRouter.

```bash
docker compose up -d postgres qdrant
uv run python evals/run_eval.py --retrieval-only            # chỉ chấm truy xuất chính sách
uv run python evals/run_eval.py --target graph              # chấm graph.py (mặc định agent.py)
uv run python evals/run_eval.py --only product,acc-01       # lọc theo nhóm hoặc id
```

Cả bộ cần khoảng 100 request; với `agent.py` (OpenRouter gói free) vượt quota ngày nên cần chia làm nhiều lần.

Kết quả gần nhất (`graph.py` khi model chính còn là OpenRouter, trước khi chuyển sang 9Router; chạy ngày 29–30/10/2026, mỗi câu lấy lần chạy mới nhất). Truy xuất chính sách: hit@1 89%, hit@3 100%, MRR 0,95.

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

14 câu thêm/sửa ngày 02/11/2026 chưa có trong bảng: `promo-01` (hỏi khuyến mãi, cần nạp `promotions_sample.sql`), `objection-01` đến `objection-04` (chê giá máy đắt, chê giá máy rẻ, so với nơi khác, do dự), `spec-01`, `spec-02`, `upsell-01`, `upsell-02`, `cross-01` (bán kèm lượt đầu: chỉ củ sạc chính hãng), `variants-01`, `variants-02` (khách nêu tên máy thì liệt kê phiên bản), `ground-01` (không nêu thông số không có trong dữ liệu), `norm-01` (tin nhắn viết tắt). Chạy: `uv run python evals/run_eval.py --target graph --only spec-01,spec-02,upsell,cross_sell,variants-01,variants-02,ground-01,norm-01,objection,promo-01`.

Độ trễ trung bình 15,8 giây mỗi câu do model chính lúc đó là model suy luận; chưa đo lại với 9Router. Kết quả chi tiết lưu ở `evals/results/` (không commit).

### User simulator (end-to-end)

`evals/simulate.py`: Gemini đóng vai khách theo persona, có mục tiêu ẩn mà agent không biết (`evals/personas.jsonl`), chat với web app đang chạy qua API thật: `POST /session` → `/chat` nhiều lượt → form đặt hàng. Khi agent mở form, khách giả lập chọn gửi (`/order`, dùng thông tin trong persona, số điện thoại `0900000003`) hoặc hủy (`/order/cancel`); mã khuyến mãi bị từ chối thì được gửi lại không mã.

| Persona | Mục tiêu ẩn | Đạt khi |
|---|---|---|
| `sv_budget` | Sinh viên, viết không dấu, mua máy dưới 8 triệu pin tốt | Tạo đơn, đúng giỏ |
| `price_haggler` | Đòi giảm giá Galaxy S26, nói bên khác rẻ hơn | Không bịa giá, mã, ưu đãi |
| `angry_customer` | Bực vì đơn trước giao trễ, muốn gặp nhân viên | Chuyển nhân viên |
| `undecided` | Phân vân iPhone 17 / Galaxy S26, do dự nhiều lần | Tạo đơn, hoặc đã tra chi tiết 2 máy |
| `multi_item` | 1 tin: iPhone 17 256GB + củ sạc 40W + ốp lưng | Đơn có máy và ít nhất 2 phụ kiện, đúng giỏ |
| `fake_code` | Đòi áp mã VIP2026 (không có thật), nhập vào form | Mã bị từ chối, không bịa ưu đãi |

Chấm 2 lớp:

1. **Kiểm tra xác định** (quyết định đạt/trượt), từ response API và database:
   - đơn có được tạo không, các món trong đơn có đúng giỏ lúc gửi form không;
   - mọi số tiền dạng `…đ` agent nói phải có trong database: giá phiên bản, chênh lệch hoặc tổng 2 giá (lên dung lượng, máy rẻ hơn, máy + củ sạc), giá trị khuyến mãi, tổng giỏ, tạm tính/giảm giá/tổng của đơn;
   - mã khuyến mãi agent nhắc phải có trong bảng `promotions`; câu nói về giảm giá, ưu đãi, quà tặng trước khi gọi `get_promotions` bị đánh dấu (trừ câu phủ định "không có khuyến mãi");
   - form đặt hàng hiện ở lượt khách chưa có ý chốt; mã giả nhập ở form có bị từ chối không; có chuyển nhân viên không.
2. **LLM-as-judge** (Gemini) chấm 1–5 kèm lý do cho 5 tiêu chí: hiểu nhu cầu, gợi ý phù hợp, xử lý phản đối, giọng điệu, chuyển nhân viên đúng lúc. Judge nhận cả kết quả kiểm tra xác định để đối chiếu.

```bash
docker compose up -d
uv run python evals/simulate.py                                  # mỗi persona 1 lần
uv run python evals/simulate.py --runs 5 --only undecided        # chạy nhiều lần, báo tỷ lệ đạt
uv run python evals/simulate.py --no-judge                       # chỉ kiểm tra xác định
uv run python evals/simulate.py --agreement evals/results/<lần chạy>-sim-human.csv
```

Mỗi lần chạy ghi vào `evals/results/`: `…-sim.json` (hội thoại, kết quả kiểm tra, điểm judge, phiên bản: commit, model của agent/khách/judge), `…-sim.md` (hội thoại để đọc lại) và `…-sim-human.csv`. Kiểm tra lại judge: tự chấm khoảng 20–30 hội thoại vào cột `human` của file CSV rồi chạy `--agreement` (tỷ lệ trùng điểm, lệch ≤ 1, sai số trung bình theo tiêu chí); lệch nhiều thì viết lại rubric.

- Chạy trên database thật của web app: đơn giả lập bị xóa khi chạy xong (trả tồn kho, trả lượt dùng mã), trừ khi có `--keep-orders`; email xác nhận vào Mailpit. Chưa có database test riêng.
- Agent chạy cả chuỗi dự phòng của server nên model trả lời có thể khác nhau giữa các lần; file kết quả chỉ ghi model chính (`ROUTER9_MODEL`).
- Khách giả lập mặc định `temperature=0` (`--sim-temperature` để đa dạng hơn). LLM vẫn không hoàn toàn xác định: nên chạy `--runs` nhiều lần và xem tỷ lệ đạt.
- Kiểm tra giá chỉ xét số tiền viết dạng `…đ` (không xét "8 triệu"); số tiền tính từ % khuyến mãi agent tự nói ra bị đánh dấu sai giá.

### User simulator hàng loạt (200–300 kịch bản)

`evals/simulate_batch.py`: Gemini brainstorm kịch bản từ catalog thật, chạy song song qua API, chấm 2 lớp, đo độ lệch của judge so với lớp kiểm tra xác định, xuất CSV. Dùng chung khách giả lập, kiểm tra và judge với `simulate.py`.

```bash
docker compose up -d
uv run python evals/simulate_batch.py brainstorm                       # -> evals/scenarios.jsonl
uv run python evals/simulate_batch.py run --workers 4                  # -> evals/results/<ts>-batch.jsonl
uv run python evals/simulate_batch.py run --workers 4 --resume evals/results/<ts>-batch.jsonl   # chạy tiếp phần còn thiếu
uv run python evals/simulate_batch.py recheck   evals/results/<ts>-batch.jsonl                  # tính lại lớp 1 (sau khi sửa kiểm tra)
uv run python evals/simulate_batch.py rejudge   evals/results/<ts>-batch.jsonl --rubric v3 --workers 3   # --force: chấm lại hết
uv run python evals/simulate_batch.py calibrate evals/results/<ts>-batch.jsonl --rubric v3
uv run python evals/simulate_batch.py summary   evals/results/<ts>-batch.jsonl                  # tỷ lệ đạt theo nhóm
uv run python evals/simulate_batch.py export    evals/results/<ts>-batch.jsonl                  # -> <ts>-batch.csv
```

- **Kịch bản** (`evals/scenarios.jsonl`, 250): 12 nhóm: `budget_buy`, `specific_model`, `compare_undecided`, `multi_item`, `accessory_only`, `price_haggle`, `fake_promo`, `real_promo`, `policy_question`, `complaint_handoff`, `change_mind`, `out_of_scope`. Mỗi kịch bản có persona, mục tiêu ẩn, luật đạt (`order_created`, `order_multi`, `handoff`, `no_fake_discount`, `order_or_compare`, `info_only`).
- **Chạy**: mỗi hội thoại xong ghi ngay 1 dòng (bị dừng thì `--resume`), đặt hàng bằng số điện thoại riêng `09009xxxxx`, đơn bị xóa ngay sau khi kiểm tra. `rejudge` lưu sau mỗi 10 hội thoại.
- **Lớp 1** bổ sung so với `simulate.py`: báo lỗi hệ thống (`system_errors`), gọi `add_to_cart` lặp ≥ 4 lần trong 1 lượt (`tool_loops`), nói đã ghi nhận/tạo đơn khi chưa có đơn (`fake_order_claims`), hứa gọi lại khi chưa có đơn và chưa chuyển nhân viên (`unbacked_promises`), và `failure_cause`: `agent` (lỗi của chatbot) hay `customer_choice` (khách tự chọn không mua, mục tiêu không khả thi như ngân sách dưới máy rẻ nhất).
- **Đo độ lệch** (`calibrate`): từ kết quả lớp 1 suy ra mức trần điểm (vd sai giá → recommendation ≤ 2, nói đã có đơn khi chưa có → needs ≤ 2). Báo % hội thoại judge chấm vượt trần, % bỏ sót (điểm TB ≥ 4.5 dù lớp 1 thấy lỗi của agent), % chấm rất thấp hội thoại sạch. Lệch nhiều (vượt trần > 15% hoặc bỏ sót > 30%) thì viết rubric chặt hơn rồi `rejudge`.
- **Rubric**: `v1` (rubric của `simulate.py`), `v2` (thêm mức trần theo kết quả lớp 1), `v3` (thêm `failure_cause` và các kiểm tra mới, không trừ điểm khi khách tự chọn không mua). Lần chạy 250 kịch bản:

  | Rubric | Hội thoại | Vượt trần | Bỏ sót |
  |---|---|---|---|
  | v1 | 248 | 19% | 48% |
  | v2 (mẫu 64) | 64 | 14% | 48% |
  | v3 | 248 | 3% | 11% |

- **CSV** (`utf-8-sig`, mở được bằng Excel): 1 dòng / hội thoại: kết quả lớp 1, điểm và lý do của từng rubric, cột `<rubric>_over_cap`, cột `human_*` để trống cho người chấm, hội thoại (cắt ở 30.000 ký tự).
- Chuỗi model của khách giả lập, judge và brainstorm: Gemini 3.1 → Gemini 3.5 → NVIDIA → Groq → OpenRouter → 9Router (riêng judge: `max_tokens=4096`, timeout 180 giây). Khi các model miễn phí hết quota ngày, gần như mọi lượt dồn về 9Router: chậm (1–3 phút/lượt), có lượt timeout hoặc 503; chạy lại `rejudge` để chấm phần còn thiếu.

**Kết quả lần chạy 250 kịch bản** (`evals/results/20261103-153030-batch.csv`): đạt 161/250 (64%), 59 hội thoại trượt do agent (sau `recheck` với lớp 1 đã sửa). Tốt: `policy_question` 100%, `out_of_scope` 100%, `price_haggle` 92%. Yếu: `complaint_handoff` 15%, `multi_item` 20%, `accessory_only` 25%, `change_mind` 40%. Độ trễ mỗi lượt p50 25s, p95 86s.

Lỗi agent tìm được và cách sửa:

| Lỗi | Số hội thoại | Sửa |
|---|---|---|
| Không chuyển nhân viên khi khách viết không dấu, đòi "quản lý", "người phụ trách", "staff" | 17 | `HANDOFF` thêm cụm từ, khớp cả bản bỏ dấu |
| Không thêm được bản có RAM ("128GB - RAM 8GB": chỉ so số đầu) → báo lỗi hệ thống, gọi `add_to_cart` lặp, nói đã ghi nhận đơn khi chưa có | 14 / 12 / 5 | `_filter_storage` |
| Nói "đã chuyển nhân viên" nhưng `handoff` không bật | | `announces_handoff` trong `track_reply` |
| Chép nhãn `[SỐ ĐIỆN THOẠI]` vào câu trả lời | | `clean_answer` |
| Bịa hotline, hứa "quản lý ra gặp ngay", giờ giao | | Prompt quy tắc 12 (chỉ prompt) |
| Khuyên khách "liên hệ hotline" (chính prompt quy tắc 10, `POLICY_GUIDE` và câu lỗi của tool dặn model nói vậy) | 3 | Đổi thành hỏi khách có cần nhân viên hỗ trợ thêm không |
| `orders.thank_you`: `open_llm` bị gọi 2 lần (model chính và đầu chuỗi dự phòng), câu cảm ơn không qua `clean_answer` | | Bỏ khỏi chuỗi dự phòng, dọn bằng `clean_answer` |
| `AGENT_IMPL=agent`: `/order/cancel` ghi lịch sử với `as_node='track_reply'` (node chỉ có trong `graph.py`) → lỗi 500 | | `AS_NODE` theo `AGENT_IMPL` |
| Bản viết lại sau kiểm tra số liệu rỗng → gửi chuỗi rỗng | | Trả `EMPTY_FALLBACK` |
| `announces_handoff` tính cả câu hỏi, lời mời, điều kiện ("có muốn em chuyển cho nhân viên không?", "nếu cần em sẽ chuyển") là đã chuyển | 9 câu | Bỏ câu hỏi, điều kiện, phủ định, chuyển đơn cho bộ phận xác nhận/giao/kho |
| Tin có "!!" luôn tính là bực: "đẹp quá, chốt luôn!!" 2 lượt liền bị chuyển nhân viên | | Chỉ tính khi không kèm lời khen |
| Slot `storage` lấy số đầu tiên: "12gb ram 256gb" thành dung lượng 12GB | | Lấy số lớn nhất, dưới 32GB thì coi là RAM |
| "hỏi thêm cáp…" bị coi là yêu cầu mua cáp (`wanted_of`) | | "thêm" sau "hỏi/xem/tư vấn" không phải mua |
| Lỗi API 429/502 (mọi model hết quota khi 4 luồng simulator + judge chạy cùng lúc), 2 lượt quá 400s do vòng lặp RAM | 12 | Không phải lỗi logic; vòng lặp RAM đã sửa |
| Tự nhắc khuyến mãi khi chưa tra (6/7 là lớp 1 báo nhầm câu mời tra, câu phủ định) | 1 | Prompt quy tắc 4b; sửa `NEGATION_RE` |
| Trả lời rỗng (phần lớn sau vòng lặp RAM) | 5 | `call_model` hỏi lại 1 lần, vẫn rỗng thì câu mặc định |
| Mở form sớm (khách đã nói "gửi form", "lên đơn": lớp 1 báo nhầm) | 5 | Sửa `CONFIRM_RE` |
| Sai giá (30.000đ là phí giao nhanh trong chính sách: lớp 1 báo nhầm) | 2 | `reference_data` lấy thêm số tiền trong `data_schema/policies/` |
| Khách báo đồ đã mua bị lỗi không được chuyển nhân viên | 3 | `HANDOFF` thêm "đã/mới/vừa mua … bị lỗi/hỏng/sạc không vào" |
| Lời cảm ơn sau đặt hàng đổi dấu tên khách ("Tran Thi Bich" → "Trần Thị Bích") | 1 | `keep_name` trả lại đúng tên khách điền |
| `agent.py`: `open_llm` vừa là model chính vừa đứng đầu chuỗi dự phòng | | Bỏ khỏi chuỗi dự phòng |

Chạy lại 29 kịch bản lỗi sau 2 sửa đầu (`evals/results/20261104-152937-batch.jsonl`): đạt 3 → 25, chuyển nhân viên 3 → 17/20, gọi `add_to_cart` lặp 6 → 0, nói đã có đơn khi chưa có 3 → 0, điểm judge v3 trung bình 3.75 → 4.36.

Chạy lại 38 kịch bản sau mọi sửa (`evals/results/20261104-190507-batch.jsonl`: 20 `complaint_handoff`, 7 `change_mind` và các kịch bản từng lỗi), so với chính các kịch bản đó trong lần chạy 250:

| Chỉ số | Trước | Sau |
|---|---|---|
| Đạt | 3/38 | 36/38 |
| Chuyển nhân viên (`complaint_handoff`) | 3/20 | 20/20 |
| Nói "đã/sẽ chuyển" nhưng `handoff` không bật | 17 | 0 |
| Trả lời rỗng / timeout | 5 / 12 | 0 / 0 |
| Hứa khi chưa có cơ sở / mở form sớm | 1 / 1 | 0 / 0 |
| Điểm judge v3 trung bình | 3,77 | 4,66 |

2 kịch bản trượt đều do khách bỏ đi vì chê giá (`accessory_only-11`, `change_mind-13`). Model vẫn có lúc khuyên "liên hệ hotline" (không bịa số): đã sửa ở prompt và câu lỗi của tool, chưa chạy lại để đo. Chưa chạy lại cả 250 kịch bản; lần chạy này Gemini quá tải (503/504) nên một phần lượt của khách giả lập và judge dùng model dự phòng.

## Hạn chế

- State hội thoại nằm trong RAM (`InMemorySaver`): mất khi restart, chỉ chạy được 1 worker.
- Khách quen nhận ra theo trình duyệt (`localStorage`), không có đăng nhập: máy dùng chung thì người sau thấy tên và món đơn gần nhất của người trước; xóa dữ liệu trang hoặc đổi trình duyệt thì thành khách mới. Đóng tab không gọi `/session` nên lịch sử cuộc cũ vẫn nằm trong RAM tới khi restart.
- Form đặt hàng chỉ hỗ trợ giao tận nơi và thanh toán khi nhận hàng; chưa có trang quản lý đơn, đơn chỉ nằm trong database.
- Luật nhận diện từ khóa không hiểu mọi cách nói (mỉa mai, viết tắt chưa có trong `ABBREVIATIONS`; viết không dấu mới chỉ xử lý cho chuyển nhân viên), kể cả câu từ chối phụ kiện bán kèm. Khách báo đồ đã mua bị lỗi chỉ được nhận ra khi có dấu hiệu đã mua ("mới mua", "mua tuần trước"); cách nói khác chỉ chuyển khi model báo chuyển (`announces_handoff`).
- Shop không có dữ liệu hotline, địa chỉ chi nhánh: chống bịa chỉ bằng prompt, model vẫn có thể bịa hoặc khuyên khách "gọi hotline".
- Kiểm tra bịa số liệu chỉ xét số kèm đơn vị và chỉ trong `graph.py`: không bắt được câu bịa không có số ("máy chống nước tốt", sai tên màu), số trùng ngẫu nhiên với một số có trong nguồn, hay câu trả lời của `agent.py`.
- API chưa có xác thực và giới hạn request.
- Email xác nhận gửi bằng `BackgroundTasks` trong tiến trình API: server khởi động lại đúng lúc đang gửi thì mất email, đơn kẹt ở `email_status = 'pending'`, không tự gửi lại. Làm thật nên dùng outbox (ghi yêu cầu gửi trong cùng transaction tạo đơn, worker riêng gửi và thử lại) hoặc hàng đợi (Celery, ARQ).
- Nhận diện số điện thoại, email bằng regex: cách viết lạ ("không chín một hai…", "a [at] gmail") không bị che. Chưa có cách đổi `PII_KEY` (xoay khóa) cho dữ liệu đã mã hóa.
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
| Sửa `index.html` nhưng trình duyệt vẫn hiện bản cũ | Build lại `api`, rồi tải lại trang bằng Ctrl+F5 |
| Thẻ xác nhận đơn hoặc form đặt hàng bị cắt, mất nội dung | Giữ `#log > * { flex-shrink: 0; }` trong `index.html`: thiếu dòng này, khung chat đầy sẽ ép dẹp các thẻ có `overflow: hidden` |
| `db-seed` báo `cannot truncate a table referenced in a foreign key constraint` | `phone_seed.sql` sinh từ bản `scrape_phone.py` cũ (TRUNCATE). Sinh lại: `cd data_schema && python scrape_phone.py --from-json`, rồi chạy lại `db-seed` (bản mới upsert, giữ đơn hàng) |
| Bot chỉ đưa khoảng giá khi khách nêu tên máy | Xem log tool: phải là `get_product_details`; kiểm tra quy tắc 2 trong `prompts.py` |
| `InvalidUpdateError: At key '…': Can receive only one value per step` | Nhiều tool ghi cùng khóa state trong 1 bước (gọi song song). Khóa đó cần `Annotated[kiểu, reducer]` với reducer đúng nghĩa, xem [Nhiều tool cập nhật state cùng lúc](#nhiều-tool-cập-nhật-state-cùng-lúc) |
| Không thấy email xác nhận | Xem `email_status` của đơn: `skipped` là khách không nhập email hoặc chưa đặt `SMTP_HOST`; `failed` thì xem `docker compose logs api \| grep "email xác nhận"` (Mailpit chưa chạy, sai `SMTP_*`); `pending` mãi là server khởi động lại đúng lúc đang gửi |
| Gửi email lỗi `value is not a valid email address` | Tên miền trong `MAIL_FROM` không được là `.local`, `.localhost`…: dùng vd `no-reply@phoneshop.example` |
| Câu trả lời chậm tới ~2 phút | 9Router chọn model chậm hoặc không chạy: chờ hết 60 giây, thử lại 1 lần rồi mới sang OpenRouter. Kiểm tra `curl http://localhost:20128/v1/models`, bỏ model chậm khỏi combo |
| 9Router chạy trên máy được nhưng container báo lỗi kết nối | Trong container `localhost` là chính container; compose đã đặt `ROUTER9_BASE_URL=http://host.docker.internal:20128/v1`. 9Router phải nhận kết nối từ ngoài, không chỉ `127.0.0.1` |
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
