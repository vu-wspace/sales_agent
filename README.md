# Phone Sales Agent

Trợ lý chat tiếng Việt cho cửa hàng bán lẻ iPhone, Samsung Galaxy và phụ kiện, xây dựng với LangChain/LangGraph. Dự án demo phục vụ học tập.

## Tính năng

- Tư vấn điện thoại theo ngân sách và nhu cầu; báo giá, màu, dung lượng, tình trạng hàng từng phiên bản.
- Tìm phụ kiện tương thích (ốp lưng, cường lực, sạc, cáp, S Pen…).
- Trả lời chính sách đổi trả, giao hàng, trả góp, bảo hành bằng RAG.
- Nhớ trạng thái hội thoại bán hàng (nhu cầu, máy đã giới thiệu, lo ngại, cảm xúc) và quản lý giỏ hàng.
- Bán thêm: gợi ý dung lượng lớn hơn khi nhu cầu cần (kèm số tiền chênh), máy nhỉnh hơn ngân sách; khi khách chốt mua thì mời sạc chính hãng nếu máy không kèm sạc, rồi ốp lưng, cường lực.
- Đặt hàng ngay trong khung chat: khách xác nhận thì hiện form (tên, số điện thoại, địa chỉ), đơn lưu vào Postgres, tóm tắt đơn lấy từ database.
- Chuẩn hóa tin nhắn (Unicode, dấu kiểu cũ, viết tắt "ko", "dc", "20tr"…) và kiểm tra câu trả lời không bịa số liệu (giá, thông số, chính sách).
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

Yêu cầu: Docker và các API key trong [Cấu hình](#cấu-hình). Khi phát triển trên máy cần thêm Python 3.12.15 và [uv](https://docs.astral.sh/uv/).

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
docker compose run --rm db-seed                           # nạp lại dữ liệu sản phẩm (giữ đơn hàng, không mất volume)
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
| `POST` | `/order` | Gửi form đặt hàng, tạo đơn từ giỏ hàng của hội thoại |
| `POST` | `/order/cancel` | Khách bấm "Hủy đặt hàng" trên form |
| `GET` | `/health` | Trạng thái Qdrant và số đoạn chính sách |

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
 "address": "123 Lê Hồng Phong, Quy Nhơn, Gia Lai", "email": "a@example.com",
 "fulfillment_type": "delivery", "payment": "cod"}

// response
{"thread_id": "5f0c…", "order_id": "DH3F9A1C2B",
 "order": {"order_id": "DH3F9A1C2B", "name": "Nguyễn Văn A", "phone": "0912345678", "email": "a@example.com",
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
- Kiểm tra giống form: tên ≥ 2 ký tự, số điện thoại `0`/`+84` + 9 số (bỏ dấu cách, chấm, gạch), địa chỉ ≥ 10 ký tự; `email` không bắt buộc (bỏ trống hoặc `null`), có nhập thì phải đúng dạng `x@y.z`. Hiện chỉ nhận `delivery` và `cod`.
- Mã lỗi: `422` thông tin sai, `409` giỏ trống hoặc hết hàng, `503` không ghi được database.

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

### Đặt hàng

1. Khách nêu tên máy mà chưa nói phiên bản ("Tui muốn mua iPhone 17", "cho tui coi Galaxy S26"): agent gọi `get_product_details`, liệt kê mỗi dung lượng 1 dòng (giá, màu, còn hàng) và hỏi khách chọn phiên bản. Khách nói đủ máy, dung lượng, màu thì sang bước 2 luôn.
2. Lượt khách chọn máy ("chốt S26 256GB đen"): agent `add_to_cart` rồi bán kèm 3 lượt (xem [Bán thêm](#bán-thêm-up-sell-cross-sell)): mời cùng lúc củ sạc và ốp lưng, cường lực → mời lại ốp lưng, cường lực → xác nhận đơn. `show_order_form` từ chối mở form trong lượt vừa thêm hàng vào giỏ và khi bán kèm chưa xong.
3. Khách xác nhận ("xác nhận đặt hàng", "chốt đơn"): agent gọi `show_order_form`, giao diện hiện form (`index.html`, theo thiết kế `static/order_form.html`) với giỏ hàng và tổng tiền.
4. Khách gửi form → `POST /order` (`orders.py`), trong 1 transaction:
   - trừ tồn kho, thiếu hàng thì hủy cả đơn;
   - khách cũ nhận theo số điện thoại (cập nhật tên, địa chỉ, email nếu có nhập), khách mới thì tạo `customers` (mã `KH…`);
   - email luôn lưu vào đơn (`orders.recipient_email`); chỉ lưu vào `customers.email` khi email chưa thuộc khách khác (`uq_customers_email` không cho 2 khách trùng email), để đơn không bị lỗi;
   - ghi `orders` (mã `DH…`) và `order_items`.
5. Câu trả lời = tóm tắt đơn viết bằng template, đọc lại từ database (khách hàng, địa chỉ, sản phẩm, tổng tiền) + 1–2 câu cảm ơn do LLM viết. LLM chỉ nhận tên khách, không viết số liệu; LLM lỗi thì dùng câu cảm ơn cố định. Trang chat hiện tóm tắt thành thẻ xác nhận (`.receipt`: mã đơn, thông tin nhận hàng, sản phẩm, tổng tiền), lời cảm ơn là bong bóng riêng.
6. Đơn được ghi vào lịch sử hội thoại, giỏ hàng làm trống; gửi form lần nữa nhận `409`.
7. Khách bấm **Hủy đặt hàng** trên form → `POST /order/cancel`: bỏ form, hiện câu "Thật tiếc, em đã hủy yêu cầu đặt hàng… Anh/chị có cần em hỗ trợ hay tư vấn gì thêm nữa không?", giỏ hàng làm trống, không tạo đơn.

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
| `cross_sell` | Các bước bán kèm đã làm: `bundle`, `again`, `confirm`; đặt hàng hoặc hủy đơn thì làm lại từ đầu |
| `declined` | Tin nhắn mới nhất của khách là từ chối ("không cần", "khỏi"…) |
| `order_form` | Lượt này hiện form đặt hàng |
| `orders` | Mã các đơn đã đặt trong hội thoại |

Mỗi bước có hướng dẫn riêng được đưa vào system prompt, vd `closing`: xác nhận giỏ và tổng tiền, hỏi khách có chốt đơn không, khách xác nhận thì mở form đặt hàng; không tự nêu phí giao hàng hay kỳ hạn trả góp.

### Bán thêm (up-sell, cross-sell)

Tool trả sẵn dữ liệu để model gợi ý bằng số liệu thật, không tự tính hay đoán:

| Loại | Dữ liệu | Khi nào model gợi ý |
|---|---|---|
| Dung lượng lớn hơn | `get_product_details` có lọc dung lượng thì kèm dòng `Dung lượng lớn hơn còn hàng: 512GB từ 45.499.000đ (thêm 6.500.000đ)…` | Khách cần nhiều bộ nhớ (quay video, chơi game…) mà chọn bản thấp |
| Máy nhỉnh hơn ngân sách | `search_products` kèm tối đa 2 máy vượt ngân sách không quá 10%, ghi số tiền vượt | Máy đó hợp nhu cầu hơn rõ rệt; chỉ nhắc 1 máy |
| Phụ kiện | `find_accessories` | Sau khi khách cho điện thoại vào giỏ, theo các bước bên dưới |

Mỗi câu trả lời tối đa 1 gợi ý bán thêm.

**Bán kèm phụ kiện** (`cross_sell_step`, `cross_sell_hint` trong `conversation.py`): 3 bước, mỗi bước đúng 1 lượt chat, không mời khi khách đang bực. Cuối mỗi lượt `track_reply` ghi bước vừa làm vào `cross_sell`, lượt sau sang bước kế tiếp.

| Bước | Khi nào | Model làm gì |
|---|---|---|
| 1. `bundle` | Lượt khách chọn máy (giỏ vừa có điện thoại) | Mời **cùng lúc** củ sạc và ốp lưng, cường lực: gọi `find_accessories` 2 lần, trả lời 2 nhóm "Củ sạc" và "Ốp lưng, cường lực", mỗi món 1 gạch đầu dòng kèm giá. Hộp đã có cáp: nói rõ, chỉ mời 2–3 củ sạc chính hãng không kèm cáp, từ rẻ tới đắt. Hộp không có cáp: củ sạc và cáp. Máy đã kèm củ sạc hoặc giỏ đã có củ sạc: chỉ mời ốp lưng, cường lực |
| 2. `again` | Khách lấy củ sạc hoặc không lấy gì | Thêm món khách chọn vào giỏ, mời lại nhẹ ốp lưng, cường lực (không nhắc lại củ sạc). Khách đã lấy ốp/cường lực thì bỏ qua bước này |
| 3. `confirm` | Khách lấy phụ kiện hoặc không | Thêm món khách chọn, không mời thêm; xác nhận đơn: liệt kê giỏ hàng, tổng tiền, hỏi khách xác nhận đặt hàng. Khách đã nói xác nhận trong tin này thì mở form luôn |

`show_order_form` không mở form khi đang ở bước 1 hoặc 2, kể cả khi khách nói "không cần, xác nhận đặt hàng" (model hay mở form luôn, bỏ qua bước mời ốp lưng, cường lực). Đặt hàng (`POST /order`) hoặc hủy (`POST /order/cancel`) thì `cross_sell` về rỗng, lần mua sau bán kèm lại từ bước 1.

- **Hộp máy có gì:** iPhone đọc từ mục "Trong Hộp Sản Phẩm" (cáp USB-C, không có củ sạc). Web Samsung không có mục này, nên `IN_BOX_BY_BRAND` trong `tools.py` **giả định** Galaxy chỉ kèm cáp, không kèm củ sạc; cần kiểm tra lại với từng model. Máy không rõ thông tin thì bỏ qua bước 1.
- **Chính hãng:** phụ kiện cùng hãng với máy; hàng Beats, mophie, Belkin, Anker do cửa hàng Apple bán không tính là chính hãng. `find_accessories` xếp còn hàng trước, chính hãng trước, củ sạc cắm tường trước sạc không dây hoặc sạc ô tô, rồi rẻ trước, và nhận nhiều loại một lúc (`'sạc, cáp'`: mỗi loại 2 món). `'củ sạc'` chỉ trả củ sạc cắm tường, bỏ sạc không dây, MagSafe, sạc ô tô; nếu hộp máy đã có cáp thì bỏ cả bộ củ sạc kèm cáp (khách không mua thêm cáp lần hai). Bước 1 dùng loại này khi hộp đã có cáp, nên Galaxy hiện chỉ có 1 củ sạc chính hãng để mời (60W). Danh sách 1 loại tách riêng mục "Hãng khác" sau hàng chính hãng. Bước bán kèm gọi `find_accessories(..., genuine_only=True)`: tool bỏ hẳn hàng hãng khác (chỉ dặn trong prompt thì model vẫn lấy củ sạc Apple, Belkin cho Galaxy cho đủ số món), có 1 món thì mời 1, không có món nào thì bỏ nhóm đó.
- **Mẫu câu trả lời:** mỗi bước có mẫu riêng trong `cross_sell_hint` (`_bundle_hint` cho bước 1): bước 1 "Em đã thêm … vào giỏ" + dòng hộp máy có gì + nhóm "Củ sạc chính hãng" + nhóm "Ốp lưng, cường lực chính hãng" + "Anh/chị muốn lấy thêm món nào không?"; bước 2 mời lại ốp lưng, cường lực + "Nếu không cần thêm, anh/chị xác nhận đặt hàng giúp em nhé."; bước 3 "Đơn hàng của anh/chị:" + từng món + "Tổng cộng" + hỏi xác nhận.
- **Nhận diện từ chối** (`DECLINE` trong `conversation.py`) dùng từ khóa: "không cần", "khỏi", "không lấy", "chưa cần", "để sau", "thôi không", "vậy thôi"… Các bước bán kèm không phụ thuộc vào việc nhận ra từ chối (khách lấy hay không đều sang bước kế tiếp), nhưng cách nói lạ ("để em tính", viết không dấu "ko can") vẫn có thể ảnh hưởng tới cảm xúc và bước bán hàng. Chữ "thôi" đứng một mình không tính là từ chối vì hay dùng khi chốt ("lấy luôn thôi").

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

Câu trả lời cuối được dọn: bỏ phần `<think>…</think>` một số model để lộ, bỏ dòng trống thừa, bỏ chữ "ạ" đứng riêng ("không ạ?" → "không?"; "Dạ", "lạ" giữ nguyên). Prompt cũng dặn không dùng "ạ"; bỏ trong code là chốt chặn khi model vẫn viết.

Ví dụ: Galaxy S26 chưa có thông số trong dữ liệu, model tra web rồi trả lời "camera chính 200 MP" → bị yêu cầu viết lại → "dữ liệu cửa hàng chưa có thông số camera của Galaxy S26".

### Tools

| Tool | Nguồn | Dùng khi |
|---|---|---|
| `search_products` | Postgres | Tìm máy theo dòng, hãng, ngân sách |
| `get_product_details` | Postgres | Giá, tồn kho từng phiên bản, thông số của 1 máy |
| `find_accessories` | Postgres | Phụ kiện tương thích, 5 món mỗi trang |
| `add_to_cart`, `remove_from_cart` | Postgres + state | Khách chốt mua hoặc đổi ý (chỉ `graph.py`) |
| `show_order_form` | State | Khách xác nhận đặt hàng: hiện form đặt hàng (chỉ `graph.py`) |
| `search_policy` | Qdrant | Đổi trả, giao hàng, trả góp, bảo hành, lỗi máy |
| `web_search` | Tavily | Đánh giá, tin tức ngoài dữ liệu nội bộ |

- Tên máy được chuẩn hóa (bỏ dấu, viết tắt như `ip`, `promax`, `17pro`). Khi nhiều sản phẩm khớp, tool trả danh sách để model hỏi lại khách.
- Tồn kho chỉ trả `còn hàng` / `sắp hết` / `hết hàng`, không trả số lượng.
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
| Chính | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` |
| Dự phòng 1 | NVIDIA `nvidia/nemotron-3.5-super-vl-120b-a12b` (bật thinking) |
| Dự phòng 2 | `gemini-3.1-flash-lite` |
| Dự phòng 3 | `gemini-3.5-flash-lite` |
| Dự phòng 4 | Groq `qwen/qwen3.8-27b` |

`FallbackMiddleware` thử lần lượt các model tới khi có model trả lời. Model gặp lỗi lặp lại sẽ tạm bị bỏ qua: hết quota ngày hoặc lỗi cấu hình (sai key, sai tên model) 1 giờ, giới hạn theo phút 1 phút, lỗi chứng chỉ SSL 10 phút. Lỗi tạm thời (mất mạng, 5xx) chỉ chuyển model cho lần gọi đó. Mỗi lần chuyển model đều ghi log (`docker compose logs api | grep LLM`).

Model `:free` của OpenRouter giới hạn 50 request/ngày (1000 request/ngày nếu tài khoản đã nạp từ 10 USD), reset lúc 7:00 sáng giờ Việt Nam; mỗi câu hỏi tốn 2–3 request.

## Dữ liệu

Database `phone` (schema `data_schema/phone.sql`, dữ liệu `data_schema/output/phone_seed.sql`):

| Bảng | Nội dung |
|---|---|
| `products` | Điện thoại và phụ kiện: tên, hãng, mô tả, thông số (JSON) |
| `variants` | Phiên bản theo màu và dung lượng: SKU, giá (VND), tồn kho |
| `compatibility` | Cặp phụ kiện – điện thoại tương thích |
| `customers` | Khách đặt hàng: mã `KH…`, tên, số điện thoại, email (duy nhất, không phân biệt hoa thường), địa chỉ mặc định |
| `orders` | Đơn hàng: mã `DH…`, khách, người nhận (tên, số điện thoại, email), địa chỉ, cách nhận, thanh toán, tổng tiền, trạng thái (`pending`) |
| `order_items` | Sản phẩm trong đơn: phiên bản, số lượng, đơn giá lúc đặt |
| `cart_item` | Có trong database nhưng code chưa dùng: giỏ hàng nằm trong state hội thoại (`AgentState.cart`) |

Schema đơn hàng ở `data_schema/orders.sql` (`CREATE TABLE IF NOT EXISTS`, cột thêm sau như email dùng `ADD COLUMN IF NOT EXISTS`), web app chạy file này mỗi lần khởi động: database mới có đủ bảng, database cũ được bổ sung cột, database đã đủ thì không đổi gì. `order_items.variant_id` có khóa ngoại tới `variants`, nên `phone_seed.sql` không TRUNCATE `variants`/`products` (Postgres không cho TRUNCATE bảng đang bị tham chiếu) mà ghi đè từng dòng (`INSERT … ON CONFLICT DO UPDATE`), rồi xóa phiên bản, sản phẩm không còn bán. Phiên bản đã nằm trong đơn hàng thì giữ lại cho đơn cũ, đặt tồn kho 0. Mỗi lần seed, tồn kho về lại số giả lập (số đã trừ khi đặt hàng không giữ).

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
│   ├── phone.sql, orders.sql, scrape_phone.py
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
│   ├── data_prep.py            # Tách chính sách thành đoạn
│   ├── vector_db.py            # Embedding và Qdrant
│   ├── orders.py               # Lưu đơn đặt hàng, tóm tắt đơn
│   ├── api.py                  # FastAPI
│   └── static/                 # Giao diện chat (index.html), mẫu form đặt hàng (order_form.html)
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

174 test, không gọi LLM, mạng hay database thật (dùng model và dữ liệu giả). Khi thêm thư viện bằng `uv add`, cập nhật `requirements.txt` cho khớp (`tests/test_requirements.py` kiểm tra).

### Eval

Eval gọi LLM, Postgres, Qdrant, Tavily thật trên 46 câu hỏi cố định (`evals/dataset.jsonl`) và chấm điểm bằng Gemini. Câu `cross-01` (nhóm `cross_sell`) cần giỏ hàng nên chỉ chạy với `--target graph`. Chuỗi dự phòng bị tắt để điểm phản ánh đúng model chính.

```bash
docker compose up -d postgres qdrant
uv run python evals/run_eval.py --retrieval-only            # chỉ chấm truy xuất chính sách
uv run python evals/run_eval.py --target graph              # chấm graph.py (mặc định agent.py)
uv run python evals/run_eval.py --only product,acc-01       # lọc theo nhóm hoặc id
```

Cả bộ cần khoảng 100 request, vượt quota ngày của OpenRouter gói free, nên cần chia làm nhiều lần.

Kết quả gần nhất (`graph.py`, chạy ngày 29–30/10/2026, mỗi câu lấy lần chạy mới nhất). Truy xuất chính sách: hit@1 89%, hit@3 100%, MRR 0,95.

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

9 câu thêm/sửa ngày 02/11/2026 chưa có trong bảng: `spec-01`, `spec-02`, `upsell-01`, `upsell-02`, `cross-01` (bán kèm 3 bước: củ sạc + ốp lưng, cường lực cùng 1 lượt), `variants-01`, `variants-02` (khách nêu tên máy thì liệt kê phiên bản), `ground-01` (không nêu thông số không có trong dữ liệu), `norm-01` (tin nhắn viết tắt). Chạy khi OpenRouter còn quota: `uv run python evals/run_eval.py --target graph --only spec-01,spec-02,upsell,cross_sell,variants-01,variants-02,ground-01,norm-01`.

Độ trễ trung bình 15,8 giây mỗi câu do model chính là model suy luận. Kết quả chi tiết lưu ở `evals/results/` (không commit).

## Hạn chế

- State hội thoại nằm trong RAM (`InMemorySaver`): mất khi restart, chỉ chạy được 1 worker.
- Form đặt hàng chỉ hỗ trợ giao tận nơi và thanh toán khi nhận hàng; chưa có trang quản lý đơn, đơn chỉ nằm trong database.
- Luật nhận diện từ khóa không hiểu mọi cách nói (viết không dấu, mỉa mai, viết tắt chưa có trong `ABBREVIATIONS`), kể cả câu từ chối phụ kiện bán kèm.
- Kiểm tra bịa số liệu chỉ xét số kèm đơn vị và chỉ trong `graph.py`: không bắt được câu bịa không có số ("máy chống nước tốt", sai tên màu), số trùng ngẫu nhiên với một số có trong nguồn, hay câu trả lời của `agent.py`.
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
| Sửa `index.html` nhưng trình duyệt vẫn hiện bản cũ | Build lại `api`, rồi tải lại trang bằng Ctrl+F5 |
| Thẻ xác nhận đơn hoặc form đặt hàng bị cắt, mất nội dung | Giữ `#log > * { flex-shrink: 0; }` trong `index.html`: thiếu dòng này, khung chat đầy sẽ ép dẹp các thẻ có `overflow: hidden` |
| `db-seed` báo `cannot truncate a table referenced in a foreign key constraint` | `phone_seed.sql` sinh từ bản `scrape_phone.py` cũ (TRUNCATE). Sinh lại: `cd data_schema && python scrape_phone.py --from-json`, rồi chạy lại `db-seed` (bản mới upsert, giữ đơn hàng) |
| Bot chỉ đưa khoảng giá khi khách nêu tên máy | Xem log tool: phải là `get_product_details`; kiểm tra quy tắc 2 trong `prompts.py` |
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
