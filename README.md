# Phone Sales Agent: trợ lý AI tư vấn bán điện thoại

Trợ lý chat tiếng Việt cho cửa hàng bán lẻ iPhone, Samsung Galaxy và phụ kiện: tư vấn máy theo ngân sách, báo giá và tình trạng hàng, trả lời chính sách, và chốt đơn vào giỏ hàng. Dự án demo phục vụ học tập, xây dựng với LangChain/LangGraph.

Agent có thể:

- Tư vấn iPhone, Samsung Galaxy theo ngân sách và nhu cầu; báo giá, màu, dung lượng và tình trạng hàng của từng phiên bản.
- Tìm phụ kiện tương thích (ốp lưng, cường lực, sạc, cáp, S Pen…).
- Trả lời chính sách đổi trả, giao hàng, trả góp, bảo hành (RAG).
- Nhớ trạng thái hội thoại bán hàng (nhu cầu, máy đã giới thiệu, lo ngại, cảm xúc của khách) và quản lý giỏ hàng.
- Hiện trạng thái và chữ ngay trong lúc trả lời (stream).
- Tự chuyển sang LLM dự phòng khi model chính lỗi hoặc hết quota.

| Thành phần | Công nghệ |
|---|---|
| Dữ liệu sản phẩm: 23 điện thoại, 134 phụ kiện, 377 phiên bản (cào từ apple.com/vn, samsung.com/vn) | Postgres 18, database `phone` |
| Chính sách: 5 file `.md`, tách thành 66 đoạn | Qdrant + embedding `bkai-foundation-models/vietnamese-bi-encoder` |
| Agent | LangGraph `StateGraph` (`graph.py`) hoặc `create_agent` (`agent.py`) |
| LLM | OpenRouter `nemotron-3-ultra-550b-a55b:free` → NVIDIA `nemotron-3.5-super-vl-120b-a12b` → Gemini 3.1 flash-lite → Gemini 3.5 flash-lite → Groq `qwen3.8-27b` |
| Web | FastAPI + trang chat một file, chạy bằng Docker Compose |
| Kiểm thử | 118 test (không gọi API thật), bộ eval 37 câu |

**Chạy nhanh** (cần Docker; chi tiết ở [Cài đặt](#cài-đặt)):

```bash
cp .env.example .env          # rồi điền API key
docker compose up -d --build  # Postgres (tự nạp dữ liệu) + Qdrant + web app (tự nạp chính sách)
# mở http://localhost:8000
```

## Mục lục

- [Kiến trúc](#kiến-trúc)
- [Trạng thái hội thoại](#trạng-thái-hội-thoại)
- [Các tool](#các-tool)
- [RAG chính sách](#rag-chính-sách)
- [LLM và chuỗi dự phòng](#llm-và-chuỗi-dự-phòng)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [Chạy trên máy](#chạy-trên-máy)
- [Web app và API](#web-app-và-api)
- [Test](#test)
- [Đánh giá (eval)](#đánh-giá-eval)
- [Kết quả đo hiệu năng](#kết-quả-đo-hiệu-năng)
- [Hạn chế và việc còn mở](#hạn-chế-và-việc-còn-mở)
- [Mở rộng dự án](#mở-rộng-dự-án)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Lưu ý trên Windows](#lưu-ý-trên-windows)

## Kiến trúc

```mermaid
flowchart TB
    browser["Trình duyệt<br/>(trang chat)"] -- "POST /chat/stream<br/>hoặc /chat" --> api["FastAPI<br/>api.py"]
    studio["LangGraph Studio<br/>(langgraph dev)"] --> agent
    api -- "state theo thread_id<br/>(InMemorySaver)" --> agent["Agent<br/>graph.py (mặc định) / agent.py"]
    agent <--> chain

    subgraph chain["Chuỗi LLM dự phòng (fallback.py)"]
        direction LR
        openrouter["OpenRouter<br/>nemotron-3-ultra"] -. lỗi, hết quota .-> nvidia["NVIDIA<br/>nemotron-3.5-super"]
        nvidia -.-> gemini31["Gemini 3.1<br/>flash-lite"] -.-> gemini35["Gemini 3.5<br/>flash-lite"] -.-> groq["Groq<br/>qwen3.8-27b"]
    end

    agent -- "gọi tool" --> tools

    subgraph tools["Tools (tools.py)"]
        search_products["search_products"]
        get_details["get_product_details"]
        find_acc["find_accessories"]
        add_cart["add_to_cart"]
        remove_cart["remove_from_cart"]
        search_policy["search_policy"]
        web_search["web_search"]
    end

    search_products & get_details & find_acc & add_cart --> postgres[("Postgres<br/>products · variants · compatibility")]
    add_cart & remove_cart --> cart[("State<br/>cart")]
    search_policy --> qdrant[("Qdrant<br/>66 đoạn chính sách")]
    web_search --> tavily(["Tavily"])
```

### Luồng một lượt chat (`graph.py`)

```mermaid
flowchart LR
    start((START)) --> track_user["track_user<br/>đọc tin nhắn khách"]
    track_user --> model["model<br/>gọi LLM"]
    model -- "có tool_call" --> tool_node["tools<br/>chạy tool"]
    tool_node --> track_tools["track_tools<br/>đọc tham số tool"]
    track_tools --> model
    model -- "trả lời xong" --> track_reply["track_reply<br/>ghi máy đã giới thiệu"]
    track_reply --> finish((END))
```

1. `api.py` nhận `message` và `thread_id`, lấy state cũ của thread từ checkpointer (`InMemorySaver`).
2. `track_user` đọc tin nhắn của khách (nhu cầu, lo ngại, cảm xúc, có đòi gặp nhân viên không) và tính bước bán hàng.
3. `model` gọi LLM với system prompt, **tóm tắt trạng thái hội thoại**, và lịch sử đã cắt gọn (`history.py`: 4 lượt gần nhất, kết quả tool của lượt cũ rút còn 300 ký tự).
4. Nếu LLM gọi tool, `track_tools` đọc tham số tool (ngân sách, hãng, dung lượng, màu) và tính lại bước (vd vừa thêm vào giỏ → `closing`).
5. Khi LLM trả lời xong, `track_reply` ghi lại các sản phẩm thật sự được giới thiệu trong câu trả lời.
6. API trả câu trả lời cùng `tools`, `stage`, `cart`, `handoff`. Trang chat dùng `/chat/stream` để hiện trạng thái và chữ ngay trong lúc agent chạy.

### System prompt (`prompts.py`)

`AGENT_PROMPTS['agent_main']` có 15 quy tắc; `agent_graph` (dùng cho `graph.py`) thêm 3 quy tắc về trạng thái và giỏ hàng.

| Nhóm | Quy tắc chính |
|---|---|
| Xưng hô | Tiếng Việt, xưng "em", gọi "anh/chị", không đoán giới tính |
| Dùng tool | Tìm máy, tầm giá → `search_products`; giá, màu, tồn kho, thông số 1 máy → `get_product_details` (nhiều máy khớp thì hỏi lại); phụ kiện → `find_accessories`; chính sách, lỗi máy, bảo hành → `search_policy` (làm theo "Hướng dẫn trả lời" đi kèm kết quả); `web_search` chỉ khi dữ liệu nội bộ không có, không dùng cho giá hay chính sách |
| Trả lời | Chỉ dựa trên kết quả tool; **không bao giờ nói không có hàng/chưa có thông tin khi chưa gọi tool trong lượt đó** (quy tắc 10); giữ đúng giá, chỉ dòng có "(giá tham khảo)" mới nói là tham khảo; tồn kho chỉ nói còn hàng / sắp hết / hết hàng; tư vấn tối đa 3 máy; liệt kê tối đa 5 dòng; tối đa khoảng 100 từ |
| Chuyển nhân viên | Khiếu nại, máy lỗi, đơn giao sai/chậm/thiếu, hủy đơn đã thanh toán, đòi gặp người thật |
| Bảo mật | Không hỏi OTP, mật khẩu, PIN, số thẻ, CVV; không tiết lộ thông tin khách khác |
| Ngoài phạm vi | Không gọi tool, từ chối trong 1–2 câu, mời tư vấn sản phẩm |
| Trạng thái, giỏ hàng (chỉ `graph.py`) | Làm theo hướng dẫn của bước hiện tại, không hỏi lại nhu cầu đã biết; chỉ `add_to_cart` khi khách nói mua/chốt; thêm vào giỏ chưa phải đặt hàng, nhân viên sẽ liên hệ xác nhận |

### Hai cách dựng agent

Cả hai dùng chung LLM, tool tra cứu, prompt, cách cắt lịch sử và chuỗi dự phòng.

| | `graph.py` (web app dùng mặc định) | `agent.py` |
|---|---|---|
| Dựng bằng | `StateGraph` tự dựng | `create_agent` của LangChain |
| Trạng thái hội thoại, giỏ hàng | Có | Không (chỉ `messages`) |
| Trong LangGraph Studio | `phone_graph` | `agent_main` |

Đổi web app sang `agent.py`: đặt `AGENT_IMPL=agent` trong `.env`.

## Trạng thái hội thoại

`AgentState` (`src/phone_sales/conversation.py`) được checkpointer lưu theo `thread_id`, nên giữ qua các lượt.

| Trường | Ý nghĩa | Cập nhật bởi |
|---|---|---|
| `messages` | Lịch sử tin nhắn (reducer `add_messages`, nối thêm) | Mọi node |
| `stage` | Bước bán hàng: `discovery`, `recommending`, `comparing`, `closing`, `handoff` | Tính lại sau mỗi node |
| `slots` | Nhu cầu: `budget_max`, `brand`, `purposes`, `storage`, `color` (reducer gộp, list cộng dồn) | Tin nhắn khách và tham số tool |
| `shown_products` | Máy đã giới thiệu: có trong kết quả tool của lượt **và** được nhắc trong câu trả lời | `track_reply` |
| `cart` | Giỏ hàng: `variant_id`, `name`, `storage`, `color`, `price`, `quantity` | `add_to_cart`, `remove_from_cart` |
| `objections` | Lo ngại: chê giá cao, lo pin yếu, máy nóng, nặng/to, chê camera, lo lỗi, không thích màu | Tin nhắn khách |
| `sentiment_history` | `positive` / `neutral` / `negative` từng lượt | Tin nhắn khách |
| `handoff` | Cần chuyển nhân viên; đã bật thì giữ cho cả hội thoại | Tin nhắn khách |

**Bước bán hàng** (`decide_stage`, xét theo thứ tự ưu tiên):

| Bước | Khi nào | Hướng dẫn đưa vào prompt |
|---|---|---|
| `handoff` | Khách đòi gặp người thật, khiếu nại, giao sai/chậm/thiếu, hủy đơn, hoặc 2 lượt liên tiếp bực bội | Xin lỗi, nêu bước cần làm, báo đã chuyển nhân viên, không bán thêm |
| `closing` | Giỏ có hàng, hoặc khách nói chốt/lấy/đặt hàng sau khi đã xem máy | Xác nhận giỏ và tổng tiền, hỏi cách nhận hàng và thanh toán |
| `comparing` | Đã xem từ 2 máy và khách hỏi so sánh, "nên chọn cái nào" | So sánh theo nhu cầu đã biết, đưa 1 gợi ý rõ ràng |
| `recommending` | Đã giới thiệu ít nhất 1 máy | Trả lời về máy đó; khách ưng thì hỏi màu/dung lượng, mời thêm vào giỏ |
| `discovery` | Chưa giới thiệu máy nào | Thiếu ngân sách hoặc nhu cầu thì hỏi tối đa 1 câu |

Mỗi lần gọi model, `describe(state)` thêm vào cuối system prompt: bước hiện tại kèm hướng dẫn ở bảng trên, nhu cầu đã biết, máy đã giới thiệu, giỏ hàng và tổng tiền, lo ngại của khách, và nhắc trả lời ngắn nếu khách đang không vui.

**Cập nhật bằng luật, không gọi thêm LLM:** dựa trên từ khóa tiếng Việt và kết quả tool, nên không tốn quota, chạy tức thì và test được.

- Từ khóa so trên chữ **có dấu** và có ranh giới từ, để tránh khớp nhầm: "đắt" không trùng "đặt"; "ông" không khớp trong "kh**ông**"; "ip" không khớp trong "ch**ip**"; "tức là" không bị tính là tức giận; "không thích" không bị tính là khen.
- Ngân sách đọc được các dạng "15 triệu", "15tr", "20 củ", "7,5 triệu", "từ 10 đến 15 triệu", "12.000.000".

## Các tool

| Tool | Nguồn | Dùng khi |
|---|---|---|
| `search_products(query='', brand='', max_price=0, category='phone')` | Postgres | Có bán máy nào, máy trong tầm giá, so sánh giá các dòng |
| `get_product_details(product, storage='', color='')` | Postgres | Giá và tồn kho từng phiên bản, RAM, thông số chính của 1 máy |
| `find_accessories(phone, accessory_type='', page=1)` | Postgres (`compatibility`) | Phụ kiện tương thích với 1 máy, mỗi trang 5 món (còn hàng trước, rẻ trước) |
| `add_to_cart(product, storage='', color='', quantity=1)` | Postgres + state | Khách đồng ý mua (chỉ `graph.py`) |
| `remove_from_cart(product)` | State | Khách đổi ý (chỉ `graph.py`) |
| `search_policy(query)` | Qdrant | Đổi trả, hoàn tiền, giao hàng, trả góp, bảo hành |
| `web_search(query)` | Tavily, 3 kết quả | Đánh giá, tin tức: thông tin không có trong dữ liệu nội bộ |

Ví dụ `get_product_details('ip 18 pro', storage='256gb')`:

```
iPhone 18 Pro (Apple)
iPhone 18 Pro. Một cỗ máy AI siêu mạnh. ...
Màn Hình: Màn hình Super Retina XDR ... OLED 6,3 inch ...
Chip: Chip A20 Pro CPU 6 lõi ...
Phiên bản (dung lượng | màu | giá | kho):
- 256GB | Bạc | 38.999.000đ | còn hàng
- 256GB | Đen | 38.999.000đ | còn hàng
...
```

**Nhận diện tên máy** (`_match`, `_tokens` trong `tools.py`):

- Chuẩn hóa: chữ thường, bỏ dấu, `+` → `plus`, "17pro" → "17 pro"; đổi viết tắt (`ip` → `iphone`, `promax` → `pro max`…); bỏ "galaxy", "samsung", "5g", "điện thoại" và dung lượng.
- Mọi từ khách gõ phải có trong tên sản phẩm. Nếu có tên trùng khớp hoàn toàn thì chỉ lấy tên đó: "iPhone 18 Pro" không kéo theo "Pro Max". Riêng `search_products` lấy cả dòng: "s26" → S26 FE, S26, S26+, S26 Ultra.
- `get_product_details`, `find_accessories`, `add_to_cart` cần đúng 1 sản phẩm; nếu nhiều sản phẩm khớp thì trả danh sách để model hỏi lại khách.

**Giỏ hàng:** `add_to_cart` tìm đúng phiên bản theo dung lượng và màu (chưa rõ thì trả danh sách phiên bản để hỏi lại), kiểm tra tồn kho, gộp số lượng nếu trùng phiên bản, rồi trả `Command` cập nhật `cart` trong state. Tool đọc state qua `InjectedState`.

**Quy ước khác:**

- `accessory_type` nhận tiếng Việt ("ốp lưng", "cường lực", "sạc dự phòng"…).
- Tồn kho chỉ trả `còn hàng` / `sắp hết` (≤ 5 máy) / `hết hàng`, **không có số lượng**, vì khi có số thì model hay đọc ra cho khách ("còn 9 máy"). `add_to_cart` vẫn kiểm tra số lượng thật.
- Mọi tool tự bắt lỗi (Postgres, Qdrant, Tavily lỗi…) và báo cho model là lỗi hệ thống, để cả lượt chat không bị hỏng.

## RAG chính sách

Thư mục `data_schema/policies/` gồm `doi-tra.md`, `giao-hang.md`, `tra-gop.md`, `bao-hanh-apple.md`, `bao-hanh-samsung.md`. Chính sách của cửa hàng (lãi suất, phí, đối tác) là **giả lập**; phần bảo hành của hãng là tóm tắt tham khảo, cần đối chiếu trang chính thức.

### Nạp vào Qdrant (`data_prep.py`, `vector_db.py`)

- Mỗi mục `##`/`###` là một đoạn; riêng mục "Câu hỏi thường gặp" thì mỗi cặp hỏi/đáp là một đoạn.
- Đoạn dài quá 900 ký tự được cắt nhỏ, vì model embedding chỉ đọc khoảng 256 token đầu; bảng dài được cắt theo dòng và lặp lại dòng tiêu đề.
- Đầu mỗi đoạn có tên tài liệu và tên mục, vd `Chính sách đổi trả - 3. Đổi trả do lỗi nhà sản xuất > 3.2. Trong 30 ngày đầu`.
- Tách từ bằng `pyvi`, embedding 768 chiều, so khớp cosine; collection `phone_policies`.
- ID mỗi đoạn tính từ tên file, thứ tự và **nội dung** đoạn. Khi khởi động, web app so ID trong Qdrant với ID tính từ các file `.md`; nếu lệch thì xóa và nạp lại.

### Chống lạc đề, lan man (`search_policy`)

1. **Lọc đoạn:** lấy 3 đoạn gần nhất kèm điểm; bỏ đoạn có điểm dưới **0,25** và đoạn kém đoạn tốt nhất quá **0,08** (`config.py`). Câu hỏi lạc đề nhận "Không tìm thấy chính sách liên quan" thay vì đoạn không liên quan.
2. **Hướng dẫn chung** (`POLICY_GUIDE`): trả lời thẳng trước, tối đa 3 ý bổ sung, không chép cả bảng, giữ đúng con số, không có thì nói chưa có thông tin.
3. **Prompt riêng từng file** (`POLICY_PROMPTS`): các đoạn được gom theo file, mỗi nhóm kèm "Hướng dẫn trả lời" về phạm vi, ý bắt buộc và điều không được nói. Ví dụ:
   - `doi-tra.md`: xác định khách thuộc trường hợp nào (lỗi trong 30 ngày, đổi ý khi còn seal, đã kích hoạt…), chỉ trả lời trường hợp đó, chưa rõ thì hỏi lại 1 câu.
   - `tra-gop.md`: tiền góp hằng tháng chỉ là ước tính, không hứa hồ sơ được duyệt; hỏi điều kiện trả góp 0% thì nêu đủ cả thẻ tín dụng lẫn công ty tài chính.
   - `bao-hanh-*.md`: tách bảo hành của hãng với quyền lợi của cửa hàng; không dùng chính sách của hãng này cho máy hãng kia.

### Viết chính sách cho dễ truy xuất

Thông tin quan trọng nên có **mục riêng, viết thành câu**, không chỉ nằm trong một ô bảng. Ví dụ: điều kiện "trả trước ≥ 30%" của gói 0% ban đầu chỉ nằm trong một ô bảng ở mục 3 của `tra-gop.md`, nên câu "trả góp 0% cần điều kiện gì" không bao giờ lấy được đoạn đó và agent trả lời thiếu ý ở mọi cấu hình model. Sau khi thêm mục "Điều kiện để được trả góp lãi 0%", cả 5 cách hỏi khác nhau đều lấy được đoạn có "30%".

Khi thêm file chính sách mới, phải thêm prompt vào `POLICY_PROMPTS` (test sẽ báo nếu thiếu). Khi sửa nhiều chính sách, nên đo lại ngưỡng lọc; cách đo ghi trong chú thích ở `config.py`.

## LLM và chuỗi dự phòng

| LLM (`llm.py`) | Model | Vai trò |
|---|---|---|
| `open_llm` | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` | Model chính |
| `nvidia_llm` | NVIDIA API `nvidia/nemotron-3.5-super-vl-120b-a12b` | Dự phòng 1 (nhanh nhất trong các model dự phòng) |
| `google_llm` | `gemini-3.1-flash-lite` | Dự phòng 2; judge của eval |
| `google_llm_alt` | `gemini-3.5-flash-lite` | Dự phòng 3 (Gemini free tính quota riêng cho từng model) |
| `groq_llm` | `qwen/qwen3.8-27b` | Dự phòng 4 (nhanh nhưng quota token/phút nhỏ nên để cuối) |

Thứ tự dự phòng được xếp theo tốc độ đo được (xem [Kết quả đo hiệu năng](#kết-quả-đo-hiệu-năng)).

### Cách chuỗi dự phòng hoạt động (`fallback.py`)

`FallbackMiddleware` dùng cho cả `graph.py` lẫn `agent.py`:

1. Thử model chính; nếu lỗi thì thử lần lượt các model trong `FALLBACK_LLMS` tới khi có model trả lời. Nếu tất cả đều lỗi, API trả 429 (quota) hoặc 502.
2. Model gặp lỗi lặp lại sẽ bị **tạm bỏ qua**, để các câu sau không mất thêm một lần gọi hỏng:

   | Lỗi | Bỏ qua trong |
   |---|---|
   | Hết quota ngày (`free-models-per-day`, `...PerDay`) | 1 giờ |
   | Giới hạn theo phút | 1 phút |
   | Lỗi cấu hình (`is_permanent_error`: sai tên model, sai/hết hạn API key, không có quyền, hết credit); log mức ERROR | 1 giờ |
   | Lỗi chứng chỉ SSL (`CERTIFICATE_VERIFY_FAILED`) | 10 phút |
   | Lỗi tạm thời (mất mạng, 5xx, quá tải) | Không bỏ qua; chỉ chuyển model cho lần gọi đó |

3. SDK của model đứng trước chỉ tự thử lại 1 lần (`RETRIES_IN_CHAIN`) rồi nhường model sau; model cuối (Groq) thử lại 3 lần (`RETRIES_LAST`).
4. Nếu mọi model đều đang bị bỏ qua thì vẫn thử hết, thay vì báo lỗi ngay.
5. Mỗi lần chuyển model đều ghi log: `docker compose logs api | grep LLM`, hoặc xem ở terminal của `uvicorn` khi chạy trên máy.

Eval tắt chuỗi dự phòng (`fallback=False`) để điểm phản ánh đúng model chính. Để thêm hoặc bớt model, sửa `FALLBACK_LLMS`; model mới phải hỗ trợ tool calling (các model `gemini-2.5-*` trả 404 với key hiện tại).

### Cấu hình NVIDIA (`ChatNVIDIA`)

- **Bật thinking** (`enable_thinking`) và `max_completion_tokens=4096`: lựa chọn đúng nhất mà vẫn nhanh (xem [Kết quả đo hiệu năng](#kết-quả-đo-hiệu-năng)).
- Phần suy nghĩ được trả riêng trong `additional_kwargs['reasoning_content']`, không lẫn vào câu trả lời và không bao giờ được gửi cho khách.
- Không có `max_retries` (SDK không tự thử lại); client bên trong tự đặt timeout 60 giây.
- Lỗi là `Exception` chung dạng `[429] Too Many Requests…`; `nvidia_status()` đọc mã HTTP ở đầu thông báo: 429 tính là hết quota, 401/402/403/404 là lỗi cấu hình.

### Giới hạn gói free

- **OpenRouter** (model `:free`): khoảng 20 request/phút; 50 request/ngày nếu tài khoản chưa nạp 10 USD (đã nạp thì 1000/ngày). Quota reset lúc 00:00 UTC, tức 7:00 sáng giờ Việt Nam. Mỗi câu hỏi tốn 2–3 request.
- **Gemini:** giới hạn theo từng model (vd `gemini-3.1-flash-lite` 15 request/phút).
- **Groq:** 1000 token đầu ra/phút, 200.000 token/ngày; đặt `max_tokens=512`.

> **Chú ý:** `ChatOpenRouter` tính `timeout` bằng **mili giây** (`timeout=60_000` là 60 giây), khác Groq và Gemini tính bằng giây. Đặt `timeout=60` thì mọi request bị hủy sau 60 ms và agent treo vài phút. `tests/test_llm.py` kiểm tra lỗi này.

## Dữ liệu

Database `phone` (schema `data_schema/phone.sql`, dữ liệu `data_schema/output/phone_seed.sql`):

| Bảng | Số dòng | Nội dung |
|---|---|---|
| `products` | 23 điện thoại + 134 phụ kiện | `product_id` (vd `apple-iphone-18-pro`), `name`, `brand`, `category` (`phone`/`accessory`), `description`, `attributes` (JSON: thông số, RAM theo dung lượng, loại phụ kiện, link nguồn, SKU có giá ước tính) |
| `variants` | 377 | `variant_id` (SKU thật, vd `MJY04X/A`, `SM-S948BZDBXXV`), `product_id`, `color`, `storage`, `price` (VND), `stock` |
| `compatibility` | 738 | Cặp (`accessory_id`, `phone_id`) |

**Lưu ý về dữ liệu demo:**

- `stock` là **số giả lập** (5–50, cố định theo SKU); SKU hết hàng trên web hãng thì `stock = 0`.
- **Một số giá Samsung là ước tính:** web Samsung chỉ nhúng giá cho một số SKU. SKU thiếu giá lấy theo SKU cùng dung lượng, hoặc giá bản thấp nhất cộng chênh lệch dung lượng. Tool ghi "(giá tham khảo)" cho các SKU này.
- Máy có nhiều mức RAM cho cùng dung lượng thì `storage` ghi kèm RAM, vd `256GB - RAM 12GB` (Galaxy A57).
- Thông số chi tiết có cho iPhone và Galaxy A/FE; dòng Galaxy S/Z chưa có (web Samsung tải bảng thông số bằng JavaScript).
- Tương thích phụ kiện: ốp lưng và miếng dán theo tên máy ghi trong tên phụ kiện; cáp và củ sạc USB-C tương thích mọi máy; phụ kiện MagSafe chỉ cho iPhone; sạc không dây và phụ kiện nam châm Samsung chỉ cho dòng S và Z.

**Cập nhật dữ liệu:**

| Dữ liệu | Cách làm |
|---|---|
| Sản phẩm | `cd data_schema && python scrape_phone.py --refresh`, rồi `docker compose run --rm db-seed` |
| Chính sách | Sửa file trong `data_schema/policies/`, rồi `docker compose up -d --build api` (API tự nạp lại). Khi chạy trên máy: `uv run python -m phone_sales.vector_db` |

`scrape_phone.py` chỉ dùng thư viện chuẩn, đọc dữ liệu nhúng sẵn trong các trang HTML công khai, lưu cache trang đã tải ở `data_schema/.cache/` (không commit), và nghỉ 0,8 giây giữa các request. Sau khi cào lại, chạy `uv run pytest` và eval nhóm `product,accessory`, vì giá trong đáp án eval có thể đã đổi.

## Cấu trúc thư mục

```
sales_agent/
├── data_schema/
│   ├── phone.sql               # Schema database phone
│   ├── scrape_phone.py         # Cào Apple/Samsung VN -> output/
│   ├── output/                 # products.json, variants.json, compatibility.json, phone_seed.sql
│   └── policies/               # 5 file chính sách .md (nạp vào Qdrant)
├── src/phone_sales/
│   ├── __init__.py             # Fix DLL torch (Windows), dùng kho chứng chỉ hệ thống, nạp .env
│   ├── config.py               # Đường dẫn, DATABASE_URL, Qdrant, ngưỡng RAG, AGENT_IMPL
│   ├── llm.py                  # LLM chính, FALLBACK_LLMS, nhận diện lỗi quota/cấu hình
│   ├── fallback.py             # FallbackMiddleware: chuỗi LLM dự phòng
│   ├── prompts.py              # System prompt, POLICY_GUIDE, POLICY_PROMPTS
│   ├── tools.py                # 7 tool + gán tool cho từng agent
│   ├── conversation.py         # AgentState, luật cập nhật trạng thái, tóm tắt đưa vào prompt
│   ├── graph.py                # Agent StateGraph có trạng thái + giỏ hàng (web app dùng)
│   ├── agent.py                # Agent create_agent (không có trạng thái riêng)
│   ├── history.py              # Cắt bớt lịch sử trước khi gửi LLM
│   ├── data_prep.py            # Tách chính sách thành đoạn
│   ├── vector_db.py            # Embedding + Qdrant, nạp chính sách
│   ├── api.py                  # FastAPI: trang chat, POST /chat, POST /chat/stream, GET /health
│   └── static/index.html       # Giao diện chat
├── tests/                      # 118 test
├── evals/                      # dataset.jsonl (37 câu), run_eval.py, results/ (không commit)
├── certs/                      # CA nội bộ (*.crt) cho mạng có SSL inspection, không commit
├── .env.example                # Mẫu .env
├── Dockerfile, docker-compose.yml
├── langgraph.json              # Graph cho `langgraph dev`: agent_main, phone_graph
├── pyproject.toml, uv.lock     # Thư viện (nguồn chính)
└── requirements.txt            # Cho pip, phải khớp pyproject.toml (có test kiểm tra)
```

## Cài đặt

Cần Python 3.12.15 (ghi trong `.python-version`), [uv](https://docs.astral.sh/uv/) và Docker.

```bash
uv sync                  # tạo .venv, cài thư viện và nhóm dev
cp .env.example .env     # rồi điền key
```

Nếu dùng pip: `pip install -r requirements.txt` (đã có index torch bản CPU). Thêm thư viện bằng `uv add …`, rồi cập nhật `requirements.txt` cho khớp; `tests/test_requirements.py` sẽ báo nếu hai file lệch nhau.

| Biến trong `.env` | Mặc định | Ý nghĩa |
|---|---|---|
| `OPENROUTER_API_KEY` | (bắt buộc) | LLM chính ([openrouter.ai/keys](https://openrouter.ai/keys)) |
| `NVIDIA_API_KEY` | (bắt buộc) | Dự phòng 1 ([build.nvidia.com](https://build.nvidia.com)) |
| `GOOGLE_API_KEY` | (bắt buộc) | Gemini: dự phòng và judge của eval ([Google AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | (bắt buộc) | Groq: dự phòng ([console.groq.com](https://console.groq.com)) |
| `TAVILY_API_KEY` | (bắt buộc) | Tool `web_search` ([tavily.com](https://tavily.com)) |
| `POSTGRES_PASSWORD` | `postgres` | Mật khẩu container `phone_db`; chỉ có hiệu lực khi tạo volume lần đầu |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/phone` | Database khi chạy code trên máy (trong Docker, compose tự đặt) |
| `QDRANT_URL` | `http://localhost:6333` | Trong Docker, compose ghi đè thành `http://qdrant:6333` |
| `AGENT_IMPL` | `graph` | Web app dùng `graph.py` (có trạng thái + giỏ hàng) hay `agent.py` |
| `AUTO_INGEST` | `true` | Web app tự nạp chính sách khi Qdrant rỗng hoặc đã cũ |
| `LANGSMITH_*` | Tắt | Trace để debug; test và eval luôn tắt |

`.env` được nạp một lần trong `src/phone_sales/__init__.py`; biến đã có sẵn trong môi trường thì không bị ghi đè. Trên Windows, xem thêm [Lưu ý trên Windows](#lưu-ý-trên-windows).

## Chạy bằng Docker

```bash
docker compose up -d --build
```

| Container | Cổng | Vai trò |
|---|---|---|
| `phone_db` | 5433 | Postgres 18.6, database `phone`, volume `postgres_data`. Dùng cổng 5433 vì 5432 thường đã có Postgres cài trên máy |
| `qdrant` | 6333 (REST + dashboard), 6334 (gRPC) | Vector DB v1.20.0, volume `qdrant_data` |
| `phone_sales_api` | 8000 | Web app (image `phone-sales-api`) |

Khi khởi động:

- **Postgres:** lần đầu (volume rỗng) tự chạy `phone.sql` rồi `phone_seed.sql`. Các lần sau không chạy lại; dùng `db-seed` để nạp dữ liệu mới.
- **API:** chờ Postgres healthy, chờ Qdrant (tối đa khoảng 60 giây), rồi tự nạp chính sách nếu Qdrant rỗng hoặc khác các file `.md` trong image.
- **Image API** chứa `src/`, `data_schema/policies/`, thư viện và model embedding (tải sẵn lúc build); chạy bằng user thường; API key được truyền qua `env_file`.
- **Thời gian build:** lần đầu khoảng 8 phút, chủ yếu để tải model embedding (~540 MB). Model nằm ở layer trước `COPY README.md` và `COPY src`, nên sửa code, chính sách hay README chỉ build lại trong vài giây. Chỉ khi đổi `pyproject.toml`, `uv.lock` hoặc `certs/` mới phải tải lại model.

```bash
docker compose ps                                         # trạng thái
docker compose logs -f api                                # log
docker compose run --rm db-seed                           # nạp lại phone_seed.sql vào Postgres (không mất volume)
docker compose exec api python -m phone_sales.vector_db   # nạp lại chính sách ngay
docker compose up -d --build api                          # build lại API sau khi sửa code hoặc chính sách
docker compose down                                       # dừng, giữ dữ liệu
docker compose down -v                                    # dừng và xóa volume (lần sau tự nạp lại)
docker exec -it phone_db psql -U postgres -d phone        # mở psql
```

Sau khi chạy: trang chat http://localhost:8000, Swagger http://localhost:8000/docs, dashboard Qdrant http://localhost:6333/dashboard.

### Mạng có SSL inspection

Nếu log báo `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`, nghĩa là proxy của mạng đang cấp lại chứng chỉ HTTPS bằng CA nội bộ. Khi chạy trên máy thì không bị, vì `truststore` dùng kho chứng chỉ Windows. Trong container, các LLM dự phòng và LangSmith sẽ lỗi nếu thiếu CA này.

Cách sửa: xuất CA vào `certs/` rồi build lại (Dockerfile cài mọi file `certs/*.crt`):

```powershell
# PowerShell: đổi 'CN=ten-CA*' thành tên CA nội bộ (xem issuer của chứng chỉ bị lỗi)
$c = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -like 'CN=ten-CA*' } | Select-Object -First 1
"-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks') + "`n-----END CERTIFICATE-----" | Set-Content -Encoding ascii certs\internal-ca.crt
docker compose up -d --build api
```

Máy phát triển hiện tại đã có CA `tma-ADCA-CA` trong `certs/internal-ca.crt` (hết hạn 01/2029). File `.crt` không được commit, nên máy khác hoặc mạng khác phải làm lại bước trên.

## Chạy trên máy

Postgres và Qdrant vẫn chạy bằng Docker, còn code chạy trên máy:

```bash
docker compose up -d postgres qdrant
uv run python -m phone_sales.vector_db           # nạp chính sách (lần đầu tải model embedding ~540 MB)
uv run python -m phone_sales.graph               # hội thoại mẫu 3 lượt, in trạng thái sau mỗi lượt
uv run python -m phone_sales.agent               # 4 câu mẫu với agent.py
uv run uvicorn phone_sales.api:app --port 8000   # web app (thêm --reload khi sửa code)
uv run langgraph dev                             # LangGraph Studio: graph agent_main và phone_graph
```

Nếu container `phone_sales_api` đang chạy thì nó đã giữ cổng 8000: dừng nó bằng `docker compose stop api`, hoặc chạy `uvicorn` với `--port 8001`.

Dùng trong code:

```python
from phone_sales.graph import build_graph
from langgraph.checkpoint.memory import InMemorySaver

graph = build_graph(checkpointer=InMemorySaver())   # checkpointer giữ state theo thread_id
config = {'configurable': {'thread_id': 'khach-1'}}
graph.invoke({'messages': [{'role': 'user', 'content': 'tầm 30 triệu muốn máy chụp ảnh đẹp'}]}, config)
res = graph.invoke({'messages': [{'role': 'user', 'content': 'chốt galaxy s26 256gb màu đen'}]}, config)
print(res['messages'][-1].text, res['stage'], res['cart'])
```

## Web app và API

| Method | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/` | Trang chat (Enter để gửi, Shift+Enter để xuống dòng, nút "Cuộc trò chuyện mới") |
| `POST` | `/chat` | Gửi tin nhắn, nhận câu trả lời đầy đủ (JSON) |
| `POST` | `/chat/stream` | Như `/chat` nhưng trả Server-Sent Events; trang chat dùng endpoint này |
| `GET` | `/health` | `{"status": "ok", "documents": 66}`, hoặc 503 nếu không kết nối được Qdrant |
| `GET` | `/docs` | Swagger UI |

### `POST /chat`

```json
// request: bỏ trống thread_id để bắt đầu hội thoại mới, gửi lại thread_id ở các lượt sau
{"message": "ok chốt galaxy s26 256gb màu đen", "thread_id": "5f0c…"}

// response
{"thread_id": "5f0c…", "answer": "Dạ em đã thêm Galaxy S26 256GB Đen Classic vào giỏ…",
 "tools": ["add_to_cart"], "stage": "closing",
 "cart": [{"variant_id": "SM-S942BZKCXXV", "name": "Galaxy S26", "storage": "256GB", "color": "Đen Classic", "price": 29990001, "quantity": 1}],
 "handoff": false}
```

- `message` dài 1–1000 ký tự; `thread_id` tối đa 100 ký tự.
- `stage`, `cart`, `handoff` là trạng thái sau lượt này. Khi `AGENT_IMPL=agent` thì luôn là `null`, `[]`, `false`. Hệ thống khác có thể dựa vào `handoff` để báo nhân viên.
- Mã lỗi: 422 (input sai), 429 (mọi LLM đều vượt giới hạn), 502 (mọi LLM đều lỗi), 503 (`/health` không kết nối được Qdrant).

### `POST /chat/stream`

Request giống `/chat`. Response là `text/event-stream`, mỗi sự kiện là một dòng `data: {json}`:

| `type` | Khi nào | Trang chat làm gì |
|---|---|---|
| `start` | Ngay khi nhận request, kèm `thread_id` | Lưu `thread_id` |
| `status` | Model bắt đầu suy nghĩ ("Em đang cân nhắc…") hoặc gọi tool (vd "Đang xem giá và tồn kho…") | Đổi dòng trạng thái cạnh 3 chấm nhảy |
| `token` | Từng đoạn chữ LLM viết | Nối vào bong bóng (chữ thô) |
| `reset` | Model quyết định gọi tool sau khi đã viết vài chữ dẫn | Xóa phần chữ tạm |
| `done` | Xong lượt; nội dung giống response của `/chat` | Hiển thị lại câu trả lời có định dạng và dòng "Đã tra cứu" |
| `error` | Có lỗi; kèm `status` (429/502/500) và `detail` | Hiện lỗi trong bong bóng |

- Chỉ chữ do LLM viết ở node `model` mới được gửi đi; kết quả tool và **nội dung suy nghĩ** của model không bao giờ được gửi cho khách.
- Header `X-Accel-Buffering: no` giữ cho chữ không bị gom lại khi đặt API sau nginx.

### Lưu ý

- Trên Windows, curl của Git Bash có thể gửi sai mã hóa tiếng Việt (API trả 400); hãy dùng PowerShell `Invoke-RestMethod`, Swagger UI hoặc Python.
- State hội thoại nằm trong RAM (`InMemorySaver`) nên mất khi restart, và uvicorn chỉ chạy 1 worker. API chưa có xác thực hay giới hạn request.

## Test

```bash
uv run pytest                                # 118 test, dưới 1 phút
uv run pytest tests/test_conversation.py -v
```

Test **không gọi LLM, Tavily, mạng hay database**: `conftest.py` đặt key giả và tắt trace; tool DB dùng `_query` giả; agent và graph dùng model giả.

| File | Kiểm tra |
|---|---|
| `test_conversation.py` | Đọc ngân sách, hãng, nhu cầu, dung lượng; không khớp nhầm từ khóa ("không", "chip", "đặt", "tức là"); lo ngại, cảm xúc; khi nào chuyển nhân viên; chuyển bước; reducer; chỉ tính máy được nhắc trong câu trả lời; tool giỏ hàng (gộp phiên bản, tồn kho, hỏi lại phiên bản, xóa); graph giữ trạng thái qua 2 lượt |
| `test_tools.py` | Nhận diện tên máy, sắp xếp dung lượng, loại phụ kiện tiếng Việt, định dạng giá và tồn kho; 3 tool tra cứu DB; chia trang phụ kiện; `search_policy` (gom theo file kèm prompt, bỏ đoạn yếu, chặn câu lạc đề, mọi file đều có prompt); tool tự bắt lỗi |
| `test_agent.py` | Chuỗi dự phòng: thử hết chuỗi tới model trả lời được (cả `agent.py` lẫn `graph.py`); tạm bỏ qua model hết quota, lỗi cấu hình, lỗi SSL; lỗi tạm thời thì không bỏ qua; mọi model bị bỏ qua thì vẫn thử hết; `fallback=False` báo lỗi của model chính |
| `test_api.py` | Endpoint, validate input, mã lỗi 429/502/503; `/chat/stream` (thứ tự sự kiện, `reset` bỏ chữ dẫn trước tool, không stream kết quả tool hay nội dung suy nghĩ, lỗi giữa chừng thành sự kiện `error`); nhớ theo `thread_id`; tự nạp chính sách khi rỗng hoặc đã cũ; thử lại khi Qdrant chưa sẵn sàng |
| `test_llm.py` | `timeout` đúng đơn vị; 5 model trong chuỗi không trùng nhau và NVIDIA đứng đầu dự phòng; chỉ model cuối thử lại nhiều lần; nhận diện lỗi cấu hình; đọc mã lỗi của NVIDIA |
| `test_data_prep.py` | Tách chính sách theo mục và FAQ, cắt bảng dài, 5 file đều tách được |
| `test_history.py` | Cắt lịch sử đúng ranh giới lượt hỏi, rút gọn kết quả tool cũ |
| `test_eval_dataset.py` | `evals/dataset.jsonl` hợp lệ: tên trường, id không trùng, tool và file `policy` tồn tại, câu hỏi không chép nguyên văn dữ liệu |
| `test_requirements.py` | `requirements.txt` khớp `pyproject.toml` |

## Đánh giá (eval)

Eval gọi LLM, Postgres, Qdrant và Tavily thật trên một bộ câu hỏi cố định rồi chấm điểm. Nên chạy lại mỗi khi đổi model, prompt, tool hoặc dữ liệu.

```bash
docker compose up -d postgres qdrant
uv run python evals/run_eval.py --retrieval-only          # chỉ chấm truy xuất chính sách, không gọi LLM
uv run python evals/run_eval.py --only product,accessory  # lọc theo nhóm hoặc id
uv run python evals/run_eval.py --target graph            # chấm graph.py (mặc định chấm agent.py)
uv run python evals/run_eval.py                           # cả 37 câu
```

Eval tắt chuỗi dự phòng, nên chỉ chạy được khi OpenRouter còn quota; câu nào gặp hết quota ngày sẽ được ghi là lỗi và không tính vào thống kê. Cả 37 câu cần khoảng 90 request, vượt 50 request/ngày của gói free, nên phải chia làm 2 ngày.

| Nhóm | Số câu | Kiểm tra |
|---|---|---|
| `policy` | 19 | Gọi `search_policy`, đúng file chính sách, đúng thời hạn/phí |
| `product` | 7 | Đúng tool sản phẩm, đúng giá (vd iPhone 18 Pro Max 256GB = 41.999.000đ) |
| `accessory` | 3 | Gọi `find_accessories`, đúng sản phẩm và giá |
| `escalation` | 3 | Giao thiếu, giao chậm, đòi gặp người thật → báo chuyển nhân viên |
| `security` | 2 | Không nhận số thẻ/CVV, không tiết lộ thông tin khách khác |
| `web` | 1 | Thông tin ngoài dữ liệu nội bộ |
| `out_of_scope` | 2 | Từ chối ngắn gọn, không gọi tool |

Các chỉ số:

- **hit@1, hit@3, MRR:** truy xuất chính sách.
- **routing:** tool đầu tiên agent gọi có đúng không.
- **keyword:** câu trả lời có đủ các từ trong `must_contain` không.
- **behavior:** chuyển nhân viên hoặc từ chối đúng lúc.
- **grounded** và **score 1–5:** do Gemini chấm.

Kết quả chi tiết lưu ở `evals/results/`.

### Kết quả hiện có

**Truy xuất chính sách** (19 câu, 30/10/2026): hit@1 89%, hit@3 100%, MRR 0,95. Hai câu có file đúng ở vị trí 2: `policy-04` (hoàn tiền khi trả bằng thẻ tín dụng, lấy `tra-gop.md` trước) và `policy-13` (bảo hành iPhone bao lâu, lấy `giao-hang.md` trước).

**Agent `graph.py`** (OpenRouter, đủ 37 câu chia 2 ngày):

| Nhóm | Số câu | routing | keyword | behavior | grounded | Điểm TB | Độ trễ TB |
|---|---|---|---|---|---|---|---|
| `policy` | 19 | 19/19 | 15/15 | – | 19/19 | 4,89 | 9,0 s |
| `product` | 7 | 6/6 | 6/6 | – | 7/7 | 4,57 | 26,9 s |
| `accessory` | 3 | 2/3 | 3/3 | – | 3/3 | 4,67 | 25,9 s |
| `escalation` | 3 | 1/2 | – | 3/3 | 3/3 | 4,67 | 25,0 s |
| `security` | 2 | 1/1 | – | 1/1 | 2/2 | 5,00 | 11,6 s |
| `web` | 1 | – | – | – | 1/1 | 5,00 | 18,8 s |
| `out_of_scope` | 2 | 2/2 | – | 2/2 | 2/2 | 5,00 | 14,4 s |
| **Tổng** | **37** | **31/33** | **24/24** | **6/6** | **37/37** | **4,81** | **15,8 s** |

Mỗi câu lấy lần chạy mới nhất: `policy`, `web`, `acc-01` chạy ngày 30/10 (`evals/results/20261030-151900-graph.json`); `policy-19` và `out_of_scope` chạy lại sau khi sửa prompt (`20261030-152815-graph.json`); các câu còn lại chạy ngày 29/10 (`20261029-*-graph.json`).

Lần chạy 30/10 phát hiện 1 lỗi, đã sửa:

| Lỗi | Cách sửa |
|---|---|
| `policy-19` ("màn hình Z Fold có vết gấp có phải lỗi không"): model trả lời theo hiểu biết riêng, không gọi `search_policy`, thiếu lưu ý không tự gỡ lớp bảo vệ màn hình | Quy tắc 4 và 10: hỏi lỗi máy, hiện tượng lạ, bảo hành cũng phải gọi `search_policy`. Chạy lại: gọi đúng tool, 5 điểm; 2 câu `out_of_scope` vẫn không gọi tool |

9/21 câu trong lần chạy 30/10 dài hơn 80 từ (prompt giới hạn khoảng 100 từ).

Bảng chi tiết lần chạy 29/10 (17 câu thuộc 5 nhóm; mỗi câu lấy lần chạy mới nhất trong 2 lần):

| Nhóm | Số câu | routing | keyword | behavior | grounded | Điểm TB | Độ trễ TB |
|---|---|---|---|---|---|---|---|
| `product` | 7 | 6/6 | 6/6 | – | 7/7 | 4,57 | 26,9 s |
| `accessory` | 3 | 2/3 | 3/3 | – | 3/3 | 4,00 | 27,7 s |
| `escalation` | 3 | 1/2 | – | 3/3 | 3/3 | 4,67 | 25,0 s |
| `security` | 2 | 1/1 | – | 1/1 | 2/2 | 5,00 | 11,6 s |
| `out_of_scope` | 2 | 2/2 | – | 2/2 | 2/2 | 5,00 | 5,7 s |
| **Tổng** | **17** | **12/14** | **9/9** | **6/6** | **17/17** | **4,59** | **22,4 s** |

Lần chạy đầu phát hiện 4 lỗi, đều đã sửa:

| Lỗi | Cách sửa |
|---|---|
| Model đọc số lượng kho cho khách | Tool không còn trả số lượng |
| Model mời khách "kiểm tra số lượng chính xác" | Sửa quy tắc 8 trong prompt |
| Model liệt kê 8 ốp lưng | `find_accessories` chia trang 5 món (chạy lại 30/10: `acc-01` từ 3 lên 5 điểm) |
| Judge hiểu sai quy tắc "giá tham khảo" và "tối đa 3 máy" | Sửa prompt chấm điểm |

Ghi chú:

- Hai câu routing chưa đạt:
  - `acc-02` gọi `search_products` trước `find_accessories` (thừa một lần gọi, kết quả vẫn đúng).
  - `esc-03` được chấm trước khi câu này đổi sang `tools_optional`.
- Độ trễ cao (trung bình 22 giây) vì model chính là model suy luận và thường gọi 2 tool mỗi câu.

## Kết quả đo hiệu năng

Các số dưới đây đo ngày 29/10/2026, mỗi cấu hình chạy 1 lần (riêng 2048 token chạy 2 lần), nên chênh lệch 1 câu có thể chỉ là dao động ngẫu nhiên.

### Tốc độ các model dự phòng

Cùng một câu hỏi có gọi tool:

| Model | Thời gian |
|---|---|
| NVIDIA | 0,9–2,3 s |
| Groq | < 2 s |
| Gemini 3.1 flash-lite | Dao động 1–23 s |
| Gemini 3.5 flash-lite | 15–20 s |

Độ trễ của Gemini đến từ phía dịch vụ: tắt hẳn thử lại (`max_retries=0`) hay đổi mức suy nghĩ đều không làm nhanh hơn rõ rệt.

### Thinking và giới hạn token của NVIDIA

24 câu eval có đáp án (`product`, `accessory`, `policy`), NVIDIA làm model chính:

| Cấu hình | Đúng tool | Đủ ý | Trung vị | Trung bình | Chậm nhất |
|---|---|---|---|---|---|
| Thinking bật, 8192 token | 24/24 | 23/24 | 5,9 s | 13,1 s | 59 s |
| **Thinking bật, 4096 token (đang dùng)** | **24/24** | **23/24** | **6,0 s** | **9,1 s** | **46 s** |
| Thinking bật, 2048 token (2 lần) | 24/24; 23/23 | 22/24; 22/23 | 5,1–7,6 s | 8,7–8,8 s | 30–42 s |
| Thinking tắt + quy tắc 10 | 23/23 | 21/23 | 3,1 s | 4,5 s | 19 s |
| Thinking tắt, chưa có quy tắc 10 | 21/23 | 19/23 | 3,3 s | 5,2 s | 20 s |

- **Không tắt thinking:** nhanh gấp đôi nhưng trả lời sai theo kiểu làm mất khách. Ví dụ tool đã trả về danh sách máy mà model vẫn nói "cửa hàng không có Samsung nào dưới 7 triệu". Quy tắc 10 giúp model luôn gọi tool, nhưng không sửa được lỗi đọc sai kết quả.
- **4096 token:** đúng ngang 8192, thời gian trung bình giảm từ 13,1 s xuống 9,1 s. Mức 2048 không nhanh hơn rõ rệt và có nguy cơ bị cắt cụt khi model cần suy nghĩ dài.

### Stream câu trả lời

Hội thoại 3 lượt qua `/chat/stream`, NVIDIA trả lời:

| Câu | Trạng thái đầu tiên | Chữ đầu tiên | Xong |
|---|---|---|---|
| "tầm 15 triệu có máy samsung nào pin trâu" | 2,4 s (cân nhắc) → 3,2 s (tìm sản phẩm) | 16,6 s | 17,4 s |
| "chào em" | 1,0 s | 1,8 s | 2,2 s |
| "ok chốt a57 128gb xanh đêm" | 0,9 s → 3,4 s (cập nhật giỏ) | 5,0 s | 5,6 s |

Khách thấy phản hồi trong khoảng 1–3 giây. Tổng thời gian không đổi, vì model suy nghĩ xong mới viết (và viết rất nhanh); lợi ích chính là khách biết hệ thống đang làm gì thay vì nhìn màn hình đứng yên.

### Chuỗi dự phòng

OpenRouter hết quota, Gemini 3.1 được thay bằng một model không tồn tại để ép lỗi:

| Lượt | Trước tối ưu | Sau tối ưu |
|---|---|---|
| Câu 1 | OpenRouter 1,6 s → model hỏng 0,8 s → Gemini 3.5 | Như cũ (lần đầu phải thử mới biết model hỏng) |
| Câu 2, 3 | Vẫn thử lại model hỏng ở mỗi câu | Bỏ qua cả OpenRouter lẫn model hỏng, gọi thẳng Gemini 3.5 |

### Lọc đoạn chính sách

Trên 19 câu chính sách: câu đúng chủ đề có điểm từ 0,30 trở lên, câu lạc đề (bitcoin, làm thơ, thời tiết, "gặp nhân viên") chỉ 0,10–0,21. Sau khi lọc, vẫn lấy đúng file 19/19, trung bình 2,2 đoạn và 1,3 file mỗi câu (trước khi lọc là 3 đoạn, thường lẫn 2–3 file).

## Hạn chế và việc còn mở

**Hạn chế:**

- State hội thoại và giỏ hàng nằm trong RAM, mất khi restart; uvicorn chỉ chạy 1 worker.
- Giỏ hàng chưa thành đơn hàng thật.
- Luật nhận diện từ khóa không hiểu mọi cách nói (mỉa mai, viết không dấu, viết tắt lạ).
- Thông số chi tiết của dòng Galaxy S/Z chưa có; tồn kho là số giả lập; một số giá Samsung là ước tính.
- Bộ eval ít câu, chưa có câu nhiều lượt; judge là LLM nên có thể chấm sai; LLM dao động giữa các lần chạy.
- API chưa có xác thực hay giới hạn request.

**Việc còn mở:**

- Truy xuất chính sách: `policy-04` và `policy-13` lấy đúng file ở vị trí 2 chứ chưa ở vị trí 1.
- Gần một nửa câu trả lời chính sách dài hơn 80 từ.

## Mở rộng dự án

**Thêm tool:**

1. Viết hàm `@tool` trong `tools.py`. Docstring chính là mô tả model đọc, nên ghi rõ dùng khi nào, không dùng khi nào. Tự bắt lỗi bên trong tool.
2. Thêm tool vào `TOOLS` và `AGENT_TOOLS` (`agent_main` và/hoặc `agent_graph`), rồi thêm quy tắc dùng tool trong `prompts.py`.
3. Thêm test và câu eval.

**Thêm trường trạng thái:** khai báo trong `AgentState` (kèm reducer nếu cần cộng dồn), cập nhật ở `track_user` / `track_tools` / `track_reply` hoặc bằng tool trả `Command`, rồi đưa vào `describe()` để model thấy.

**Ý tưởng tiếp theo:**

- Tool đặt hàng thật từ giỏ hàng.
- Checkpointer Postgres (`langgraph-checkpoint-postgres`) để state không mất khi restart và chạy được nhiều worker.
- Câu eval nhiều lượt.
- Cào thêm thông số dòng Galaxy S/Z.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| "Lỗi hệ thống khi tra cứu kho hàng" | Postgres chưa chạy hoặc sai `DATABASE_URL` | `docker compose up -d postgres`; khi chạy trên máy dùng cổng **5433** |
| Giá/sản phẩm chưa đổi sau khi cào lại | File khởi tạo Postgres chỉ chạy lần đầu | `docker compose run --rm db-seed` |
| `password authentication failed` sau khi đổi `POSTGRES_PASSWORD` | Mật khẩu chỉ được đặt lúc tạo volume | `docker compose down -v` rồi chạy lại |
| `search_policy` không tìm thấy gì | Chưa nạp chính sách | `uv run python -m phone_sales.vector_db` |
| Agent trả lời thiếu ý dù chính sách có ghi | Đoạn chứa thông tin không được truy xuất | Xem [Viết chính sách cho dễ truy xuất](#viết-chính-sách-cho-dễ-truy-xuất) |
| `/chat` trả 429 | Mọi model trong chuỗi đều vượt giới hạn | Chờ quota reset; xem `docker compose logs api \| grep LLM` |
| `/chat` trả 502 | Mọi model đều lỗi (sai key, mất mạng, SSL…) | Kiểm tra key trong `.env` và log |
| Trong Docker, model dự phòng báo `CERTIFICATE_VERIFY_FAILED` | Thiếu CA nội bộ trong `certs/` | Xem [Mạng có SSL inspection](#mạng-có-ssl-inspection) |
| Câu trả lời khác thường | Model dự phòng đang trả lời | `docker compose logs api \| grep "model dự phòng"` |
| `/health` trả 503 | Qdrant chưa chạy hoặc sai `QDRANT_URL` | `docker compose up -d qdrant` |
| Agent treo vài phút | `timeout` của OpenRouter bị đặt bằng giây | Đặt `timeout=60_000` trong `llm.py` |
| Log `Failed to multipart ingest runs` | LangSmith không gửi được trace | Đặt `LANGSMITH_TRACING=false`; không ảnh hưởng chức năng |
| `uvicorn` báo `[Errno 10048] ... 8000` | Container `phone_sales_api` đang giữ cổng 8000 | Dùng web app trong Docker; hoặc `docker compose stop api`; hoặc chạy `uvicorn` với `--port 8001` |
| Sửa code nhưng Docker không đổi | Image cũ | `docker compose up -d --build api` |

## Lưu ý trên Windows

- **`OSError: [WinError 1114] ... torch\lib\c10.dll`:** antivirus (vd McAfee) nạp `msvcp140.dll` bản cũ vào Python, trong khi `torch` cần bản mới hơn. Tạo file `.pth` để nạp DLL của Windows ngay khi Python khởi động (**chạy lại mỗi khi tạo lại `.venv`**):

  ```bash
  uv run python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[-1], '_msvcp140_preload.pth').write_text(\"import sys, ctypes; sys.platform == 'win32' and ctypes.WinDLL('msvcp140.dll')\n\")"
  ```

  `src/phone_sales/__init__.py` cũng nạp trước DLL này, nhưng không đủ cho `langgraph dev`.
- **`UnicodeEncodeError`** khi in tiếng Việt ra pipe hoặc file: đặt `PYTHONUTF8=1`.
- **Git Bash tự đổi đường dẫn** trong lệnh `docker run ... /app/...`: thêm `MSYS_NO_PATHCONV=1` trước lệnh.
