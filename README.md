# phone-sales-agent

Agent tư vấn bán điện thoại bằng tiếng Việt cho một cửa hàng bán lẻ (demo, phục vụ học tập), xây dựng với LangChain/LangGraph.

Agent làm được:
- Tư vấn iPhone, Samsung Galaxy theo ngân sách và nhu cầu; báo giá, màu, dung lượng, tình trạng hàng của từng phiên bản.
- Tìm phụ kiện tương thích (ốp lưng, cường lực, sạc, cáp, S Pen…).
- Trả lời chính sách đổi trả, giao hàng, trả góp, bảo hành (RAG).
- Nhớ trạng thái hội thoại bán hàng (nhu cầu, máy đã giới thiệu, lo ngại, cảm xúc của khách) và quản lý giỏ hàng.
- Tự chuyển sang LLM dự phòng khi model chính hết quota.

| Thành phần | Công nghệ |
|---|---|
| Dữ liệu sản phẩm: 23 điện thoại, 134 phụ kiện, 377 phiên bản (cào từ apple.com/vn, samsung.com/vn) | Postgres 18 (database `phone`) |
| Chính sách: 5 file `.md`, tách thành 66 đoạn | Qdrant + embedding `bkai-foundation-models/vietnamese-bi-encoder` |
| Agent | LangGraph `StateGraph` (`graph.py`), hoặc `create_agent` (`agent.py`) |
| LLM | OpenRouter `nemotron-3-ultra-550b-a55b:free` → NVIDIA `nemotron-3.5-super-vl-120b-a12b` → Gemini 3.1 flash-lite → Gemini 3.5 flash-lite → Groq `qwen3.8-27b` |
| Web | FastAPI + trang chat một file, chạy bằng Docker Compose |
| Kiểm thử | 118 test (không gọi API thật), bộ eval 37 câu |

**Chạy nhanh** (cần Docker và file `.env`, xem [Cài đặt](#cài-đặt)):

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
- [Chuỗi LLM dự phòng](#chuỗi-llm-dự-phòng)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [Chạy trên máy](#chạy-trên-máy)
- [Web app và API](#web-app-và-api)
- [Test](#test)
- [Đánh giá (eval)](#đánh-giá-eval)
- [Mở rộng dự án](#mở-rộng-dự-án)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Lưu ý trên Windows](#lưu-ý-trên-windows)

## Kiến trúc

```
Trình duyệt (trang chat)                          LangGraph Studio (langgraph dev)
        │ POST /chat                                         │
        ▼                                                    ▼
FastAPI (api.py) ── lưu state theo thread_id ──> graph.py (mặc định) hoặc agent.py
                                                             │
  LLM: OpenRouter ─lỗi/hết quota─> NVIDIA ──> Gemini 3.1 ──> Gemini 3.5 ──> Groq   (fallback.py)
                                                             │ gọi tool
   ┌───────────────┬─────────────────┬───────────────┬───────┴───────┬──────────────┬────────────────┐
   ▼               ▼                 ▼               ▼               ▼              ▼                ▼
search_products get_product_details find_accessories add_to_cart  remove_from_cart search_policy  web_search
   └──────────── Postgres: products, variants, compatibility ─────────────┘   Qdrant (RAG)    Tavily
```

**Luồng một lượt chat** (`graph.py`):

```
START → track_user → model ─(có tool_call)→ tools → track_tools → model … ─(trả lời xong)→ track_reply → END
```

1. `api.py` nhận `message` + `thread_id`, lấy state cũ của thread từ checkpointer (`InMemorySaver`).
2. `track_user` đọc tin nhắn của khách: nhu cầu, lo ngại, cảm xúc, có đòi gặp nhân viên không, rồi tính bước bán hàng.
3. `model` gọi LLM với system prompt + **tóm tắt trạng thái hội thoại** + lịch sử đã cắt gọn (`history.py`: 4 lượt gần nhất, kết quả tool cũ rút còn 300 ký tự).
4. LLM gọi tool nếu cần; `track_tools` đọc tham số tool (ngân sách, hãng, dung lượng, màu) và tính lại bước (vd vừa thêm vào giỏ → `closing`).
5. Khi LLM trả lời xong, `track_reply` ghi lại các sản phẩm thật sự được giới thiệu trong câu trả lời.
6. API trả `answer`, `tools` đã gọi, cùng `stage`, `cart`, `handoff`. Trang chat dùng `/chat/stream`: hiện trạng thái và chữ ngay trong lúc agent đang chạy (xem [Web app và API](#web-app-và-api)).

**System prompt** (`prompts.py`): `AGENT_PROMPTS['agent_main']` gồm 15 quy tắc; `graph.py` dùng `agent_graph` = 15 quy tắc đó + 3 quy tắc về trạng thái và giỏ hàng.

| Nhóm | Quy tắc chính |
|---|---|
| Xưng hô | Tiếng Việt, xưng "em", gọi "anh/chị", không đoán giới tính |
| Dùng tool | Tìm máy/tầm giá → `search_products`; giá, màu, tồn kho, thông số 1 máy → `get_product_details` (nhiều máy khớp thì hỏi lại); phụ kiện → `find_accessories`; chính sách → `search_policy` (làm theo "Hướng dẫn trả lời" kèm kết quả); `web_search` chỉ khi dữ liệu nội bộ không có, không dùng cho giá hay chính sách |
| Trả lời | Chỉ dựa trên kết quả tool; **không bao giờ nói không có hàng / chưa có thông tin khi chưa gọi tool trong lượt đó** (quy tắc 10); giữ đúng giá, chỉ dòng có "(giá tham khảo)" mới nói là tham khảo; tồn kho chỉ nói còn hàng / sắp hết / hết hàng; tư vấn tối đa 3 máy; tối đa khoảng 100 từ |
| Chuyển nhân viên | Khiếu nại, máy lỗi, đơn giao sai/chậm/thiếu, hủy đơn đã thanh toán, đòi gặp người thật |
| Bảo mật | Không hỏi OTP, mật khẩu, PIN, số thẻ, CVV; không lộ thông tin khách khác |
| Ngoài phạm vi | Không gọi tool, từ chối 1–2 câu, mời tư vấn sản phẩm |
| Trạng thái, giỏ hàng (chỉ `graph.py`) | Làm theo hướng dẫn của bước hiện tại, không hỏi lại nhu cầu đã biết; chỉ `add_to_cart` khi khách nói mua/chốt; giỏ hàng chưa phải đơn hàng, nhân viên sẽ liên hệ xác nhận |

**Hai cách dựng agent** dùng chung LLM, tool tra cứu, prompt, cách cắt lịch sử và chuỗi dự phòng:

| | `graph.py` — web app dùng mặc định | `agent.py` |
|---|---|---|
| Dựng bằng | `StateGraph` tự dựng | `create_agent` của LangChain |
| Trạng thái hội thoại, giỏ hàng | Có | Không (chỉ `messages`) |
| Dùng khi | Web app, LangGraph Studio (`phone_graph`) | So sánh, eval, LangGraph Studio (`agent_main`) |

Đổi web app sang `agent.py`: đặt `AGENT_IMPL=agent` trong `.env`.

## Trạng thái hội thoại

`AgentState` (`src/phone_sales/conversation.py`) được checkpointer lưu theo `thread_id`, nên giữ qua các lượt:

| Trường | Ý nghĩa | Cập nhật |
|---|---|---|
| `messages` | Lịch sử tin nhắn (reducer `add_messages`, nối thêm) | Mọi node |
| `stage` | `discovery` → `recommending` → `comparing` → `closing`, hoặc `handoff` | Tính lại sau mỗi node |
| `slots` | Nhu cầu: `budget_max`, `brand`, `purposes`, `storage`, `color` (reducer gộp; list cộng dồn) | Tin nhắn khách + tham số tool |
| `shown_products` | Máy đã giới thiệu: có trong kết quả tool của lượt **và** được nhắc trong câu trả lời | `track_reply` |
| `cart` | Giỏ hàng: `variant_id`, `name`, `storage`, `color`, `price`, `quantity` | `add_to_cart`, `remove_from_cart` |
| `objections` | Lo ngại: chê giá cao, lo pin yếu, máy nóng, nặng/to, chê camera, lo lỗi, không thích màu | Tin nhắn khách |
| `sentiment_history` | `positive` / `neutral` / `negative` từng lượt | Tin nhắn khách |
| `handoff` | Cần chuyển nhân viên. Đã bật thì giữ cho cả hội thoại | Tin nhắn khách |

**Cách xác định bước bán hàng** (`decide_stage`, theo thứ tự ưu tiên):

| Bước | Khi nào | Hướng dẫn đưa vào prompt |
|---|---|---|
| `handoff` | Khách đòi gặp người thật, khiếu nại, giao sai/chậm/thiếu, hủy đơn, hoặc **2 lượt liên tiếp bực bội** | Xin lỗi, nêu bước cần làm, báo đã chuyển nhân viên, không bán thêm |
| `closing` | Giỏ có hàng, hoặc khách nói chốt/lấy/đặt hàng sau khi đã xem máy | Xác nhận giỏ và tổng tiền, hỏi cách nhận hàng và thanh toán |
| `comparing` | Đã xem ≥ 2 máy và khách hỏi so sánh / "nên chọn cái nào" | So sánh theo nhu cầu đã biết, đưa 1 gợi ý rõ ràng |
| `recommending` | Đã giới thiệu ít nhất 1 máy | Trả lời về máy đó; khách ưng thì hỏi màu/dung lượng, mời thêm vào giỏ |
| `discovery` | Chưa giới thiệu máy nào | Thiếu ngân sách hoặc nhu cầu thì hỏi tối đa 1 câu |

Mỗi lần gọi model, `describe(state)` thêm vào cuối system prompt: bước hiện tại kèm hướng dẫn ở bảng trên, nhu cầu đã biết, máy đã giới thiệu, giỏ hàng và tổng tiền, lo ngại của khách (dặn trả lời thẳng vào lo ngại), và nhắc trả lời ngắn nếu khách đang không vui.

**Cập nhật bằng luật, không gọi thêm LLM:** từ khóa tiếng Việt và kết quả tool. Nhờ vậy không tốn quota, chạy tức thì và test được. Từ khóa được so trên chữ **có dấu**, có ranh giới từ, để tránh khớp nhầm:
- "đắt" không trùng "đặt";
- "ông" không khớp trong "kh**ông**", "ip" không khớp trong "ch**ip**";
- "tức là" không bị tính là tức giận, "không thích" không bị tính là khen.

Ngân sách đọc được các dạng "15 triệu", "15tr", "20 củ", "7,5 triệu", "từ 10 đến 15 triệu", "12.000.000".

**Hạn chế:** luật từ khóa không hiểu mọi cách nói (mỉa mai, viết không dấu, viết tắt lạ); giỏ hàng chưa thành đơn hàng thật; state nằm trong RAM nên mất khi restart.

## Các tool

| Tool | Nguồn | Dùng khi |
|---|---|---|
| `search_products(query='', brand='', max_price=0, category='phone')` | Postgres | Có bán máy nào, máy trong tầm giá, so sánh giá các dòng |
| `get_product_details(product, storage='', color='')` | Postgres | Giá và tồn kho từng phiên bản, RAM, thông số chính của 1 máy |
| `find_accessories(phone, accessory_type='', page=1)` | Postgres (`compatibility`) | Phụ kiện tương thích với 1 máy, mỗi trang 5 món (còn hàng trước, rẻ trước) |
| `add_to_cart(product, storage='', color='', quantity=1)` | Postgres + state | Khách đồng ý mua (chỉ `graph.py`) |
| `remove_from_cart(product)` | state | Khách đổi ý (chỉ `graph.py`) |
| `search_policy(query)` | Qdrant | Đổi trả, hoàn tiền, giao hàng, trả góp, bảo hành |
| `web_search(query)` | Tavily, 3 kết quả | Đánh giá, tin tức – thông tin không có trong dữ liệu nội bộ |

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
- Mọi từ khách gõ phải có trong tên sản phẩm. Có tên trùng khớp hoàn toàn thì chỉ lấy tên đó: "iPhone 18 Pro" không kéo theo "Pro Max". Riêng `search_products` lấy cả dòng: "s26" → S26 FE, S26, S26+, S26 Ultra.
- `get_product_details`, `find_accessories`, `add_to_cart` cần đúng 1 sản phẩm; nhiều sản phẩm khớp thì trả danh sách để model hỏi lại khách.

**Giỏ hàng:** `add_to_cart` tìm đúng phiên bản theo dung lượng/màu (chưa rõ thì trả danh sách phiên bản để hỏi lại), kiểm tra tồn kho, gộp số lượng nếu cùng phiên bản, rồi trả `Command` cập nhật `cart` trong state. Tool đọc state qua `InjectedState`.

**Khác:** `accessory_type` nhận tiếng Việt ("ốp lưng", "cường lực", "sạc dự phòng"…). Tồn kho chỉ trả `còn hàng` / `sắp hết` (≤ 5 máy) / `hết hàng`, **không có số lượng**: eval cho thấy có số là model đọc ra cho khách ("còn 9 máy"); `add_to_cart` vẫn kiểm tra số lượng thật. Mọi tool tự bắt lỗi (Postgres, Qdrant, Tavily sập…) và báo cho model là lỗi hệ thống, để cả lượt chat không bị hỏng.

## RAG chính sách

`data_schema/policies/`: `doi-tra.md`, `giao-hang.md`, `tra-gop.md`, `bao-hanh-apple.md`, `bao-hanh-samsung.md`. Chính sách của cửa hàng (lãi suất, phí, đối tác) là **giả lập**; phần bảo hành của hãng là tóm tắt tham khảo, cần đối chiếu trang chính thức.

**Nạp vào Qdrant** (`data_prep.py`, `vector_db.py`):
- Mỗi mục `##`/`###` là một đoạn; mục "Câu hỏi thường gặp" thì mỗi cặp hỏi/đáp là một đoạn.
- Đoạn dài quá 900 ký tự được cắt nhỏ (model embedding chỉ đọc khoảng 256 token đầu); bảng dài cắt theo dòng, lặp lại tiêu đề bảng.
- Đầu mỗi đoạn gắn tên tài liệu và mục, vd `Chính sách đổi trả - 3. Đổi trả do lỗi nhà sản xuất > 3.2. Trong 30 ngày đầu`.
- Tách từ bằng `pyvi`, embedding 768 chiều, so cosine. Collection `phone_policies`.
- ID đoạn tính từ tên file, thứ tự và **nội dung** đoạn. Khi khởi động, web app so ID trong Qdrant với ID tính từ file `.md`; lệch thì tự xóa và nạp lại.

**Chống lạc đề, lan man** (`search_policy`):
1. **Lọc đoạn:** lấy 3 đoạn gần nhất kèm điểm; bỏ đoạn dưới **0,25** và đoạn kém đoạn tốt nhất quá **0,08** (`config.py`). Ngưỡng đo trên 19 câu chính sách của bộ eval: câu đúng chủ đề có điểm từ 0,30, câu lạc đề (bitcoin, làm thơ, thời tiết, "gặp nhân viên") 0,10–0,21. Câu lạc đề nhận "Không tìm thấy chính sách liên quan" thay vì đoạn không liên quan.
2. **Hướng dẫn chung** (`POLICY_GUIDE`): trả lời thẳng trước, tối đa 3 ý bổ sung, không chép cả bảng, giữ đúng con số, không có thì nói chưa có thông tin.
3. **Prompt riêng từng file** (`POLICY_PROMPTS`): đoạn được gom theo file, mỗi nhóm kèm "Hướng dẫn trả lời" về phạm vi, ý bắt buộc và điều không được nói. Ví dụ:
   - `doi-tra.md`: xác định khách thuộc trường hợp nào (lỗi trong 30 ngày, đổi ý còn seal, đã kích hoạt…), chỉ trả lời trường hợp đó, chưa rõ thì hỏi lại 1 câu.
   - `tra-gop.md`: tiền góp hằng tháng chỉ là ước tính, không hứa hồ sơ được duyệt.
   - `bao-hanh-*.md`: tách bảo hành của hãng với quyền lợi của cửa hàng; không dùng chính sách hãng này cho máy hãng kia.

Kết quả trên 19 câu chính sách: vẫn ra đúng file 19/19, trung bình 2,2 đoạn / 1,3 file mỗi câu (trước khi lọc là 3 đoạn, thường lẫn 2–3 file).

Thêm file chính sách mới thì phải thêm prompt vào `POLICY_PROMPTS` (test báo nếu thiếu). Sửa nhiều chính sách thì nên đo lại ngưỡng (cách đo ghi trong chú thích ở `config.py`).

## Chuỗi LLM dự phòng

| LLM (`llm.py`) | Model | Vai trò |
|---|---|---|
| `open_llm` | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` | Model chính |
| `nvidia_llm` | NVIDIA API `nvidia/nemotron-3.5-super-vl-120b-a12b` (bật thinking) | Dự phòng 1: nhanh nhất trong các model dự phòng |
| `google_llm` | `gemini-3.1-flash-lite` | Dự phòng 2; judge của eval |
| `google_llm_alt` | `gemini-3.5-flash-lite` | Dự phòng 3 (Gemini free tính quota riêng từng model) |
| `groq_llm` | `qwen/qwen3.8-27b` | Dự phòng 4: nhanh nhưng quota token/phút nhỏ nên để cuối |

Thứ tự dự phòng xếp theo tốc độ đo ngày 29/10/2026 (cùng một câu có gọi tool): NVIDIA 0,9–2,3 s, Gemini 3.1 dao động 1–23 s, Gemini 3.5 15–20 s, Groq dưới 2 s.

**NVIDIA** (`ChatNVIDIA`) có vài điểm khác các SDK còn lại:
- Phần "thinking" được trả riêng trong `additional_kwargs['reasoning_content']`, không lẫn vào câu trả lời gửi khách.
- Không có `max_retries` (SDK không tự thử lại) và tự đặt timeout 60 s.
- **Giữ thinking bật, `max_completion_tokens=4096`** (đo 29/10/2026 trên 24 câu eval có đáp án, NVIDIA làm model chính):

  | Cấu hình | Đúng tool | Đủ ý | Trung vị | Trung bình | Chậm nhất |
  |---|---|---|---|---|---|
  | Thinking bật, 8192 | 24/24 | 23/24 | 5,9 s | 13,1 s | 59 s |
  | **Thinking bật, 4096 (đang dùng)** | 24/24 | 23/24 | 6,0 s | 9,1 s | 46 s |
  | Thinking bật, 2048 (2 lần) | 24/24, 23/23 | 22/24, 22/23 | 5,1–7,6 s | 8,7–8,8 s | 30–42 s |
  | Thinking tắt + quy tắc 10 | 23/23 | 21/23 | 3,1 s | 4,5 s | 19 s |

  - Tắt thinking nhanh gấp đôi nhưng trả lời sai kiểu làm mất khách: tool đã trả về danh sách máy mà model vẫn nói "cửa hàng không có Samsung nào dưới 7 triệu". Quy tắc 10 giúp model luôn gọi tool (trước khi có quy tắc: 21/23 đúng tool, 19/23 đủ ý) nhưng không sửa được lỗi đọc sai kết quả.
  - Giảm token cắt bớt các lần suy nghĩ quá dài (trung bình 13,1 → 9,1 s) mà không giảm độ đúng; 2048 không nhanh hơn 4096 rõ rệt nên chọn 4096 để không bị cắt cụt khi cần suy nghĩ dài.
  - `policy-10` ("trả góp 0% cần điều kiện gì") thiếu ý ở mọi cấu hình trong các lần đo này. Nguyên nhân là **truy xuất**, không phải model: điều kiện "trả trước ≥ 30%" chỉ nằm trong 1 ô bảng ở mục 3 của `tra-gop.md` nên không vào top 6 đoạn. Đã sửa bằng mục mới "Điều kiện để được trả góp lãi 0%" trong `tra-gop.md` (gom điều kiện của cả thẻ tín dụng và công ty tài chính) + prompt `tra-gop.md` dặn nêu đủ 2 hình thức. Sau khi sửa: 5/5 cách hỏi khác nhau đều lấy được đoạn có "30%", 3/3 câu trả lời của agent nêu đúng điều kiện; eval truy xuất 19 câu không đổi.
- Lỗi là `Exception` chung dạng `[429] Too Many Requests…`; `nvidia_status()` đọc mã HTTP ở đầu thông báo: 429 tính là hết quota, 401/402/403/404 là lỗi cấu hình.

`FallbackMiddleware` (`fallback.py`) dùng cho cả `graph.py` lẫn `agent.py`:
1. Thử model chính, lỗi thì thử lần lượt `FALLBACK_LLMS` tới khi có model trả lời. Tất cả lỗi thì API trả 429 (quota) hoặc 502.
2. Model bị giới hạn quota bị **tạm bỏ qua**: hết quota ngày → 1 giờ; giới hạn theo phút → 1 phút. Nhờ vậy khi OpenRouter hết quota, các câu sau đi thẳng sang Gemini, không mất thêm một lần gọi hỏng mỗi câu.
3. Lỗi lặp lại ở mọi lần gọi cũng bị tạm bỏ qua:
   - Lỗi cấu hình (`is_permanent_error`: sai tên model, sai/hết hạn API key, không có quyền, hết credit) → **1 giờ**, log mức ERROR "kiểm tra tên model/API key".
   - Lỗi chứng chỉ SSL (`CERTIFICATE_VERIFY_FAILED`, xem [Mạng có SSL inspection](#mạng-có-ssl-inspection)) → **10 phút**.
   - Lỗi tạm thời (mất mạng, 5xx, quá tải) chỉ chuyển model cho lần gọi đó.
4. **Ít thử lại trong chuỗi:** SDK của mỗi model tự thử lại khi gặp lỗi tạm thời. Model đứng trước trong chuỗi chỉ thử lại 1 lần (`RETRIES_IN_CHAIN`) rồi nhường model sau, không chờ backoff 3 lần; model cuối (Groq) thử lại 3 lần (`RETRIES_LAST`) vì sau nó không còn model nào.
5. Mọi model đều đang bị bỏ qua thì vẫn thử hết, còn hơn báo lỗi ngay.
6. Ghi log mỗi lần chuyển: `docker compose logs api | grep LLM` (chạy trên máy thì xem ở terminal của `uvicorn`).

Đo thật ngày 29/10/2026 (OpenRouter hết quota, Gemini 3.1 được thay bằng một model không tồn tại để ép lỗi):

| Lượt | Trước tối ưu | Sau tối ưu |
|---|---|---|
| Câu 1 | OpenRouter 1,6 s → model hỏng 0,8 s → Gemini 3.5 | Như cũ (lần đầu phải thử mới biết hỏng) |
| Câu 2, 3 | Vẫn thử lại model hỏng mỗi câu (0,4–0,5 s) | Bỏ qua cả OpenRouter lẫn model hỏng, gọi thẳng Gemini 3.5 |

Phần lớn thời gian chờ là **độ trễ của chính model dự phòng**, không phải do chuỗi: cùng một câu, Gemini 3.5 mất 15–20 s, Gemini 3.1 dao động 1–23 s, Groq dưới 2 s (đo trước khi thêm NVIDIA; giờ NVIDIA đứng trước Gemini nên thường không phải chờ Gemini). Đã thử tắt hẳn thử lại (`max_retries=0`) và đổi mức "suy nghĩ" của Gemini, đều không nhanh hơn rõ rệt, nên đây là độ trễ phía dịch vụ/mạng.

Eval tắt dự phòng (`fallback=False`) để điểm phản ánh đúng model chính. Thêm hoặc bớt model: sửa `FALLBACK_LLMS` (model mới phải hỗ trợ tool calling; các model `gemini-2.5-*` trả 404 với key hiện tại).

**Giới hạn gói free:**
- **OpenRouter** model `:free`: khoảng 20 request/phút, 50 request/ngày nếu tài khoản chưa nạp 10 USD (nạp rồi thì 1000/ngày). Reset 00:00 UTC, tức 7:00 sáng giờ Việt Nam. Mỗi câu hỏi tốn 2–3 request.
- **Gemini:** giới hạn theo từng model (vd `gemini-3.1-flash-lite` 15 request/phút).
- **Groq:** 1000 token đầu ra/phút, 200.000 token/ngày; `max_tokens=512`.

> **Chú ý:** `ChatOpenRouter` tính `timeout` bằng **mili giây** (`timeout=60_000` = 60 giây), khác Groq/Gemini tính bằng giây. Đặt `timeout=60` thì mọi request bị hủy sau 60ms và agent treo vài phút. `tests/test_llm.py` kiểm tra lỗi này.

## Dữ liệu

Database `phone` (schema `data_schema/phone.sql`, dữ liệu `data_schema/output/phone_seed.sql`):

| Bảng | Số dòng | Nội dung |
|---|---|---|
| `products` | 23 điện thoại + 134 phụ kiện | `product_id` (vd `apple-iphone-18-pro`), `name`, `brand`, `category` (`phone`/`accessory`), `description`, `attributes` (JSON: thông số, RAM theo dung lượng, loại phụ kiện, link nguồn, SKU có giá ước tính) |
| `variants` | 377 | `variant_id` (SKU thật, vd `MJY04X/A`, `SM-S948BZDBXXV`), `product_id`, `color`, `storage`, `price` (VND), `stock` |
| `compatibility` | 738 | Cặp (`accessory_id`, `phone_id`) |

**Lưu ý dữ liệu (demo):**
- `stock` là **số giả lập** (5–50, cố định theo SKU); SKU hết hàng trên web hãng thì `stock = 0`.
- **Một số giá Samsung là ước tính**: web Samsung chỉ nhúng giá cho một số SKU. SKU thiếu giá lấy theo SKU cùng dung lượng, hoặc giá bản thấp nhất cộng chênh lệch dung lượng. Tool ghi "(giá tham khảo)" cho các SKU này.
- Cùng dung lượng nhưng nhiều mức RAM thì `storage` ghi kèm RAM, vd `256GB - RAM 12GB` (Galaxy A57).
- Thông số chi tiết có cho iPhone và Galaxy A/FE; dòng Galaxy S/Z chưa có (web Samsung tải bảng thông số bằng JavaScript).
- Tương thích phụ kiện:
  - Ốp lưng, miếng dán: theo tên máy ghi trong tên phụ kiện.
  - Cáp, củ sạc USB-C: tương thích mọi máy.
  - Phụ kiện MagSafe: chỉ iPhone.
  - Sạc không dây, phụ kiện nam châm Samsung: chỉ dòng S và Z.

**Cập nhật dữ liệu:**

| Dữ liệu | Cách làm |
|---|---|
| Sản phẩm | `cd data_schema && python scrape_phone.py --refresh`, rồi `docker compose run --rm db-seed` |
| Chính sách | Sửa file trong `data_schema/policies/`, rồi `docker compose up -d --build api` (API tự nạp lại). Trên máy: `uv run python -m phone_sales.vector_db` |

`scrape_phone.py` chỉ dùng thư viện chuẩn, đọc dữ liệu nhúng sẵn trong các trang HTML công khai, cache trang đã tải ở `data_schema/.cache/` (không commit), nghỉ 0,8 giây giữa các request. Sau khi cào lại, chạy `uv run pytest` và eval nhóm `product,accessory`, vì giá trong đáp án của eval có thể đã đổi.

## Cấu trúc thư mục

```
sales_agent/
├── data_schema/
│   ├── phone.sql              # Schema database phone
│   ├── scrape_phone.py        # Cào Apple/Samsung VN -> output/
│   ├── output/                # products.json, variants.json, compatibility.json, phone_seed.sql
│   └── policies/              # 5 file chính sách .md (nạp vào Qdrant)
├── src/phone_sales/
│   ├── __init__.py            # Fix DLL torch (Windows), dùng kho chứng chỉ hệ thống, nạp .env
│   ├── config.py              # Đường dẫn, DATABASE_URL, Qdrant, ngưỡng RAG, AGENT_IMPL
│   ├── llm.py                 # LLM chính + FALLBACK_LLMS, nhận diện lỗi quota
│   ├── fallback.py            # FallbackMiddleware: chuỗi LLM dự phòng
│   ├── prompts.py             # System prompt, POLICY_GUIDE, POLICY_PROMPTS
│   ├── tools.py               # 7 tool + gán tool cho từng agent
│   ├── conversation.py        # AgentState + luật cập nhật trạng thái + tóm tắt đưa vào prompt
│   ├── graph.py               # Agent StateGraph có trạng thái + giỏ hàng (web app dùng)
│   ├── agent.py               # Agent create_agent (không có trạng thái riêng)
│   ├── history.py             # Cắt bớt lịch sử trước khi gửi LLM
│   ├── data_prep.py           # Tách chính sách thành đoạn
│   ├── vector_db.py           # Embedding + Qdrant, nạp chính sách
│   ├── api.py                 # FastAPI: trang chat, POST /chat, /health
│   └── static/index.html      # Giao diện chat
├── tests/                     # 118 test
├── evals/                     # dataset.jsonl (37 câu), run_eval.py, results/ (không commit)
├── certs/                     # CA nội bộ (*.crt) cho mạng có SSL inspection, không commit
├── .env.example               # Mẫu .env
├── Dockerfile, docker-compose.yml
├── langgraph.json             # Graph cho `langgraph dev`: agent_main, phone_graph
├── pyproject.toml, uv.lock    # Thư viện (nguồn chính)
└── requirements.txt           # Cho pip, phải khớp pyproject.toml (có test kiểm tra)
```

## Cài đặt

Cần Python 3.12.15 (ghi trong `.python-version`), [uv](https://docs.astral.sh/uv/) và Docker.

```bash
uv sync                  # tạo .venv, cài thư viện + nhóm dev
cp .env.example .env     # rồi điền key
```

Dùng pip thì `pip install -r requirements.txt` (đã có index torch bản CPU). Thêm thư viện bằng `uv add …` rồi cập nhật `requirements.txt` cho khớp; `tests/test_requirements.py` báo nếu hai file lệch nhau.

| Biến trong `.env` | Mặc định | Ý nghĩa |
|---|---|---|
| `OPENROUTER_API_KEY` | (bắt buộc) | LLM chính ([openrouter.ai/keys](https://openrouter.ai/keys)) |
| `NVIDIA_API_KEY` | (bắt buộc) | Dự phòng 1 ([build.nvidia.com](https://build.nvidia.com)) |
| `GOOGLE_API_KEY` | (bắt buộc) | Gemini: dự phòng + judge ([Google AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | (bắt buộc) | Groq: dự phòng ([console.groq.com](https://console.groq.com)) |
| `TAVILY_API_KEY` | (bắt buộc) | Tool `web_search` ([tavily.com](https://tavily.com)) |
| `POSTGRES_PASSWORD` | `postgres` | Mật khẩu container `phone_db`. Chỉ có hiệu lực khi khởi tạo volume lần đầu |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/phone` | DB khi chạy code trên máy (trong Docker, compose tự đặt) |
| `QDRANT_URL` | `http://localhost:6333` | Trong Docker, compose ghi đè thành `http://qdrant:6333` |
| `AGENT_IMPL` | `graph` | Web app dùng `graph.py` (có trạng thái + giỏ hàng) hay `agent.py` |
| `AUTO_INGEST` | `true` | Web app tự nạp chính sách khi Qdrant rỗng hoặc đã cũ |
| `LANGSMITH_*` | tắt | Trace để debug; test và eval luôn tắt |

`.env` được nạp một lần trong `src/phone_sales/__init__.py`; biến đã có trong môi trường thì không bị ghi đè. Trên Windows, xem thêm [Lưu ý trên Windows](#lưu-ý-trên-windows).

## Chạy bằng Docker

```bash
docker compose up -d --build
```

| Container | Cổng | Vai trò |
|---|---|---|
| `phone_db` | 5433 | Postgres 18.6, database `phone`, volume `postgres_data`. Dùng 5433 vì 5432 thường đã có Postgres cài trên máy |
| `qdrant` | 6333 (REST + dashboard), 6334 (gRPC) | Vector DB v1.20.0, volume `qdrant_data` |
| `phone_sales_api` | 8000 | Web app (image `phone-sales-api`) |

Khi khởi động:
- **Postgres** lần đầu (volume rỗng) tự chạy `phone.sql` rồi `phone_seed.sql`. Các lần sau không chạy lại, dùng `db-seed` để nạp dữ liệu mới.
- **API** chờ Postgres healthy, chờ Qdrant (tối đa ~60 giây), tự nạp chính sách nếu Qdrant rỗng hoặc khác file `.md` trong image.
- Image API chứa `src/`, `data_schema/policies/`, thư viện và model embedding (tải sẵn lúc build); chạy bằng user thường; key truyền qua `env_file`.
- Lần build đầu mất khoảng 8 phút, chủ yếu để tải model (~540MB). Model nằm ở layer trước `COPY README.md`/`src`, nên sửa code, chính sách hay README chỉ build lại vài giây. Chỉ khi đổi `pyproject.toml`, `uv.lock` hoặc `certs/` mới phải tải lại model.

```bash
docker compose ps                                       # trạng thái
docker compose logs -f api                              # log
docker compose run --rm db-seed                         # nạp lại phone_seed.sql vào Postgres (không mất volume)
docker compose exec api python -m phone_sales.vector_db   # nạp lại chính sách ngay
docker compose up -d --build api                        # build lại API sau khi sửa code/chính sách
docker compose down                                     # dừng, giữ dữ liệu
docker compose down -v                                  # dừng và xóa volume (lần sau tự nạp lại)
docker exec -it phone_db psql -U postgres -d phone      # mở psql
```

Sau khi chạy: trang chat http://localhost:8000, Swagger http://localhost:8000/docs, dashboard Qdrant http://localhost:6333/dashboard.

### Mạng có SSL inspection

Nếu log báo `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain`, proxy của mạng đang cấp lại chứng chỉ HTTPS bằng CA nội bộ. Trên máy, `truststore` đã dùng kho chứng chỉ Windows nên không bị. Trong container thì Gemini và LangSmith bị lỗi; chuỗi dự phòng khi đó chỉ còn Groq.

Máy hiện tại đã cài CA `tma-ADCA-CA` vào `certs/internal-ca.crt` (hết hạn 01/2029) và Gemini trong container đã gọi được. File `.crt` không được commit, nên máy khác hoặc mạng khác phải làm lại bước dưới.

Cách sửa: xuất CA vào `certs/` rồi build lại (Dockerfile cài mọi `certs/*.crt`):

```powershell
# PowerShell: đổi 'CN=ten-CA*' thành tên CA nội bộ (xem issuer của chứng chỉ bị lỗi)
$c = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -like 'CN=ten-CA*' } | Select-Object -First 1
"-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks') + "`n-----END CERTIFICATE-----" | Set-Content -Encoding ascii certs\internal-ca.crt
docker compose up -d --build api
```

## Chạy trên máy

Postgres và Qdrant vẫn chạy bằng Docker, code chạy trên máy:

```bash
docker compose up -d postgres qdrant
uv run python -m phone_sales.vector_db           # nạp chính sách (lần đầu tải model embedding ~540MB)
uv run python -m phone_sales.graph               # hội thoại mẫu 3 lượt, in trạng thái sau mỗi lượt
uv run python -m phone_sales.agent               # 4 câu mẫu với agent.py
uv run uvicorn phone_sales.api:app --port 8000   # web app (thêm --reload khi sửa code)
uv run langgraph dev                           # LangGraph Studio: graph agent_main và phone_graph
```

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
| `GET` | `/` | Trang chat (Enter gửi, Shift+Enter xuống dòng, nút "Cuộc trò chuyện mới") |
| `POST` | `/chat` | Gửi tin nhắn, nhận câu trả lời đầy đủ (JSON) |
| `POST` | `/chat/stream` | Như `/chat` nhưng trả Server-Sent Events: trạng thái và chữ hiện dần (trang chat dùng endpoint này) |
| `GET` | `/health` | `{"status": "ok", "documents": 66}`, hoặc 503 nếu không kết nối được Qdrant |
| `GET` | `/docs` | Swagger UI |

```json
// POST /chat — bỏ trống thread_id để bắt đầu hội thoại mới, gửi lại thread_id ở các lượt sau
{"message": "ok chốt galaxy s26 256gb màu đen", "thread_id": "5f0c…"}

// response
{"thread_id": "5f0c…", "answer": "Dạ em đã thêm Galaxy S26 256GB Đen Classic vào giỏ…",
 "tools": ["add_to_cart"], "stage": "closing",
 "cart": [{"variant_id": "SM-S942BZKCXXV", "name": "Galaxy S26", "storage": "256GB", "color": "Đen Classic", "price": 29990001, "quantity": 1}],
 "handoff": false}
```

- `message` 1–1000 ký tự, `thread_id` tối đa 100 ký tự.
- `stage`, `cart`, `handoff` là trạng thái sau lượt này. Với `AGENT_IMPL=agent` thì luôn `null`, `[]`, `false`. Hệ thống khác có thể dùng `handoff` để báo nhân viên.
- Mã lỗi: 422 (input sai), 429 (mọi LLM vượt giới hạn), 502 (mọi LLM lỗi), 503 (`/health` không kết nối được Qdrant).
- Trên Windows, curl của Git Bash có thể gửi sai mã hóa tiếng Việt (API trả 400); dùng PowerShell `Invoke-RestMethod`, Swagger UI hoặc Python.

**`/chat/stream`**: cùng request với `/chat`; response là `text/event-stream`, mỗi sự kiện là 1 dòng `data: {json}`:

| `type` | Khi nào | Giao diện làm gì |
|---|---|---|
| `start` | Ngay khi nhận request, kèm `thread_id` | Lưu `thread_id` |
| `status` | Model bắt đầu suy nghĩ (`Em đang cân nhắc…`) hoặc gọi tool (vd `Đang xem giá và tồn kho…`) | Đổi dòng trạng thái cạnh 3 chấm nhảy |
| `token` | Từng đoạn chữ LLM viết | Nối vào bong bóng (chữ thô) |
| `reset` | Model quyết định gọi tool sau khi đã viết vài chữ dẫn | Xóa phần chữ tạm |
| `done` | Xong lượt, nội dung giống response của `/chat` | Hiển thị lại câu trả lời có định dạng + "Đã tra cứu" |
| `error` | Lỗi, kèm `status` (429/502/500) và `detail` | Hiện lỗi trong bong bóng |

- Chỉ chữ do LLM viết ở node `model` được gửi đi; kết quả tool và **nội dung suy nghĩ** của model (NVIDIA thinking) không bao giờ được gửi cho khách, chỉ báo trạng thái "Em đang cân nhắc…".
- Đo ngày 29/10/2026 (NVIDIA trả lời, thinking bật, hội thoại 3 lượt):

  | Câu | Trạng thái đầu tiên | Chữ đầu tiên | Xong |
  |---|---|---|---|
  | "tầm 15 triệu có máy samsung nào pin trâu" | 2,4 s (cân nhắc) → 3,2 s (tìm sản phẩm) | 16,6 s | 17,4 s |
  | "chào em" | 1,0 s | 1,8 s | 2,2 s |
  | "ok chốt a57 128gb xanh đêm" | 0,9 s → 3,4 s (cập nhật giỏ) | 5,0 s | 5,6 s |

  Khách thấy phản hồi trong khoảng 1–3 s. Chữ thường tới gần cuối vì model suy nghĩ xong mới viết (và viết nhanh), nên tổng thời gian không đổi; lợi ích chính là khách biết hệ thống đang làm gì thay vì màn hình đứng yên.
- Header `X-Accel-Buffering: no` để chữ không bị gom lại nếu đặt API sau nginx.

**Giới hạn:** state hội thoại nằm trong RAM (`InMemorySaver`) nên mất khi restart, và uvicorn chỉ chạy 1 worker. Chạy nhiều worker thì cần checkpointer dùng chung, vd `langgraph-checkpoint-postgres`. API chưa có xác thực hay giới hạn request.

## Test

```bash
uv run pytest                          # 118 test, dưới 1 phút
uv run pytest tests/test_conversation.py -v
```

Test **không gọi LLM, Tavily, mạng hay database**: `conftest.py` đặt key giả và tắt trace; tool DB dùng `_query` giả; agent và graph dùng model giả.

| File | Kiểm tra |
|---|---|
| `test_conversation.py` | Đọc ngân sách/hãng/nhu cầu/dung lượng; không khớp nhầm từ khóa ("không", "chip", "đặt", "tức là"); lo ngại, cảm xúc; khi nào chuyển nhân viên; chuyển bước; reducer; chỉ tính máy được nhắc trong câu trả lời; tool giỏ hàng (gộp phiên bản, tồn kho, hỏi lại phiên bản, xóa); graph giữ trạng thái qua 2 lượt |
| `test_tools.py` | Nhận diện tên máy, sắp xếp dung lượng, loại phụ kiện tiếng Việt, định dạng giá/tồn kho; 3 tool tra cứu DB; `search_policy` (gom theo file kèm prompt, bỏ đoạn yếu, chặn câu lạc đề, mọi file có prompt); tool tự bắt lỗi |
| `test_agent.py` | Chuỗi dự phòng: đi hết chuỗi tới model trả lời được (cả `agent.py` lẫn `graph.py`); tạm bỏ qua model hết quota (1 phút / 1 giờ), lỗi cấu hình (1 giờ) hoặc lỗi chứng chỉ SSL (10 phút), lỗi tạm thời thì không; mọi model bị bỏ qua thì vẫn thử hết; `fallback=False` báo lỗi model chính |
| `test_data_prep.py` | Tách chính sách theo mục và FAQ, cắt bảng dài, 5 file đều tách được |
| `test_api.py` | Endpoint, validate input, mã lỗi 429/502/503; `/chat/stream` (thứ tự sự kiện, `reset` bỏ chữ dẫn trước tool, không stream kết quả tool hay nội dung thinking, lỗi giữa chừng thành sự kiện `error`); nhớ theo `thread_id`, tự nạp chính sách khi rỗng hoặc đã cũ, thử lại khi Qdrant chưa sẵn sàng |
| `test_history.py` | Cắt lịch sử đúng ranh giới lượt hỏi, rút gọn kết quả tool cũ |
| `test_llm.py` | `timeout` đúng đơn vị; 5 model trong chuỗi không trùng nhau, NVIDIA đứng đầu dự phòng; chỉ model cuối chuỗi thử lại nhiều lần; nhận diện lỗi cấu hình; đọc mã lỗi của NVIDIA (429, 401…) |
| `test_eval_dataset.py` | `evals/dataset.jsonl` hợp lệ: tên trường, id không trùng, tool và file `policy` tồn tại, câu hỏi không chép nguyên văn dữ liệu |
| `test_requirements.py` | `requirements.txt` khớp `pyproject.toml` |

## Đánh giá (eval)

Eval gọi LLM, Postgres, Qdrant, Tavily thật trên bộ câu hỏi cố định rồi chấm. Chạy khi đổi model, prompt, tool hoặc dữ liệu.

```bash
docker compose up -d postgres qdrant
uv run python evals/run_eval.py --retrieval-only          # chỉ chấm truy xuất chính sách, không gọi LLM
uv run python evals/run_eval.py --only product,accessory  # lọc theo nhóm hoặc id
uv run python evals/run_eval.py                           # cả 37 câu
uv run python evals/run_eval.py --target graph            # chấm graph.py thay cho agent.py
```

Eval tắt chuỗi dự phòng; hết quota ngày thì câu đó ghi là lỗi và không tính vào thống kê. Cả 37 câu cần khoảng 90 request OpenRouter, vượt 50 request/ngày của gói free, nên phải chia 2 ngày.

| Nhóm | Số câu | Kiểm tra |
|---|---|---|
| `policy` | 19 | Gọi `search_policy`, đúng file chính sách, đúng thời hạn/phí |
| `product` | 7 | Đúng tool sản phẩm, đúng giá (vd iPhone 18 Pro Max 256GB = 41.999.000đ) |
| `accessory` | 3 | `find_accessories`, đúng sản phẩm và giá |
| `escalation` | 3 | Giao thiếu, giao chậm, đòi gặp người thật → báo chuyển nhân viên |
| `security` | 2 | Không nhận số thẻ/CVV, không lộ thông tin khách khác |
| `web` | 1 | Thông tin ngoài dữ liệu nội bộ |
| `out_of_scope` | 2 | Từ chối ngắn gọn, không gọi tool |

Chỉ số: **hit@1/hit@3/MRR** (truy xuất chính sách), **routing** (tool đầu tiên đúng), **keyword** (có đủ `must_contain`), **behavior** (chuyển nhân viên / từ chối đúng lúc), **grounded** và **score 1–5** (Gemini chấm). Kết quả lưu ở `evals/results/`.

**Kết quả hiện có:**

Truy xuất chính sách (28/10/2026): hit@1 89%, hit@3 100%, MRR 0.95.

Agent `graph.py` (29/10/2026, OpenRouter, 17 câu thuộc 5 nhóm; mỗi câu lấy lần chạy mới nhất trong 2 lần chạy, xem `evals/results/20261029-*-graph.json`):

| Nhóm | Số câu | routing | keyword | behavior | grounded | Điểm TB | Độ trễ TB |
|---|---|---|---|---|---|---|---|
| `product` | 7 | 6/6 | 6/6 | - | 7/7 | 4.57 | 26.9 s |
| `accessory` | 3 | 2/3 | 3/3 | - | 3/3 | 4.00 | 27.7 s |
| `escalation` | 3 | 1/2 | - | 3/3 | 3/3 | 4.67 | 25.0 s |
| `security` | 2 | 1/1 | - | 1/1 | 2/2 | 5.00 | 11.6 s |
| `out_of_scope` | 2 | 2/2 | - | 2/2 | 2/2 | 5.00 | 5.7 s |
| **Tổng** | **17** | **12/14** | **9/9** | **6/6** | **17/17** | **4.59** | **22.4 s** |

- Lần chạy đầu tìm ra các lỗi đã được sửa:
  - Model đọc số lượng kho cho khách → tool bỏ số lượng.
  - Model mời "kiểm tra số lượng chính xác" → sửa quy tắc 8 trong prompt.
  - Judge hiểu sai quy tắc "giá tham khảo" và "tối đa 3 máy" → sửa prompt chấm điểm.
  - Model liệt kê 8 ốp lưng dù prompt dặn tối đa 5 → `find_accessories` chia trang 5 món. **Sửa này chưa chạy lại eval** (hết quota), nên `acc-01` vẫn ghi 3 điểm.
- Routing chưa đạt:
  - `acc-02` gọi `search_products` trước `find_accessories` (thừa một lần gọi, kết quả đúng).
  - `esc-03` được chấm trước khi câu này đổi sang `tools_optional` (theo quy tắc 12, tra chính sách trước khi chuyển nhân viên là hợp lệ).
- 2 câu từng lỗi `provider_overloaded` do OpenRouter quá tải: eval tắt dự phòng; trên web app thì chuỗi dự phòng sẽ đỡ những lỗi này.
- Độ trễ cao (trung bình 22 giây) vì model chính là model suy luận và thường gọi 2 tool mỗi câu.
- **Chưa chạy** với bộ câu mới: nhóm `policy` (19 câu) và `web` (1 câu). Bộ eval cũng chưa có câu nhiều lượt; trạng thái hội thoại và giỏ hàng mới được kiểm tra bằng test và chạy thử qua `/chat`.

Hạn chế: ít câu, LLM dao động giữa các lần chạy, judge là LLM nên có thể chấm sai.

## Mở rộng dự án

**Thêm tool:**
1. Viết hàm `@tool` trong `tools.py`. Docstring là mô tả model đọc, nên ghi rõ dùng khi nào, không dùng khi nào. Tự bắt lỗi bên trong tool.
2. Thêm vào `TOOLS` và `AGENT_TOOLS` (`agent_main` và/hoặc `agent_graph`), thêm quy tắc trong `prompts.py`.
3. Thêm test và câu eval.

**Thêm trường trạng thái:** khai báo trong `AgentState` (kèm reducer nếu cần cộng dồn), cập nhật ở `track_user` / `track_tools` / `track_reply` hoặc bằng tool trả `Command`, rồi đưa vào `describe()` để model thấy.

**Ý tưởng tiếp theo:**
- Tool đặt hàng thật từ giỏ hàng.
- Checkpointer Postgres để state không mất khi restart.
- Câu eval nhiều lượt.
- Cào thêm thông số dòng Galaxy S/Z.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| "Lỗi hệ thống khi tra cứu kho hàng" | Postgres chưa chạy hoặc sai `DATABASE_URL` | `docker compose up -d postgres`; trên máy dùng cổng **5433** |
| Giá/sản phẩm chưa đổi sau khi cào lại | File init chỉ chạy lần đầu | `docker compose run --rm db-seed` |
| `password authentication failed` sau khi đổi `POSTGRES_PASSWORD` | Mật khẩu chỉ đặt lúc tạo volume | `docker compose down -v` rồi chạy lại |
| `search_policy` không tìm thấy gì | Chưa nạp chính sách | `uv run python -m phone_sales.vector_db` |
| `/chat` trả 429 | Mọi model trong chuỗi vượt giới hạn | Chờ quota reset; `docker compose logs api \| grep LLM` |
| `/chat` trả 502 | Mọi model lỗi (sai key, mất mạng, SSL…) | Kiểm tra key trong `.env` và log |
| Trong Docker chỉ Groq trả lời | Gemini lỗi `CERTIFICATE_VERIFY_FAILED` (thiếu CA nội bộ trong `certs/`) | Xem [Mạng có SSL inspection](#mạng-có-ssl-inspection) |
| Câu trả lời khác thường | Model dự phòng đang trả lời | `docker compose logs api \| grep "model dự phòng"` |
| `/health` trả 503 | Qdrant chưa chạy hoặc sai `QDRANT_URL` | `docker compose up -d qdrant` |
| Agent treo vài phút | `timeout` OpenRouter đặt bằng giây | `timeout=60_000` trong `llm.py` |
| Log `Failed to multipart ingest runs` | LangSmith không gửi được trace | `LANGSMITH_TRACING=false`; không ảnh hưởng chức năng |
| `uvicorn` báo `[Errno 10048] ... ('127.0.0.1', 8000)` | Container `phone_sales_api` đang giữ cổng 8000 | Dùng luôn web app trong Docker; hoặc `docker compose stop api` rồi chạy `uvicorn`; hoặc chạy `uvicorn ... --port 8001` |
| Sửa code nhưng Docker không đổi | Image cũ | `docker compose up -d --build api` |

## Lưu ý trên Windows

- **`OSError: [WinError 1114] ... torch\lib\c10.dll`:** antivirus (vd McAfee) nạp `msvcp140.dll` bản cũ vào Python, mà `torch` cần bản mới hơn. Tạo file `.pth` để nạp DLL của Windows khi Python khởi động (**chạy lại mỗi khi tạo lại `.venv`**):

  ```bash
  uv run python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[-1], '_msvcp140_preload.pth').write_text(\"import sys, ctypes; sys.platform == 'win32' and ctypes.WinDLL('msvcp140.dll')\n\")"
  ```

  `src/phone_sales/__init__.py` cũng nạp trước DLL này, nhưng không đủ cho `langgraph dev`.
- **`UnicodeEncodeError`** khi in tiếng Việt ra pipe/file: đặt `PYTHONUTF8=1`.
- **Git Bash đổi đường dẫn** trong lệnh `docker run ... /app/...`: thêm `MSYS_NO_PATHCONV=1` trước lệnh.
