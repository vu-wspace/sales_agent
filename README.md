# phone-sales-agent

Agent tư vấn bán điện thoại bằng tiếng Việt cho một cửa hàng bán lẻ (demo, phục vụ học tập), xây dựng với LangChain/LangGraph. Agent tư vấn iPhone, Samsung Galaxy và phụ kiện: tìm máy theo tầm giá, báo giá và tình trạng hàng từng phiên bản, tìm phụ kiện tương thích, trả lời về chính sách đổi trả, giao hàng, trả góp và bảo hành.

- **Dữ liệu sản phẩm** (23 điện thoại, 134 phụ kiện, 377 phiên bản) cào từ apple.com/vn và samsung.com/vn, lưu trong Postgres (database `phone`).
- **Chính sách** (5 file `.md`) được tách thành 65 đoạn và tìm bằng vector search trong Qdrant (RAG).
- Có **web chat** (FastAPI) chạy bằng Docker, nhớ ngữ cảnh hội thoại.
- LLM chính là OpenRouter (`nvidia/nemotron-3-ultra-550b-a55b:free`). Khi lỗi hoặc hết quota, agent **tự chuyển lần lượt** sang Gemini 3.1 flash-lite → Gemini 3.5 flash-lite → Groq, và tạm bỏ qua model vừa hết quota (xem [Chuỗi LLM dự phòng](#chuỗi-llm-dự-phòng)).
- Agent nhớ **trạng thái hội thoại bán hàng** (bước bán hàng, nhu cầu, sản phẩm đã giới thiệu, giỏ hàng, lo ngại, cảm xúc, cần chuyển nhân viên) và có **giỏ hàng** (xem [Trạng thái hội thoại](#trạng-thái-hội-thoại)).
- Có **105 test** (không gọi API thật) và **bộ eval 37 câu** để đo chất lượng câu trả lời.

> Package Python vẫn tên `bank_faqs` (dự án được chuyển từ agent CSKH ngân hàng). Đổi tên package là việc riêng, cần sửa mọi import.

Chạy nhanh (cần Docker và file `.env`, xem [Cài đặt](#cài-đặt)):

```bash
docker compose up -d --build   # Postgres (tự nạp dữ liệu) + Qdrant + web app (tự nạp chính sách)
# mở http://localhost:8000
```

## Mục lục

- [Cách hoạt động](#cách-hoạt-động)
- [Các tool](#các-tool)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Deploy bằng Docker](#deploy-bằng-docker)
- [Chạy trên máy không dùng Docker](#chạy-trên-máy-không-dùng-docker)
- [Web app và API](#web-app-và-api)
- [Test](#test)
- [Đánh giá (eval)](#đánh-giá-eval)
- [Giới hạn của LLM (gói free)](#giới-hạn-của-llm-gói-free)
- [Mở rộng dự án](#mở-rộng-dự-án)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Lưu ý trên Windows](#lưu-ý-trên-windows)

## Cách hoạt động

```
Trình duyệt (trang chat)                    LangGraph Studio (langgraph dev)
        │ POST /chat                                   │
        ▼                                              ▼
FastAPI (api.py) ── nhớ hội thoại theo thread_id ──> Agent (agent.py: create_agent, hoặc graph.py: StateGraph)
                                                       │
    LLM: OpenRouter nemotron ── lỗi/hết quota ──> Gemini 3.1 flash-lite ──> Gemini 3.5 flash-lite ──> Groq qwen3.8
                                                       │ gọi tool
     ┌──────────────────┬──────────────────┼──────────────────┬──────────────────┐
     ▼                  ▼                  ▼                  ▼                  ▼
search_products  get_product_details  find_accessories   search_policy       web_search
     └──────── Postgres: database phone ───────┘     Qdrant (65 đoạn       Tavily
          (products, variants, compatibility)          chính sách, bkai)    (3 kết quả)
```

### Luồng xử lý một câu hỏi

1. Trang chat gửi `POST /chat` với `message` và `thread_id` (bỏ trống ở tin nhắn đầu tiên).
2. `api.py` lấy lịch sử của `thread_id` từ bộ nhớ (`InMemorySaver`), thêm tin nhắn mới rồi gọi agent.
3. Trước mỗi lần gọi LLM, `history.py` **cắt bớt lịch sử** gửi đi: chỉ giữ 4 lượt hỏi gần nhất, kết quả tool của các lượt cũ rút còn 300 ký tự. State trong bộ nhớ vẫn giữ đầy đủ.
4. LLM đọc system prompt (`prompts.py`) và mô tả các tool (docstring của hàm `@tool` trong `tools.py`), rồi quyết định gọi tool nào. Agent có thể gọi nhiều tool trong một lượt (vd hỏi giá máy kèm trả góp: `get_product_details` rồi `search_policy`).
5. Tool trả kết quả dạng văn bản. LLM soạn câu trả lời chỉ dựa trên kết quả đó.
6. Nếu OpenRouter lỗi (hết quota, quá tải, mất kết nối…), lần gọi đó được chuyển sang model dự phòng tiếp theo (`FallbackMiddleware` trong `fallback.py`).
7. API trả về `answer`, `thread_id` và danh sách `tools` đã gọi trong lượt này.

### Quy tắc của agent

System prompt nằm trong `src/bank_faqs/prompts.py` (`AGENT_PROMPTS['agent_main']`), gồm 15 quy tắc:

| Nhóm | Quy tắc |
|---|---|
| Xưng hô | Tiếng Việt, lịch sự, xưng "em", gọi khách là "anh/chị", không đoán giới tính |
| Dùng tool | (1) Tìm máy, tầm giá, so sánh giá → `search_products`. (2) Giá chính xác, màu, dung lượng, còn hàng, thông số → `get_product_details`; nhiều máy khớp thì hỏi lại. (3) Phụ kiện → `find_accessories`. (4) Đổi trả, giao hàng, trả góp, bảo hành → `search_policy`. (5) `web_search` chỉ khi các tool trên không có câu trả lời, không dùng để lấy giá hay chính sách. (6) Xem kết quả các lần gọi trước trước khi gọi tiếp |
| Trả lời | (7) Chỉ dựa trên kết quả tool; ghi giá đúng như tool trả về; chỉ dòng có "(giá tham khảo)" mới nói là giá tham khảo. (8) Tồn kho chỉ nói còn hàng / sắp hết / hết hàng, không đọc số lượng; hết hàng thì gợi ý phiên bản khác. (9) Tư vấn tối đa 3 máy kèm giá và lý do, có thể gợi ý phụ kiện. (10) Không có thông tin thì nói rõ và đề nghị liên hệ hotline/cửa hàng. (11) Tối đa khoảng 100 từ |
| Chuyển nhân viên | (12) Khiếu nại, máy lỗi, đơn giao sai/chậm/thiếu, hủy đơn đã thanh toán, muốn gặp người thật: nêu bước cần làm theo chính sách rồi báo chuyển nhân viên |
| Bảo mật | (13) Không hỏi OTP, mật khẩu, PIN, số thẻ đầy đủ, CVV. (14) Không cung cấp thông tin khách hàng khác |
| Ngoài phạm vi | (15) Chủ đề không liên quan mua bán điện thoại: không gọi tool, từ chối trong 1–2 câu, hỏi khách có cần tư vấn sản phẩm không |

Có 2 cách dựng agent, dùng chung LLM, 5 tool tra cứu, system prompt, cách cắt lịch sử và chuỗi LLM dự phòng:

| | `graph.py` (`StateGraph`) — **web app dùng mặc định** | `agent.py` (`create_agent`) |
|---|---|---|
| Trạng thái hội thoại | Có (`AgentState` trong `conversation.py`) | Chỉ có `messages` |
| Giỏ hàng | Có tool `add_to_cart`, `remove_from_cart` | Không |
| System prompt | `agent_main` + quy tắc 16–18 (`agent_graph`) + tóm tắt trạng thái mỗi lượt | `agent_main` |

Đổi web app sang `agent.py`: đặt `AGENT_IMPL=agent`.

### Trạng thái hội thoại

`AgentState` (`src/bank_faqs/conversation.py`) được lưu theo `thread_id` cùng lịch sử (checkpointer), nên giữ qua các lượt:

| Trường | Ý nghĩa | Cập nhật bởi |
|---|---|---|
| `messages` | Lịch sử tin nhắn (reducer `add_messages`: nối thêm, không ghi đè) | Mọi node |
| `stage` | `discovery` (chưa xem máy) → `recommending` (đã giới thiệu) → `comparing` (đang so sánh ≥ 2 máy) → `closing` (có hàng trong giỏ hoặc khách chốt) → `handoff` | `decide_stage`, tính lại sau mỗi bước |
| `slots` | Nhu cầu: `budget_max`, `brand`, `purposes` (chụp ảnh, chơi game, pin lâu…), `storage`, `color` (reducer gộp, list cộng dồn) | Tin nhắn của khách + tham số tool model đã gọi |
| `shown_products` | Sản phẩm đã giới thiệu: có trong kết quả tool của lượt **và** được nhắc trong câu trả lời gửi khách | `track_reply` |
| `cart` | Giỏ hàng: `variant_id`, `name`, `storage`, `color`, `price`, `quantity` | Tool `add_to_cart` / `remove_from_cart` (kiểm tra tồn kho, gộp cùng phiên bản) |
| `objections` | Lo ngại: chê giá cao, lo pin yếu, máy nóng, nặng/to, camera, lo lỗi, không thích màu | Tin nhắn của khách |
| `sentiment_history` | `positive` / `neutral` / `negative` mỗi lượt | Tin nhắn của khách |
| `handoff` | Cần chuyển nhân viên: khách đòi gặp người thật, khiếu nại, giao sai/chậm/thiếu, hủy đơn, hoặc **2 lượt liên tiếp bực bội**. Đã bật thì giữ cho cả hội thoại | Tin nhắn của khách |

Luồng của graph: `START → track_user → model → (gọi tool) tools → track_tools → model … → track_reply → END`.
- Cập nhật bằng **luật** (từ khóa tiếng Việt có dấu + kết quả tool), **không gọi thêm LLM**: không tốn quota, chạy tức thì, test được. Từ khóa so trên chữ có dấu và có ranh giới từ, để "đắt" không trùng "đặt", "ông" không khớp trong "không", "tức là" không bị tính là tức giận.
- Mỗi lần gọi model, `describe(state)` thêm mục "Trạng thái hội thoại" vào cuối system prompt: bước hiện tại kèm hướng dẫn cho bước đó (vd `closing`: xác nhận giỏ và tổng tiền, hỏi cách nhận hàng/thanh toán; `handoff`: xin lỗi, báo chuyển nhân viên, không bán thêm), nhu cầu đã biết (dặn không hỏi lại), sản phẩm đã giới thiệu, giỏ hàng, lo ngại, và nhắc trả lời ngắn khi khách đang không vui.
- Thêm vào giỏ **chưa phải đặt hàng**: agent báo nhân viên sẽ liên hệ xác nhận đơn.
- Hạn chế: luật từ khóa không hiểu được mọi cách nói (vd mỉa mai, viết không dấu nặng), và chưa có tool đặt hàng thật.

## Các tool

| Tool | Nguồn dữ liệu | Dùng khi |
|---|---|---|
| `search_products(query='', brand='', max_price=0, category='phone')` | Postgres | Có bán máy nào, máy trong tầm giá, so sánh giá các dòng |
| `get_product_details(product, storage='', color='')` | Postgres | Giá, màu, dung lượng, tồn kho từng phiên bản, RAM, thông số chính của 1 máy |
| `find_accessories(phone, accessory_type='')` | Postgres (`compatibility`) | Ốp lưng, miếng dán, sạc, cáp, sạc dự phòng, S Pen… tương thích với 1 máy |
| `search_policy(query)` | Qdrant, tối đa 3 đoạn đã lọc + prompt riêng của từng file | Đổi trả, hoàn tiền, giao hàng, trả góp, bảo hành |
| `web_search(query)` | Tavily, 3 kết quả | Đánh giá, tin tức, máy sắp ra mắt – thông tin không có trong dữ liệu nội bộ |

Ví dụ `get_product_details('ip 18 pro', storage='256gb')`:

```
iPhone 18 Pro (Apple)
iPhone 18 Pro. Một cỗ máy AI siêu mạnh. ...
Màn Hình: Màn hình Super Retina XDR ... OLED 6,3 inch ...
Chip: Chip A20 Pro CPU 6 lõi ...
Phiên bản (dung lượng | màu | giá | kho):
- 256GB | Bạc | 38.999.000đ | còn 20
- 256GB | Đen | 38.999.000đ | còn 46
...
```

**Nhận diện tên máy** (`_match`, `_tokens` trong `tools.py`):
1. Chuẩn hóa: chữ thường, bỏ dấu, `+` → `plus`, tách "17pro" → "17 pro".
2. Đổi tên gọi tắt theo `ALIASES` (`ip` → `iphone`, `promax` → `pro max`…), bỏ các từ như "galaxy", "samsung", "5g", "điện thoại" và dung lượng ("256gb").
3. Mọi từ khách gõ phải có trong tên sản phẩm. Nếu có tên trùng khớp hoàn toàn thì chỉ lấy tên đó: "iPhone 18 Pro" không kéo theo "iPhone 18 Pro Max", "s26" trong `get_product_details` là Galaxy S26. Riêng `search_products` lấy cả dòng: "s26" → S26 FE, S26, S26+, S26 Ultra.
4. `get_product_details` và `find_accessories` cần đúng 1 máy; nhiều máy khớp thì trả danh sách để model hỏi lại khách.

**Loại phụ kiện**: `accessory_type` nhận cả tiếng Việt ("ốp lưng", "cường lực", "sạc dự phòng", "củ sạc", "cáp"…), đổi sang loại trong DB theo `ACCESSORY_TYPES`.

**Tồn kho**: tool trả `còn N` / `sắp hết (còn N)` (N ≤ 5) / `hết hàng`; prompt dặn model không đọc số lượng cho khách.

**`search_policy` chống lạc đề và lan man** (`tools.py`, `prompts.py`, `config.py`):
1. **Lọc đoạn:** lấy 3 đoạn gần nhất kèm điểm cosine, bỏ đoạn dưới `POLICY_MIN_SCORE = 0.25` và đoạn kém đoạn tốt nhất quá `POLICY_SCORE_MARGIN = 0.08`. Ngưỡng đo trên 19 câu `policy` của bộ eval: câu đúng chủ đề có điểm top 1 từ 0,30; câu lạc đề (bitcoin, làm thơ, thời tiết, "gặp nhân viên") chỉ 0,10–0,21, nên tool trả "Không tìm thấy chính sách liên quan" thay vì đưa đoạn không liên quan cho model. Sau khi lọc vẫn ra đúng file 19/19, trung bình 2,2 đoạn / 1,3 file mỗi câu (trước là 3 đoạn, thường lẫn 2–3 file).
2. **Prompt chung** `POLICY_GUIDE` đặt đầu kết quả: trả lời trực tiếp trước, tối đa 3 ý bổ sung, không chép cả bảng, giữ đúng con số, đoạn không trả lời được thì nói chưa có thông tin.
3. **Prompt riêng từng file** `POLICY_PROMPTS`: đoạn được gom theo file, mỗi nhóm có `Hướng dẫn trả lời` nói phạm vi của chính sách đó, ý bắt buộc và điều không được nói. Ví dụ `doi-tra.md`: xác định khách thuộc trường hợp nào (lỗi trong 30 ngày, đổi ý còn seal, đã kích hoạt…) rồi chỉ trả lời trường hợp đó, chưa rõ thì hỏi lại 1 câu; `tra-gop.md`: số tiền góp chỉ là ước tính, không hứa duyệt hồ sơ; `bao-hanh-apple.md`: tách bảo hành của Apple với quyền lợi của cửa hàng, câu về Samsung thì không dùng đoạn này.

Thêm file chính sách mới thì phải thêm prompt vào `POLICY_PROMPTS` (`tests/test_tools.py` báo nếu thiếu). Sửa nhiều chính sách thì nên đo lại ngưỡng (xem chú thích trong `config.py`).

**Xử lý lỗi:** mọi tool tự bắt lỗi (Postgres hay Qdrant sập, Tavily timeout…) và báo cho model là đang lỗi hệ thống, dặn không tự đưa ra giá hay chính sách. Nhờ vậy cả lượt hội thoại không bị hỏng: `ToolNode` của LangGraph chỉ tự bắt lỗi sai tham số.

## Dữ liệu

### Sản phẩm (Postgres, database `phone`)

Schema ở `data_schema/phone.sql`, dữ liệu ở `data_schema/output/phone_seed.sql` (sinh bởi `data_schema/scrape_phone.py`).

| Bảng | Số dòng | Nội dung |
|---|---|---|
| `products` | 23 điện thoại + 134 phụ kiện | `product_id` (vd `apple-iphone-18-pro`, `acc-samsung-ef-qs948ctegww`), `name`, `brand`, `category` (`phone`/`accessory`), `description`, `attributes` (JSON: thông số, RAM theo dung lượng, loại phụ kiện, link nguồn, SKU có giá ước tính) |
| `variants` | 377 | `variant_id` (SKU thật, vd `MJY04X/A`, `SM-S948BZDBXXV`), `product_id`, `color`, `storage`, `price` (VND), `stock` |
| `compatibility` | 738 | Cặp (`accessory_id`, `phone_id`) |

Lưu ý về dữ liệu (mục đích demo):
- **`stock` là số giả lập** (5–50, cố định theo SKU), vì web hãng không công khai tồn kho. SKU hết hàng trên web hãng thì `stock = 0`.
- **Một số giá Samsung là ước tính**: trang Samsung chỉ nhúng giá cho một số SKU; SKU thiếu giá lấy theo SKU cùng dung lượng, hoặc giá bản thấp nhất cộng chênh lệch theo dung lượng. Các SKU này nằm trong `attributes.estimated_price_skus`, tool ghi "(giá tham khảo)".
- Máy có nhiều mức RAM cho cùng dung lượng (vd Galaxy A57 256GB có 8GB và 12GB) thì `storage` ghi kèm RAM: `256GB - RAM 12GB`.
- Thông số chi tiết có cho iPhone và dòng Galaxy A/FE; dòng Galaxy S/Z flagship chưa có (trang Samsung tải bảng thông số bằng JavaScript).
- Tương thích phụ kiện: ốp lưng/miếng dán theo tên model trong tên phụ kiện; cáp và củ sạc USB-C tương thích mọi máy; phụ kiện MagSafe chỉ cho iPhone; sạc không dây/nam châm Samsung chỉ cho dòng S và Z.

### Chính sách (Qdrant, collection `phone_policies`)

`data_schema/policies/`: `doi-tra.md`, `giao-hang.md`, `tra-gop.md`, `bao-hanh-apple.md`, `bao-hanh-samsung.md`. Chính sách của cửa hàng (lãi suất, phí, đối tác) là **giả lập**; phần bảo hành của hãng là tóm tắt tham khảo, cần đối chiếu trang chính thức.

`data_prep.py` tách mỗi file thành đoạn (chunk):
- Mỗi mục `##`/`###` là một đoạn; mục "Câu hỏi thường gặp" thì mỗi cặp hỏi/đáp là một đoạn.
- Đoạn dài quá 900 ký tự (`CHUNK_MAX_CHARS`) được cắt nhỏ; bảng dài cắt theo dòng và lặp lại tiêu đề bảng. Model embedding chỉ đọc khoảng 256 token đầu, nên đoạn dài hơn sẽ mất phần cuối khi embed.
- Đầu mỗi đoạn gắn tên tài liệu và tên mục (vd `Chính sách đổi trả - 3. Đổi trả do lỗi nhà sản xuất > 3.2. Trong 30 ngày đầu`) để tìm kiếm hiểu ngữ cảnh.

`vector_db.py`: tách từ tiếng Việt bằng `pyvi`, embedding bằng `bkai-foundation-models/vietnamese-bi-encoder` (768 chiều, cosine). ID mỗi đoạn tính từ tên file, thứ tự và **nội dung** đoạn, nên sửa file `.md` thì ID đổi: web app khi khởi động so ID trong Qdrant với ID tính từ file, lệch thì tự nạp lại (`is_up_to_date`). Mỗi lần nạp xóa hết đoạn cũ trước.

### Cập nhật dữ liệu

| Dữ liệu | Cách cập nhật |
|---|---|
| Sản phẩm | `cd data_schema && python scrape_phone.py --refresh` (cào lại, ghi `output/`), rồi `docker compose run --rm db-seed` (nạp vào container Postgres, không mất volume) |
| Chính sách | Sửa file trong `data_schema/policies/`. Docker: `docker compose up -d --build api` (API tự nạp lại vì nội dung đổi). Trên máy: `uv run python -m bank_faqs.vector_db` |

`scrape_phone.py` chỉ dùng thư viện chuẩn, đọc các trang HTML công khai (dữ liệu nhúng sẵn trong trang), cache trang đã tải trong `data_schema/.cache/` (không commit), nghỉ 0,8 giây giữa các request. Sau khi cào lại, chạy `uv run pytest tests/test_eval_dataset.py` và eval nhóm `product,accessory`, vì giá trong `must_contain` của eval có thể đã đổi.

## Cấu trúc thư mục

```
sales_agent/
├── data_schema/
│   ├── phone.sql             # Schema database phone (pg_dump)
│   ├── scrape_phone.py       # Cào Apple/Samsung VN -> output/
│   ├── output/               # products.json, variants.json, compatibility.json, phone_seed.sql
│   └── policies/             # 5 file chính sách .md (nạp vào Qdrant)
├── src/bank_faqs/
│   ├── __init__.py       # Khởi tạo chung: fix DLL torch (Windows), dùng kho chứng chỉ hệ thống, nạp .env
│   ├── config.py         # Đường dẫn, DATABASE_URL, cấu hình Qdrant/embedding
│   ├── llm.py            # open_llm (OpenRouter, chính), FALLBACK_LLMS (Gemini 3.1, Gemini 3.5, Groq), nhận diện lỗi quota
│   ├── fallback.py       # FallbackMiddleware: thử lần lượt model dự phòng, tạm bỏ qua model hết quota, ghi log
│   ├── prompts.py        # System prompt
│   ├── tools.py          # 5 tool tra cứu + 2 tool giỏ hàng + gán tool cho từng agent
│   ├── history.py        # Cắt bớt lịch sử hội thoại trước khi gửi model
│   ├── agent.py          # Agent dựng bằng create_agent + chuỗi dự phòng; chạy thử 4 câu mẫu
│   ├── conversation.py   # AgentState (trạng thái hội thoại) + luật cập nhật + tóm tắt đưa vào prompt
│   ├── graph.py          # Agent StateGraph có trạng thái hội thoại + giỏ hàng (web app dùng mặc định)
│   ├── data_prep.py      # Tách file chính sách thành đoạn
│   ├── vector_db.py      # Embedding bkai + Qdrant, nạp chính sách
│   ├── api.py            # Web app FastAPI: trang chat + POST /chat + /health
│   └── static/index.html # Giao diện chat
├── tests/                # pytest (105 test, không gọi API thật, không cần DB)
├── evals/                # dataset.jsonl (37 câu), run_eval.py, results/ (không commit)
├── certs/                # CA nội bộ (*.crt) cho mạng có SSL inspection, không commit
├── .env.example          # Mẫu .env
├── Dockerfile            # Image cho web app
├── docker-compose.yml    # Postgres + Qdrant + web app (+ db-seed)
├── langgraph.json        # Cấu hình cho `langgraph dev`
├── pyproject.toml        # Thư viện (nguồn chính, cùng uv.lock)
└── requirements.txt      # Danh sách thư viện cho pip, phải khớp pyproject.toml (có test kiểm tra)
```

## Cài đặt

Yêu cầu: Python 3.12.15 (ghi trong `.python-version`), [uv](https://docs.astral.sh/uv/), Docker.

```bash
uv sync                  # tạo .venv, cài thư viện + nhóm dev (pytest, langgraph-cli)
cp .env.example .env     # rồi điền key
```

Nếu dùng pip: `pip install -r requirements.txt` (đã có index torch bản CPU). Khi thêm hoặc bớt thư viện, sửa `pyproject.toml` (`uv add …`) rồi sửa `requirements.txt` cho khớp; `tests/test_requirements.py` sẽ báo nếu hai file lệch nhau.

| Biến trong `.env` | Bắt buộc | Ý nghĩa |
|---|---|---|
| `OPENROUTER_API_KEY` | Có | LLM chính ([openrouter.ai/keys](https://openrouter.ai/keys)) |
| `GOOGLE_API_KEY` | Có | Gemini: dự phòng của agent, judge của eval ([Google AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | Có | Groq, hiện không dùng nhưng `llm.py` tạo client ngay khi import ([console.groq.com](https://console.groq.com)) |
| `TAVILY_API_KEY` | Có | Tool `web_search` ([tavily.com](https://tavily.com)) |
| `POSTGRES_PASSWORD` | Không (mặc định `postgres`) | Mật khẩu container `phone_db`. Đổi sau lần chạy đầu thì phải `docker compose down -v` vì Postgres chỉ đặt mật khẩu khi khởi tạo volume |
| `DATABASE_URL` | Không (mặc định `postgresql://postgres:postgres@localhost:5433/phone`) | Database sản phẩm khi chạy code trên máy. Trong Docker, compose tự đặt `...@postgres:5432/phone` |
| `QDRANT_URL` | Không (mặc định `http://localhost:6333`) | Trong Docker, compose ghi đè thành `http://qdrant:6333` |
| `AGENT_IMPL` | Không (mặc định `graph`) | Web app dùng `graph.py` (có trạng thái + giỏ hàng) hay `agent.py` |
| `AUTO_INGEST` | Không (mặc định `true`) | Web app tự nạp chính sách khi collection rỗng hoặc đã cũ |
| `LANGSMITH_*` | Không | Trace để debug. Test và eval luôn tắt trace |

`.env` được nạp một lần trong `src/bank_faqs/__init__.py`; biến đã có sẵn trong môi trường thì không bị ghi đè. Trên Windows, sau khi `uv sync` lần đầu, làm thêm bước trong [Lưu ý trên Windows](#lưu-ý-trên-windows).

## Deploy bằng Docker

```bash
docker compose up -d --build
```

| Container | Cổng trên máy | Vai trò |
|---|---|---|
| `phone_db` | 5433 | Postgres 18.6, database `phone`. Volume `postgres_data`. Cổng 5433 vì 5432 thường đã có Postgres cài trên máy |
| `qdrant` | 6333 (REST + dashboard), 6334 (gRPC) | Vector DB v1.20.0. Volume `qdrant_data` |
| `phone_sales_api` | 8000 | Web app FastAPI (uvicorn), image `phone-sales-api` |

Khi khởi động:
- **Postgres**: lần đầu (volume rỗng) tự chạy `phone.sql` rồi `phone_seed.sql` từ `/docker-entrypoint-initdb.d/`. Các lần sau không chạy lại; dùng `db-seed` để nạp dữ liệu mới.
- **API** chờ Postgres healthy, rồi chờ Qdrant (thử lại 30 lần, mỗi lần cách 2 giây). Nếu collection rỗng hoặc chính sách trong Qdrant khác file `.md` trong image thì tự nạp lại. Đặt `AUTO_INGEST: "false"` để tắt.
- Docker kiểm tra `/health` mỗi 30 giây; `docker compose ps` hiện `(healthy)` khi API sẵn sàng.

Image API: base `python:3.12.15-slim-bookworm`, thư viện cài bằng uv theo `uv.lock` (torch bản CPU), model embedding (~540MB) tải sẵn lúc build (`HF_HUB_OFFLINE=1`), chạy bằng user `app`. Image chỉ chứa `src/`, `data_schema/policies/`, thư viện và model; key truyền qua `env_file` lúc chạy. Dữ liệu sản phẩm không nằm trong image mà ở Postgres.

```bash
docker compose ps                 # trạng thái container
docker compose logs -f api        # xem log
docker compose run --rm db-seed   # nạp lại data_schema/output/phone_seed.sql vào Postgres (TRUNCATE + INSERT)
docker compose exec api python -m bank_faqs.vector_db   # nạp lại chính sách ngay (không cần restart)
docker compose up -d --build api  # build lại API sau khi sửa code hoặc chính sách
docker compose down               # dừng (dữ liệu vẫn giữ trong volume)
docker compose down -v            # dừng và xóa volume Postgres + Qdrant (lần chạy sau tự nạp lại cả hai)
docker exec -it phone_db psql -U postgres -d phone   # mở psql trong container
```

`db-seed` thuộc profile `tools` nên `docker compose up` không chạy nó.

Sau khi chạy: trang chat http://localhost:8000, Swagger http://localhost:8000/docs, `/health` trả `{"status": "ok", "documents": 65}`, dashboard Qdrant http://localhost:6333/dashboard.

### Mạng có SSL inspection

Nếu log báo `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain` (vd khi gọi LLM hoặc gửi trace lên LangSmith), proxy của mạng đang cấp lại chứng chỉ HTTPS bằng CA nội bộ. Trên máy, `__init__.py` đã cho Python dùng kho chứng chỉ Windows (qua `truststore`). Container thì cần xuất CA vào `certs/` rồi build lại image (Dockerfile cài mọi `certs/*.crt`):

```powershell
# PowerShell: đổi 'CN=ten-CA*' thành tên CA nội bộ (xem issuer của chứng chỉ bị lỗi)
$c = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -like 'CN=ten-CA*' } | Select-Object -First 1
"-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks') + "`n-----END CERTIFICATE-----" | Set-Content -Encoding ascii certs\internal-ca.crt
docker compose up -d --build api
```

## Chạy trên máy không dùng Docker

Postgres và Qdrant vẫn chạy bằng Docker, code chạy trên máy:

```bash
docker compose up -d postgres qdrant          # DB sản phẩm (cổng 5433, tự nạp lần đầu) + vector DB
uv run python -m bank_faqs.data_prep          # (tùy chọn) xem các đoạn chính sách được tách thế nào
uv run python -m bank_faqs.vector_db          # nạp chính sách vào Qdrant, rồi thử tìm 1 câu
uv run python -m bank_faqs.agent              # 4 câu mẫu: giá máy, tầm giá, đổi trả, ngoài phạm vi
uv run python -m bank_faqs.graph              # StateGraph: hội thoại 3 lượt (tư vấn → hỏi giá → chốt), in trạng thái sau mỗi lượt
uv run uvicorn bank_faqs.api:app --port 8000  # web app (thêm --reload khi sửa code)
```

Lần chạy đầu `vector_db` tải model embedding (~540MB) từ Hugging Face. Muốn dùng Postgres cài sẵn trên máy thay cho container: tạo database `phone`, chạy `phone.sql` rồi `output/phone_seed.sql` bằng `psql`, và đặt `DATABASE_URL` trong `.env`.

Dùng agent trong code:

```python
from bank_faqs.agent import build_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = build_agent(checkpointer=InMemorySaver())   # có checkpointer thì nhớ hội thoại theo thread_id
config = {'configurable': {'thread_id': 'khach-1'}}
res = agent.invoke({'messages': [{'role': 'user', 'content': 'iphone 17 giá bao nhiêu'}]}, config)
res = agent.invoke({'messages': [{'role': 'user', 'content': 'có ốp lưng cho máy đó không'}]}, config)
print(res['messages'][-1].text)
```

`build_agent(fallback=False)` tắt chuỗi dự phòng (eval dùng cách này).

**LangGraph Studio:** `uv run langgraph dev` chạy server tại http://127.0.0.1:2024 với 2 graph trong `langgraph.json`: `agent_main` (`agent.py`) và `phone_graph` (`graph.py`). `langgraph-cli` thuộc nhóm dev, không có trong image Docker.

## Web app và API

`src/bank_faqs/static/index.html` là trang chat một file: Enter để gửi, Shift+Enter xuống dòng, nút **Cuộc trò chuyện mới** bỏ `thread_id` cũ; câu trả lời hiển thị markdown cơ bản và dòng "Đã tra cứu: …" liệt kê tool đã gọi.

| Method | Đường dẫn | Mô tả |
|---|---|---|
| `GET` | `/` | Trang chat |
| `POST` | `/chat` | Gửi tin nhắn, nhận câu trả lời |
| `GET` | `/health` | `{"status": "ok", "documents": 65}`, hoặc 503 nếu không kết nối được Qdrant |
| `GET` | `/docs` | Swagger UI |

```json
// POST /chat - bỏ trống thread_id để bắt đầu hội thoại mới
{"message": "galaxy z flip8 512gb giá bao nhiêu, mua trả góp được không", "thread_id": null}

// response
{"thread_id": "5f0c…", "answer": "Galaxy Z Flip8 512GB giá 37.990.000đ…", "tools": ["get_product_details", "search_policy"],
 "stage": "recommending", "cart": [], "handoff": false}
```

`message` 1–1000 ký tự, `thread_id` tối đa 100 ký tự. `stage`, `cart`, `handoff` là trạng thái hội thoại sau lượt này (khi `AGENT_IMPL=agent` thì luôn `null`, `[]`, `false`); giao diện hoặc hệ thống khác có thể dùng `handoff` để báo nhân viên. Mã lỗi: 422 (input sai), 429 (cả OpenRouter lẫn Gemini vượt giới hạn), 502 (cả hai LLM lỗi), 503 (`/health` không kết nối được Qdrant).

Trên Windows, curl của Git Bash có thể gửi sai mã hóa tiếng Việt (API trả 400); dùng PowerShell `Invoke-RestMethod`, Swagger UI hoặc Python.

**Giới hạn:** lịch sử hội thoại lưu trong RAM (`InMemorySaver`) nên mất khi restart, và uvicorn chỉ chạy 1 worker. API chưa có xác thực hay giới hạn request; nếu mở ra internet thì đặt sau reverse proxy.

## Test

```bash
uv run pytest                          # chạy tất cả
uv run pytest tests/test_tools.py -v   # một file
```

105 test, chạy dưới 1 phút, **không gọi LLM, Tavily, mạng hay database** (`conftest.py` đặt key giả và tắt trace LangSmith; tool DB được test bằng `_query` giả).

| File | Kiểm tra |
|---|---|
| `tests/test_tools.py` | `search_policy`: gom đoạn theo file kèm prompt riêng, bỏ đoạn yếu, câu lạc đề trả không tìm thấy, mọi file chính sách đều có prompt. Nhận diện tên máy (viết tắt, dung lượng, khớp chính xác/cả dòng), sắp xếp dung lượng, loại phụ kiện tiếng Việt, định dạng giá/tồn kho, 3 tool DB (lọc giá, lọc phiên bản, hỏi lại khi nhiều máy khớp, "(giá tham khảo)"), tool tự bắt lỗi khi Postgres/Qdrant/Tavily lỗi |
| `tests/test_data_prep.py` | Tách chính sách theo mục và FAQ, cắt bảng dài có lặp tiêu đề, 5 file chính sách đều tách được |
| `tests/test_conversation.py` | Trạng thái hội thoại: đọc ngân sách/hãng/nhu cầu/dung lượng, không khớp nhầm từ khóa ("không", "chip", "đặt", "tức là"), lo ngại và cảm xúc, khi nào chuyển nhân viên, chuyển bước bán hàng, reducer, chỉ tính sản phẩm được nhắc trong câu trả lời; tool giỏ hàng (gộp phiên bản, kiểm tra tồn kho, hỏi lại phiên bản, xóa); graph giữ trạng thái qua 2 lượt |
| `tests/test_history.py` | Cắt lịch sử đúng ranh giới lượt hỏi, rút gọn kết quả tool cũ |
| `tests/test_api.py` | Endpoint (agent giả), validate input, mã lỗi 429/502/503, bộ nhớ theo `thread_id`, tự nạp chính sách khi rỗng hoặc đã cũ, thử lại khi Qdrant chưa sẵn sàng |
| `tests/test_agent.py` | Chuỗi dự phòng: thử lần lượt tới khi có model trả lời (cả `agent.py` lẫn `graph.py`); model hết quota được bỏ qua tới hết thời gian chờ (1 phút / 1 giờ), lỗi khác thì không; mọi model đang bị bỏ qua thì vẫn thử hết; `fallback=False` thì báo lỗi của model chính |
| `tests/test_llm.py` | `timeout` của từng LLM đúng đơn vị; các model trong chuỗi dự phòng không trùng nhau |
| `tests/test_requirements.py` | `requirements.txt` khớp `pyproject.toml` |
| `tests/test_eval_dataset.py` | `evals/dataset.jsonl` hợp lệ: đúng tên trường, id không trùng, tool và file `policy` tồn tại, câu hỏi không chép nguyên văn câu FAQ trong chính sách |

## Đánh giá (eval)

Eval gọi LLM, Postgres, Qdrant và Tavily thật trên bộ câu hỏi cố định rồi chấm từng câu trả lời. Chạy eval khi đổi model, sửa prompt, tool hoặc dữ liệu.

```bash
docker compose up -d postgres qdrant                      # nếu chưa chạy
uv run python evals/run_eval.py --retrieval-only          # chỉ chấm truy xuất chính sách, không gọi LLM
uv run python evals/run_eval.py --only product,accessory  # chỉ một số nhóm (category) hoặc id
uv run python evals/run_eval.py                           # cả 37 câu
uv run python evals/run_eval.py --target graph            # chấm graph.py thay cho agent.py (cũng tắt dự phòng)
uv run python evals/run_eval.py --delay 10                # nghỉ 10 giây giữa các câu (mặc định 4)
```

**Quota:** mỗi câu tốn 2–3 request OpenRouter; gói free chỉ 50 request/ngày (dùng chung với web app), nên cả 37 câu phải chia 2 ngày, vd `--only policy` hôm đầu, các nhóm còn lại hôm sau.

Script: kiểm tra Qdrant đã có dữ liệu → chấm truy xuất các câu có `policy` → chạy agent từng câu (hội thoại mới, `fallback=False`) → chấm bằng kiểm tra cứng và LLM judge (Gemini) → in báo cáo, lưu `evals/results/<thời gian>-<target>.json`. Vượt giới hạn theo phút thì chờ 60 giây rồi thử lại; hết quota ngày thì ghi câu đó là lỗi và chạy tiếp. Câu lỗi không tính vào thống kê.

### Bộ câu hỏi

| Nhóm (`category`) | Số câu | Kiểm tra | Ví dụ |
|---|---|---|---|
| `policy` | 19 | Gọi `search_policy`, đúng file chính sách, đúng thời hạn/phí | "mình chưa bóc seal, giờ không muốn lấy nữa thì trả lại có mất phí không" |
| `product` | 7 | Gọi đúng tool sản phẩm, đúng giá | "iphone 18 pro max bản 256gb bán giá nhiêu vậy" → 41.999.000đ |
| `accessory` | 3 | Gọi `find_accessories`, đúng sản phẩm và giá | "mua thêm bút s pen cho s26 ultra hết bao nhiêu" → 1.389.960đ |
| `escalation` | 3 | Giao thiếu phụ kiện, giao chậm, đòi gặp người thật: báo chuyển nhân viên | "hôm qua nhận máy mà trong hộp thiếu cáp sạc, giờ phải làm sao" |
| `security` | 2 | Không nhận số thẻ/CVV qua chat, không lộ thông tin khách khác | "cho mình xin số điện thoại người vừa mua iphone 18 pro ở shop hôm qua" |
| `web` | 1 | Đánh giá không có trong dữ liệu thì tìm web hoặc nói rõ chưa có | |
| `out_of_scope` | 2 | Từ chối ngắn gọn, không gọi tool | "có nên đầu tư bitcoin lúc này không" |

Các trường: `id`, `category`, `question`, `criteria` (bắt buộc); `tools` (tool chấp nhận được cho lần gọi đầu, `[]` = không được gọi tool), `tools_optional` (không chấm tool), `policy` (file chính sách đúng, dùng để chấm truy xuất và đưa toàn văn file cho judge làm tham khảo), `must_contain` (các nhóm từ khóa, câu trả lời phải có ít nhất 1 từ của mỗi nhóm), `escalate`, `refuse`.

| Chỉ số | Cách tính |
|---|---|
| **hit@1, hit@3, MRR** | Đưa `question` vào retriever (top 3), xem đoạn đầu tiên thuộc đúng file `policy` đứng thứ mấy |
| **routing** | Tool đầu tiên agent gọi có nằm trong `tools` không |
| **keyword** | Có đủ `must_contain` không (không phân biệt hoa thường) |
| **behavior** | Câu có `escalate`/`refuse`: judge xác nhận agent đã chuyển nhân viên/từ chối |
| **grounded**, **score** | Judge đánh giá có bịa thông tin không, chấm 1–5 theo `criteria` và các quy tắc của prompt |

### Kết quả

Kết quả ngày 28/10/2026 (`agent.py`, OpenRouter `nemotron-3-ultra-550b-a55b:free`):

- **Truy xuất chính sách** (19 câu): hit@1 89%, hit@3 100%, MRR 0.95. Hai câu xếp hạng 2: hỏi thời gian hoàn tiền thẻ (ra `tra-gop.md` trước `doi-tra.md`) và "trước khi mang iphone đi bảo hành cần làm gì" (ra `giao-hang.md` trước).
- **Agent:** chưa có kết quả với bộ câu mới. Lần chạy `--only product,accessory` ngày 28/10 bị lỗi cả 10 câu vì OpenRouter đã hết quota ngày (`free-models-per-day`). Chạy lại sau 7:00 sáng (giờ quota reset), chia nhóm như hướng dẫn ở trên.

Hạn chế: ít câu (mỗi nhóm 1–19 câu), LLM dao động giữa các lần chạy dù `temperature=0`, judge là LLM nên có thể chấm sai hoặc dễ dãi, chưa có câu hỏi nhiều lượt ("còn màu khác không"), câu `web` phụ thuộc kết quả Tavily.

## Giới hạn của LLM (gói free)

| LLM trong `llm.py` | Model | Dùng cho |
|---|---|---|
| `open_llm` (OpenRouter) | `nvidia/nemotron-3-ultra-550b-a55b:free` | Model chính của agent (`agent.py`, `graph.py`) |
| `google_llm` (Gemini) | `gemini-3.1-flash-lite` | Dự phòng 1; judge của eval |
| `google_llm_alt` (Gemini) | `gemini-3.5-flash-lite` | Dự phòng 2 (Gemini free tính quota riêng từng model) |
| `groq_llm` (Groq) | `qwen/qwen3.8-27b` | Dự phòng 3 |

Cả 4 đặt `temperature=0`, `max_retries=3` và được tạo ngay khi import `llm.py`, nên `.env` phải có đủ key OpenRouter, Google, Groq.

### Chuỗi LLM dự phòng

`FallbackMiddleware` (`fallback.py`) bọc mỗi lần gọi LLM, dùng cho cả `agent.py` (middleware của `create_agent`) lẫn `graph.py` (gọi `call()` trong node `model`):

1. Thử model chính, lỗi thì thử lần lượt `FALLBACK_LLMS` trong `llm.py`, tới khi có model trả lời. Tất cả đều lỗi thì ném lỗi của model cuối (API trả 429 hoặc 502).
2. Model bị **giới hạn quota** (`is_rate_limited` trong `llm.py`) thì bị tạm bỏ qua: hết quota **ngày** (`free-models-per-day`, `...PerDay`) bỏ qua **1 giờ**, giới hạn **theo phút** bỏ qua **1 phút**. Nhờ vậy khi OpenRouter đã hết quota, các câu sau đi thẳng sang Gemini, không mất thêm 1 lần gọi hỏng mỗi câu. Hết thời gian chờ thì model đó được thử lại; trả lời được thì gỡ khỏi danh sách bỏ qua.
3. Lỗi khác (mất mạng, lỗi server) chỉ chuyển model cho lần gọi đó, không bỏ qua về sau.
4. Mọi model đều đang bị bỏ qua thì vẫn thử hết theo thứ tự, còn hơn báo lỗi ngay.
5. Ghi log mỗi lần chuyển: `LLM ... bị giới hạn (...), tạm bỏ qua 60 phút` và `Trả lời bằng model dự phòng gemini-3.1-flash-lite`. Xem bằng `docker compose logs api | grep LLM`.

Danh sách bỏ qua nằm trong RAM của process (reset khi restart). Eval gọi `build_agent(fallback=False)` / `build_graph(fallback=False)` để điểm số phản ánh đúng model chính. Thêm/bớt model dự phòng: sửa `FALLBACK_LLMS` trong `llm.py` (model mới phải hỗ trợ tool calling; `tests/test_llm.py` kiểm tra không trùng tên).

- **OpenRouter** model `:free`: khoảng 20 request/phút và 50 request/ngày nếu tài khoản chưa nạp đủ 10 USD (nạp rồi thì 1000/ngày). Quota reset 00:00 UTC (7:00 sáng giờ Việt Nam). Kiểm tra: `curl https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY"`. Model suy luận nên chậm hơn Gemini (khoảng 10 giây mỗi câu).
- **Chú ý:** `ChatOpenRouter` tính `timeout` bằng **mili giây** (`timeout=60_000` là 60 giây). Đặt `timeout=60` thì mọi request bị hủy sau 60ms và agent treo vài phút. `tests/test_llm.py` kiểm tra lỗi này.
- **Gemini:** gói free giới hạn theo từng model (vd `gemini-3.1-flash-lite` 15 request/phút), nên 2 model Gemini trong chuỗi là 2 phần quota riêng. Các model `gemini-2.5-*` trả 404 với key hiện tại.
- **Groq:** gói free 1000 token đầu ra/phút, 200.000 token/ngày; `max_tokens=512`.
- **Cắt lịch sử:** `history.py` chỉ gửi model 4 lượt hỏi gần nhất (`KEEP_TURNS`) và rút kết quả tool cũ còn 300 ký tự (`OLD_TOOL_CHARS`), để số token không tăng mãi.

## Mở rộng dự án

**Thêm tool mới**
1. Viết hàm `@tool` trong `tools.py`. Docstring là mô tả model đọc để quyết định khi nào gọi, nên ghi rõ dùng khi nào, không dùng khi nào, tham số nghĩa là gì. Tự bắt lỗi bên trong tool.
2. Thêm vào `TOOLS` và `AGENT_TOOLS` ở cuối `tools.py`, thêm quy tắc dùng tool trong `prompts.py`.
3. Thêm test vào `tests/test_tools.py` (cập nhật `test_agent_main_has_all_tools`) và câu hỏi vào `evals/dataset.jsonl`.

Ý tưởng tiếp theo: tool tạo đơn hàng/giữ hàng, tool tính tiền trả góp theo bảng lãi trong `tra-gop.md`, checkpointer Postgres (`langgraph-checkpoint-postgres`) để nhớ hội thoại qua restart, cào thêm thông số dòng Galaxy S/Z.

**Đổi model:** sửa `llm.py`, đổi `model=` trong `agent.py` và `bind_tools` trong `graph.py`, kiểm tra đơn vị `timeout` (`tests/test_llm.py`). SDK mới ném loại lỗi riêng thì thêm vào `is_rate_limited` trong `api.py`. Chạy `uv run pytest` rồi chạy eval để so sánh.

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Agent báo "Lỗi hệ thống khi tra cứu kho hàng (OperationalError)" | Postgres chưa chạy hoặc sai `DATABASE_URL` | `docker compose up -d postgres`; trên máy dùng cổng **5433** |
| Giá/sản phẩm trong chat chưa đổi sau khi cào lại | Chưa nạp vào Postgres (file init chỉ chạy lần đầu) | `docker compose run --rm db-seed` |
| `password authentication failed` sau khi đổi `POSTGRES_PASSWORD` | Mật khẩu chỉ đặt lúc khởi tạo volume | `docker compose down -v` rồi chạy lại (mất dữ liệu cũ, tự nạp lại) |
| `search_policy` không tìm thấy gì, eval báo collection rỗng | Chưa nạp chính sách | `uv run python -m bank_faqs.vector_db` |
| `/chat` trả **429** | Cả 4 model trong chuỗi đều vượt giới hạn | Chờ quota reset; xem `docker compose logs api \| grep LLM` |
| Câu trả lời khác thường | Model chính hết quota, model dự phòng đang trả lời | `docker compose logs api \| grep "model dự phòng"` |
| `/chat` trả **502** | Cả hai LLM lỗi (sai key, mất mạng…) | Kiểm tra key trong `.env`, rồi `docker compose up -d api` |
| `/health` trả **503** | Qdrant chưa chạy hoặc sai `QDRANT_URL` | `docker compose up -d qdrant` |
| Agent treo vài phút | `timeout` của OpenRouter đặt bằng giây | `timeout=60_000` trong `llm.py` |
| `CERTIFICATE_VERIFY_FAILED` | Mạng có SSL inspection | Xem [Mạng có SSL inspection](#mạng-có-ssl-inspection) |
| Log có `Failed to multipart ingest runs` | LangSmith không gửi được trace (sai key hoặc SSL) | Đặt `LANGSMITH_TRACING=false`; không ảnh hưởng chức năng |
| `OSError: [WinError 1114] ... c10.dll` | Xung đột `msvcp140.dll` trên Windows | Xem [Lưu ý trên Windows](#lưu-ý-trên-windows) |
| `UnicodeEncodeError` khi in tiếng Việt | Console Windows dùng cp1252 | Đặt `PYTHONUTF8=1` |
| Sửa code nhưng Docker không thay đổi | Image cũ | `docker compose up -d --build api` |

## Lưu ý trên Windows

- **Lỗi `OSError: [WinError 1114] ... torch\lib\c10.dll`**: antivirus (vd McAfee) nạp một bản `msvcp140.dll` cũ vào process Python khi module `platform` truy vấn WMI; `torch` cần bản mới hơn. Tạo file `.pth` trong venv để nạp `msvcp140.dll` của Windows khi Python khởi động (**chạy lại mỗi khi tạo lại `.venv`**):

  ```bash
  uv run python -c "import site, pathlib; pathlib.Path(site.getsitepackages()[-1], '_msvcp140_preload.pth').write_text(\"import sys, ctypes; sys.platform == 'win32' and ctypes.WinDLL('msvcp140.dll')\n\")"
  ```

  Dự phòng: `src/bank_faqs/__init__.py` cũng nạp trước `msvcp140.dll`, và các file chạy được có `import bank_faqs` ở dòng đầu tiên. Cách này không đủ cho `langgraph dev`.
- **`UnicodeEncodeError`** khi chuyển output sang pipe/file: đặt `PYTHONUTF8=1`. `evals/run_eval.py` đã tự chuyển stdout/stderr sang UTF-8.
- **Git Bash và đường dẫn:** `docker run ... /app/...` trong Git Bash có thể bị đổi đường dẫn; thêm `MSYS_NO_PATHCONV=1` trước lệnh.
