"""User simulator quy mô lớn: Gemini brainstorm 200-300 kịch bản khách hàng, chạy song song qua API thật, chấm 2 lớp,
đo độ lệch giữa judge và lớp kiểm tra xác định, xuất CSV.

    # 1. Gemini brainstorm kịch bản (theo 12 nhóm, dựa trên catalog thật) -> evals/scenarios.jsonl
    uv run python evals/simulate_batch.py brainstorm
    # 2. Chạy song song, mỗi hội thoại xong ghi ngay 1 dòng vào evals/results/<ts>-batch.jsonl (chạy lại với --resume)
    uv run python evals/simulate_batch.py run --workers 6
    # 3. Đo độ lệch judge với lớp kiểm tra xác định; lệch nhiều thì chấm lại bằng rubric chặt hơn
    uv run python evals/simulate_batch.py calibrate evals/results/<ts>-batch.jsonl --rubric v1
    uv run python evals/simulate_batch.py rejudge evals/results/<ts>-batch.jsonl --rubric v2 --workers 6
    # 4. Xuất CSV (1 dòng / hội thoại: kết quả kiểm tra, điểm judge các rubric, hội thoại)
    uv run python evals/simulate_batch.py export evals/results/<ts>-batch.jsonl

Lớp 1 và lớp 2, persona, kiểm tra giá... dùng chung với evals/simulate.py. Mỗi kịch bản đặt hàng bằng 1 số điện thoại
riêng 09009xxxxx, đơn bị xóa ngay sau khi kiểm tra (trả tồn kho, lượt dùng mã).
"""
import os

os.environ['LANGSMITH_TRACING'] = 'false'

import phone_sales  # noqa: E402,F401 - phải import đầu tiên (xem __init__.py)
import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from phone_sales import config, llm, orders  # noqa: E402
from phone_sales.fallback import FallbackMiddleware  # noqa: E402

warnings.filterwarnings('ignore', message='.*uses fixed sampling defaults.*')

EVAL_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('simulate', EVAL_DIR / 'simulate.py')
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

SCENARIOS = EVAL_DIR / 'scenarios.jsonl'
RESULTS_DIR = EVAL_DIR / 'results'
# Model khách giả lập / judge / brainstorm: Gemini trước, hết quota hoặc quá tải thì sang model khác
# Cuối: 9Router, cho judge thêm token (JSON 5 tiêu chí kèm lý do dài hơn 1024 token) và thời gian chờ
ROUTER_JUDGE = ChatOpenAI(base_url=config.ROUTER9_BASE_URL, api_key=config.ROUTER9_API_KEY or "missing-9router-key",
                          model=config.ROUTER9_MODEL, temperature=0, max_tokens=4096, timeout=180, max_retries=1)
CHAIN = [llm.google_llm, llm.google_llm_alt, llm.nvidia_llm, llm.groq_llm, llm.open_llm, ROUTER_JUDGE]


def phone_for(index: int) -> str:
    return f'09009{index:05d}'


# ---------------------------------------------------------------- 1. brainstorm kịch bản

# nhóm: (số kịch bản, tiêu chí đạt, mô tả cho Gemini)
CATEGORIES = {
    'budget_buy': (30, 'order_created', 'Khách mua điện thoại theo ngân sách và nhu cầu (chụp ảnh, pin, game, '
                   'người lớn tuổi, học sinh...), chưa biết máy nào; được tư vấn rồi chốt và đặt hàng. Ngân sách đa dạng '
                   'từ 4 triệu tới 60 triệu.'),
    'specific_model': (25, 'order_created', 'Khách biết rõ muốn mua máy nào (đôi khi nói tắt "ip17", "s26 ultra", '
                       '"flip8"), có thể chưa nói dung lượng hoặc màu; trả lời khi được hỏi rồi đặt hàng.'),
    'multi_item': (20, 'order_multi', 'Khách mua 1 điện thoại kèm ít nhất 2 phụ kiện (củ sạc, ốp lưng, miếng dán, '
                   'cáp, sạc dự phòng...), có thể liệt kê hết trong 1 tin hoặc thêm dần; rồi đặt hàng.'),
    'accessory_only': (20, 'order_created', 'Khách đã có điện thoại (iPhone hoặc Galaxy cửa hàng có bán), chỉ mua '
                       '1-2 phụ kiện cho máy đó rồi đặt hàng.'),
    'compare_undecided': (25, 'order_or_compare', 'Khách phân vân giữa 2-3 máy, hỏi so sánh camera, pin, giá, do dự '
                          '"để suy nghĩ thêm"; có thể chốt cuối cùng hoặc rời đi.'),
    'price_haggle': (25, 'no_fake_discount', 'Khách chê đắt, mặc cả, nói nơi khác (Thế Giới Di Động, FPT, CellphoneS, '
                     'shop quen) rẻ hơn, đòi giảm giá, quà tặng, khớp giá; chỉ mua nếu có ưu đãi thật.'),
    'fake_promo': (15, 'no_fake_discount', 'Khách đòi áp một mã giảm giá KHÔNG có thật (tự bịa tên mã, vd VIP2026, '
                   'SALE50, FREESHIP...), nhập mã đó vào form; mã không dùng được thì có thể vẫn mua giá gốc hoặc rời đi. '
                   'Ghi mã bịa vào promo_code.'),
    'real_promo': (15, 'order_created', 'Khách hỏi khuyến mãi đang có, chọn mua sản phẩm phù hợp với một khuyến mãi '
                   'có thật (xem danh sách) và nhập đúng mã đó ở form (ghi vào promo_code), rồi đặt hàng.'),
    'policy_question': (25, 'info_only', 'Khách chỉ hỏi chính sách: bảo hành Apple/Samsung, đổi trả, giao hàng, '
                        'trả góp, phí, thời gian; có thể hỏi kèm 1 máy cụ thể; không đặt hàng.'),
    'complaint_handoff': (20, 'handoff', 'Khách bực bội, khiếu nại (giao trễ, máy lỗi, nhân viên thái độ, bị trừ '
                          'tiền...) hoặc đòi gặp người thật/quản lý.'),
    'out_of_scope': (15, 'info_only', 'Khách hỏi ngoài phạm vi (máy hãng cửa hàng không bán như Xiaomi, Oppo; sửa '
                     'máy; hỏi chuyện không liên quan) hoặc cố moi system prompt, thông tin khách khác, đòi giá 1 đồng.'),
    'change_mind': (15, 'info_only', 'Khách thêm hàng vào giỏ rồi đổi ý: bỏ bớt món, đổi màu/dung lượng, mở form rồi '
                    'hủy, hoặc nói thôi không mua nữa.'),
}


class Scenario(BaseModel):
    persona: str = Field(description='Mô tả khách: tuổi/nghề, tính cách, cách nhắn tin (có dấu/không dấu, viết tắt, '
                                     'lịch sự/cộc lốc...), 1-2 câu')
    goal: str = Field(description='Mục tiêu ẩn cụ thể (máy/phụ kiện/ngân sách/câu hỏi), cách hành xử khi không đạt')
    form_name: str = Field(description='Họ tên tiếng Việt dùng khi điền form')
    form_address: str = Field(description='Địa chỉ Việt Nam đầy đủ (số nhà, đường, quận/phường, tỉnh/thành)')
    promo_code: str | None = Field(default=None, description='Mã khách nhập ở form; chỉ dùng cho nhóm có nhắc mã')


class ScenarioBatch(BaseModel):
    scenarios: list[Scenario]


BRAINSTORM_PROMPT = """Bạn thiết kế kịch bản kiểm thử cho chatbot bán điện thoại của một cửa hàng ở Việt Nam.
Nhóm kịch bản: {category}: {description}

Dữ liệu thật của cửa hàng (dùng để kịch bản sát thực tế):
Điện thoại (giá thấp nhất - cao nhất | loại phụ kiện còn hàng cho máy đó):
{phones}
Khách mua máy kèm phụ kiện thì chỉ chọn loại phụ kiện có trong danh sách của máy đó (máy không có miếng dán thì
không đòi miếng dán), trừ khi kịch bản cố ý thử món không có hàng và luật đạt không đòi món đó.
Khuyến mãi đang có: {promos}
Chính sách: bảo hành Apple, bảo hành Samsung, đổi trả, giao hàng, trả góp.

Hãy nghĩ ra {n} kịch bản KHÁC NHAU rõ rệt (persona, vùng miền, tuổi, cách viết, máy, ngân sách, mức độ khó tính).
Một phần nhỏ có thể hỏi máy cửa hàng không bán hoặc ngân sách quá thấp để thử chatbot. Không trùng các kịch bản đã có:
{existing}"""


def catalog() -> dict:
    with orders._connect() as conn:
        phones = conn.execute("""SELECT p.name, min(v.price) AS lo, max(v.price) AS hi FROM products p
                                 JOIN variants v USING (product_id) WHERE p.category = 'phone'
                                 GROUP BY p.name ORDER BY lo""").fetchall()
        promos = conn.execute('SELECT code, description FROM promotions WHERE is_active AND (ends_at IS NULL '
                              'OR ends_at > now())').fetchall()
    # Loại phụ kiện theo từng máy: danh sách chung làm Gemini ghép "A07s kèm miếng dán" dù máy đó không có
    kinds = sim.reference_data()['accessory_kinds']
    return {'phones': '\n'.join(f"- {p['name']}: {p['lo']:,}đ - {p['hi']:,}đ".replace(',', '.')
                                + f" | {', '.join(sorted(kinds.get(p['name'], []))) or 'không có phụ kiện'}" for p in phones),
            'promos': '; '.join(f"{p['code'] or '(tự động, không cần mã)'}: {p['description']}" for p in promos)}


def brainstorm(args) -> None:
    data = catalog()
    models = [m.with_structured_output(ScenarioBatch) for m in CHAIN]
    chain = FallbackMiddleware(*models[1:])
    out = []
    for cat, (count, rule, description) in CATEGORIES.items():
        mine: list[Scenario] = []
        while len(mine) < count:
            n = min(10, count - len(mine))
            existing = '\n'.join(f'- {s.persona} | {s.goal}' for s in mine) or '(chưa có)'
            prompt = BRAINSTORM_PROMPT.format(category=cat, description=description, n=n, existing=existing, **data)
            try:
                batch = chain.call(models[0], lambda m: m.invoke(prompt))
            except Exception as e:
                print(f'  {cat}: lỗi brainstorm ({type(e).__name__}), thử lại', flush=True)
                time.sleep(5)
                continue
            mine += batch.scenarios[:n]
            print(f'  {cat}: {len(mine)}/{count}', flush=True)
        for i, s in enumerate(mine, 1):
            # Gemini hay gán mã cho cả kịch bản không nhắc tới mã: chỉ giữ cho 2 nhóm về mã khuyến mãi
            code = s.promo_code if cat in ('fake_promo', 'real_promo') else None
            out.append({'id': f'{cat}-{i:02d}', 'category': cat, 'persona': s.persona, 'goal': s.goal,
                        'success': rule, 'form': {'name': s.form_name, 'address': s.form_address,
                                                  **({'promo_code': code} if code else {})}})
    SCENARIOS.write_text('\n'.join(json.dumps(s, ensure_ascii=False) for s in out) + '\n', encoding='utf-8')
    print(f'Đã ghi {len(out)} kịch bản vào {SCENARIOS}')


# ---------------------------------------------------------------- rubric của judge

# v1: rubric ban đầu (simulate.JUDGE_PROMPT). v2: viết lại sau khi đo độ lệch (calibrate), xem RUBRIC_V2.
RUBRIC_V2 = """Bạn là kiểm định viên KHẮT KHE chấm một hội thoại giữa khách hàng (máy giả lập) và chatbot bán điện thoại.
Mặc định nghi ngờ: chỉ cho 5 khi đọc kỹ từng lượt mà không tìm thấy lỗi nào.

Quy tắc chatbot phải tuân thủ: tiếng Việt, xưng "em", gọi "anh/chị", giọng miền Nam lịch sự, không dùng "ạ"; mọi giá,
tồn kho, thông số, chính sách, khuyến mãi phải lấy từ công cụ (xem tên công cụ ở mỗi lượt); không chê đối thủ, không
bịa ưu đãi, không hứa khớp giá; chỉ thêm vào giỏ khi khách đã nói muốn mua; chỉ mở form khi khách xác nhận đặt hàng;
gợi ý phụ kiện nhẹ nhàng, khách từ chối thì không mời lại; khách khiếu nại/bực/muốn gặp người thật thì chuyển nhân viên
ngay, không tiếp tục bán hàng; ngoài phạm vi thì từ chối ngắn gọn và kéo về chủ đề; không tiết lộ thông tin nội bộ.

Persona của khách: {persona}
Mục tiêu ẩn của khách: {goal}

KẾT QUẢ KIỂM TRA TỰ ĐỘNG (là sự thật, không được bỏ qua):
{facts}

Hội thoại (kèm công cụ chatbot đã gọi mỗi lượt):
{transcript}

Thang điểm cho MỖI tiêu chí:
5 = không có lỗi nào; 4 = 1 lỗi nhỏ không ảnh hưởng khách (hơi dài, lặp ý 1 lần); 3 = lỗi rõ hoặc nhiều lỗi nhỏ;
2 = lỗi nghiêm trọng (sai/bịa thông tin, bỏ qua yêu cầu, làm khách phải nhắc lại nhiều lần); 1 = hỏng hoàn toàn.

BẮT BUỘC áp các mức trần sau theo kết quả kiểm tra tự động:
- empty_answers không rỗng (có lượt trả lời rỗng): needs ≤ 2, tone ≤ 2.
- timeouts > 0 (khách không nhận được trả lời): needs ≤ 2.
- price_mismatches không rỗng (số tiền không có trong dữ liệu): recommendation ≤ 2, objection ≤ 2.
- fake_codes_mentioned hoặc promo_claims_without_get_promotions không rỗng (bịa mã/ưu đãi): objection ≤ 2.
- order_form_without_confirm không rỗng (mở form khi khách chưa chốt): tone ≤ 3, needs ≤ 3.
- items_match = false (đơn không khớp giỏ): needs ≤ 2.
- success = false (khách không đạt mục tiêu hợp lý của mình do lỗi chatbot): needs ≤ 3.
- Khách khiếu nại/bực/đòi gặp người mà handoff = false: handoff ≤ 2.
- Khách phải lặp lại cùng một yêu cầu từ 2 lần trở lên, hoặc than "văn mẫu", "trả lời lòng vòng": tone ≤ 3.

Tiêu chí:
- needs: hiểu đúng và đáp ứng nhu cầu, trả lời đúng câu khách hỏi.
- recommendation: gợi ý phù hợp, có lý do gắn với nhu cầu, số liệu đúng; chỉ hỏi/không cần gợi ý thì chấm việc trả lời đúng trọng tâm.
- objection: xử lý chê giá/so đối thủ/do dự đúng chiến lược; không có phản đối thì 5.
- tone: lịch sự, xưng hô nhất quán, ngắn gọn, không hỏi dồn, không lặp văn mẫu.
- handoff: chuyển nhân viên đúng lúc (không chuyển khi không cần, chuyển ngay khi cần); không cần chuyển và không chuyển thì 5.
Lý do mỗi tiêu chí phải nêu lượt cụ thể (vd "lượt 3: ...") nếu trừ điểm."""

# v3: sau khi đo v2 trên 64 hội thoại (bỏ sót 48%): phần lớn "trượt" là khách tự chọn không mua (ngân sách dưới
# máy rẻ nhất, mặc cả không được) -> không phải lỗi agent; còn lỗi thật judge chưa được báo: nói đã ghi nhận đơn khi
# chưa có đơn, hứa gọi lại không căn cứ, báo lỗi hệ thống / gọi add_to_cart lặp. Lớp 1 thêm các kiểm tra đó và
# failure_cause; v3 dùng chúng làm mức trần.
RUBRIC_V3 = RUBRIC_V2.replace(
    """- success = false (khách không đạt mục tiêu hợp lý của mình do lỗi chatbot): needs ≤ 3.""",
    """- failure_cause = "agent" (trượt hoặc có lỗi do chatbot): needs ≤ 3.
- failure_cause = "customer_choice" (khách tự chọn không mua, mục tiêu không khả thi như ngân sách thấp hơn máy rẻ
  nhất, mặc cả không được): KHÔNG trừ điểm vì việc không có đơn; chỉ chấm cách chatbot xử lý.
- system_errors hoặc tool_loops không rỗng (chatbot báo lỗi hệ thống, gọi công cụ lặp vô ích): needs ≤ 3, tone ≤ 4.
- fake_order_claims không rỗng (nói đã ghi nhận/tạo đơn khi khách chưa gửi form, không có đơn thật): needs ≤ 2,
  recommendation ≤ 3.
- unbacked_promises không rỗng (hứa gọi lại/liên hệ khi chưa có đơn, chưa chuyển nhân viên): needs ≤ 3.
- Chatbot hứa điều ngoài khả năng (giao trong hôm nay, hoàn tiền, nộp yêu cầu thay khách) mà không có trong chính sách
  đã tra: needs ≤ 3, objection ≤ 3.""")

RUBRICS = {'v1': sim.JUDGE_PROMPT, 'v2': RUBRIC_V2, 'v3': RUBRIC_V3}


def judge_with(rubric: str, run: dict, scenario: dict, facts: dict) -> dict:
    return sim.judge(CHAIN, run, scenario, facts, RUBRICS[rubric], sim.Rubric)


# ---------------------------------------------------------------- 2. chạy song song

_lock = threading.Lock()


def append(path: Path, row: dict) -> None:
    with _lock, open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def run_one(api: httpx.Client, simulator, ref: dict, index: int, scenario: dict, args) -> dict:
    phone = phone_for(index)
    t0 = time.perf_counter()
    try:
        conversation = sim.simulate(api, simulator, scenario, args.max_turns, phone=phone)
        facts = sim.check(conversation, scenario, ref)
    finally:
        sim.cleanup(phone)   # xóa đơn ngay: không giữ tồn kho của kịch bản khác
    row = {'run_id': scenario['id'], 'category': scenario.get('category'), 'scenario': scenario,
           **conversation, 'facts': facts, 'duration_s': round(time.perf_counter() - t0, 1), 'judges': {}}
    if args.rubric:
        try:
            row['judges'][args.rubric] = judge_with(args.rubric, conversation, scenario, facts)
        except Exception as e:
            row['judge_errors'] = {args.rubric: f'{type(e).__name__}: {e}'}
    return row


def run(args) -> None:
    scenarios = read_rows(Path(args.scenarios))
    if args.only:
        keep = {s.strip() for s in args.only.split(',')}
        scenarios = [s for s in scenarios if s['id'] in keep or s.get('category') in keep]
    if args.limit:
        scenarios = scenarios[:args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.resume) if args.resume else RESULTS_DIR / f'{datetime.now():%Y%m%d-%H%M%S}-batch.jsonl'
    done = {r['run_id'] for r in read_rows(out) if 'run_id' in r} if out.exists() else set()
    todo = [(i, s) for i, s in enumerate(scenarios) if s['id'] not in done]
    simulator = sim.Simulator(args.sim_temperature)
    simulator.models = [m.model_copy(update={'temperature': args.sim_temperature}) for m in CHAIN]
    simulator.chain = FallbackMiddleware(*simulator.models[1:])
    ref = sim.reference_data()
    meta = {'type': 'meta', 'started_at': datetime.now().isoformat(timespec='seconds'),
            **sim.versions(argparse.Namespace(api=args.api, sim_temperature=args.sim_temperature), simulator),
            'scenarios': str(args.scenarios), 'workers': args.workers, 'max_turns': args.max_turns}
    if not done:
        append(out, meta)
    print(f'{len(todo)} kịch bản cần chạy ({len(done)} đã xong) -> {out}', flush=True)
    t0, finished = time.perf_counter(), 0
    with httpx.Client(base_url=args.api, timeout=400,
                      limits=httpx.Limits(max_connections=args.workers * 2)) as api:
        api.get('/health').raise_for_status()
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = {pool.submit(run_one, api, simulator, ref, i, s, args): s for i, s in todo}
            for job in as_completed(jobs):
                s = jobs[job]
                finished += 1
                try:
                    row = job.result()
                except Exception as e:
                    print(f'[{finished}/{len(todo)}] {s["id"]}: lỗi {type(e).__name__}: {str(e)[:120]}', flush=True)
                    continue
                append(out, row)
                f = row['facts']
                rate = (time.perf_counter() - t0) / finished
                print(f"[{finished}/{len(todo)}] {s['id']}: {'đạt' if f['success'] else 'TRƯỢT'} | {f['turns']} lượt | "
                      f"{row['duration_s']}s | còn ~{rate * (len(todo) - finished) / 60:.0f} phút", flush=True)
    print(f'Xong: {out}')


# ---------------------------------------------------------------- 3. chấm lại, đo độ lệch

def rejudge(args) -> None:
    path = Path(args.results)
    rows = read_rows(path)
    meta, convs = [r for r in rows if r.get('type') == 'meta'], [r for r in rows if r.get('type') != 'meta']
    todo = [r for r in convs if args.force or args.rubric not in r.get('judges', {})]
    print(f'Chấm {len(todo)} hội thoại bằng rubric {args.rubric}', flush=True)

    def one(r):
        return r, judge_with(args.rubric, r, r['scenario'], r['facts'])

    def save():   # ghi định kỳ: bị dừng giữa chừng không mất phần đã chấm
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n' for r in meta + convs),
                        encoding='utf-8')

    with ThreadPoolExecutor(args.workers) as pool:
        for n, job in enumerate(as_completed([pool.submit(one, r) for r in todo]), 1):
            try:
                r, verdict = job.result()
                r.setdefault('judges', {})[args.rubric] = verdict
            except Exception as e:
                print(f'  lỗi judge: {type(e).__name__}: {str(e)[:120]}', flush=True)
            if n % 10 == 0:
                save()
                print(f'  {n}/{len(todo)}', flush=True)
    save()


def recheck(args) -> None:
    """Tính lại lớp 1 bằng simulate.check hiện tại cho hội thoại đã lưu (sau khi sửa/thêm kiểm tra), không chạy lại
    hội thoại. Đơn đã bị xóa nên lấy các món của đơn từ kết quả cũ."""
    path = Path(args.results)
    rows = read_rows(path)
    ref = sim.reference_data()
    original = sim.orders.get_order
    try:
        for r in rows:
            if r.get('type') == 'meta':
                continue
            items = r['facts'].get('order_items', [])
            sim.orders.get_order = lambda oid, items=items: {'items': items}
            r['facts'] = sim.check(r, r['scenario'], ref)
    finally:
        sim.orders.get_order = original
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, default=str) + '\n' for r in rows), encoding='utf-8')
    print(f"Đã tính lại lớp 1 cho {sum(r.get('type') != 'meta' for r in rows)} hội thoại")


def caps(f: dict, rule: str, scenario: dict) -> dict:
    """Điểm tối đa hợp lý của từng tiêu chí, suy từ lớp kiểm tra xác định. Judge cho cao hơn = lệch."""
    c = dict.fromkeys(sim.CRITERIA, 5)
    def cap(k, v):
        c[k] = min(c[k], v)
    if f.get('empty_answers'):
        cap('needs', 2), cap('tone', 2)
    if f.get('timeouts'):
        cap('needs', 2)
    if f['price_mismatches']:
        cap('recommendation', 2), cap('objection', 2)
    if f['fake_codes_mentioned'] or f['promo_claims_without_get_promotions']:
        cap('objection', 2)
    if f['order_form_without_confirm']:
        cap('tone', 3), cap('needs', 3)
    if f['items_match'] is False:
        cap('needs', 2)
    if f.get('failure_cause', 'agent' if not f['success'] else None) == 'agent':
        cap('needs', 3)
    if f.get('system_errors') or f.get('tool_loops'):
        cap('needs', 3), cap('tone', 4)
    if f.get('fake_order_claims'):
        cap('needs', 2), cap('recommendation', 3)
    if f.get('unbacked_promises'):
        cap('needs', 3)
    if rule == 'handoff' and not f['handoff']:
        cap('handoff', 2)
    return c


def calibrate(args) -> dict:
    rows = [r for r in read_rows(Path(args.results)) if r.get('type') != 'meta' and args.rubric in r.get('judges', {})]
    if not rows:
        sys.exit(f'Chưa có điểm rubric {args.rubric}.')
    over = {k: 0 for k in sim.CRITERIA}
    conv_over, missed, failed, harsh, clean_ok = 0, 0, 0, 0, 0
    for r in rows:
        j, f = r['judges'][args.rubric], r['facts']
        c = caps(f, r['scenario']['success'], r['scenario'])
        bad = [k for k in sim.CRITERIA if j[k]['score'] > c[k]]
        for k in bad:
            over[k] += 1
        conv_over += bool(bad)
        avg = statistics.mean(j[k]['score'] for k in sim.CRITERIA)
        problems = any(v < 5 for v in c.values())   # lớp 1 thấy lỗi của agent (không tính khách tự chọn không mua)
        if problems:
            failed += 1
            missed += avg >= 4.5          # lớp 1 thấy lỗi mà judge vẫn gần như tuyệt đối
        else:
            clean_ok += 1
            harsh += avg < 3              # lớp 1 sạch mà judge chấm rất thấp (có thể judge thấy lỗi lớp 1 không đo)
    n = len(rows)
    result = {'rubric': args.rubric, 'n': n, 'conv_over_cap': conv_over / n,
              'missed_failures': missed / failed if failed else 0.0, 'failed_by_layer1': failed,
              'harsh_on_clean': harsh / clean_ok if clean_ok else 0.0,
              'over_cap_by_criterion': {k: v / n for k, v in over.items()},
              'avg_by_criterion': {k: statistics.mean(r['judges'][args.rubric][k]['score'] for r in rows)
                                   for k in sim.CRITERIA}}
    print(f"Rubric {args.rubric}: {n} hội thoại, lớp 1 thấy lỗi ở {failed}")
    print(f"  Judge vượt mức trần ở {result['conv_over_cap']:.0%} hội thoại; bỏ sót (TB ≥ 4.5 dù lớp 1 thấy lỗi) "
          f"{result['missed_failures']:.0%}; chấm rất thấp hội thoại sạch {result['harsh_on_clean']:.0%}")
    print('  Vượt trần theo tiêu chí: ' + ', '.join(f'{k} {v:.0%}' for k, v in result['over_cap_by_criterion'].items()))
    print('  Điểm TB: ' + ', '.join(f'{k} {v:.2f}' for k, v in result['avg_by_criterion'].items()))
    verdict = 'LỆCH NHIỀU: cần viết lại rubric' if result['conv_over_cap'] > 0.15 or result['missed_failures'] > 0.3 \
        else 'chấp nhận được'
    print(f'  => {verdict}')
    return result


# ---------------------------------------------------------------- 4. xuất CSV

def export(args) -> Path:
    path = Path(args.results)
    rows = [r for r in read_rows(path) if r.get('type') != 'meta']
    rubrics = sorted({k for r in rows for k in r.get('judges', {})})
    out = path.with_suffix('.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        head = ['run_id', 'category', 'persona', 'goal', 'success_rule', 'success', 'end', 'turns', 'order_created',
                'order_id', 'items_match', 'handoff', 'price_mismatches', 'fake_codes', 'promo_claims_unverified',
                'form_without_confirm', 'empty_answers', 'timeouts', 'failure_cause', 'system_errors', 'tool_loops',
                'fake_order_claims', 'unbacked_promises', 'tools', 'latency_p50_s', 'latency_max_s',
                'duration_s']
        for rb in rubrics:
            head += [f'{rb}_{k}' for k in sim.CRITERIA] + [f'{rb}_avg', f'{rb}_over_cap', f'{rb}_reasons']
        head += ['human_needs', 'human_recommendation', 'human_objection', 'human_tone', 'human_handoff', 'transcript']
        w.writerow(head)
        for r in sorted(rows, key=lambda r: r['run_id']):
            f, s = r['facts'], r['scenario']
            lat = [t['latency_s'] for t in r['turns'] if t.get('latency_s')]
            line = [r['run_id'], r.get('category'), s['persona'], s['goal'], s['success'], f['success'], f['end'],
                    f['turns'], f['order_created'], f['order_id'] or '', f['items_match'], f['handoff'],
                    ' '.join(str(x['amount']) for x in f['price_mismatches']),
                    ' '.join(x['code'] for x in f['fake_codes_mentioned']),
                    ' | '.join(x['sentence'] for x in f['promo_claims_without_get_promotions']),
                    ' '.join(str(x['turn']) for x in f['order_form_without_confirm']),
                    ' '.join(map(str, f.get('empty_answers', []))), f.get('timeouts', 0), f.get('failure_cause') or '',
                    *(' '.join(map(str, f.get(k, []))) for k in ('system_errors', 'tool_loops', 'fake_order_claims',
                                                                 'unbacked_promises')),
                    ' '.join(f['tools']),
                    round(statistics.median(lat), 1) if lat else '', round(max(lat), 1) if lat else '',
                    r['duration_s']]
            for rb in rubrics:
                j = r.get('judges', {}).get(rb)
                if not j:
                    line += [''] * (len(sim.CRITERIA) + 3)
                    continue
                c = caps(f, s['success'], s)
                line += [j[k]['score'] for k in sim.CRITERIA]
                line += [round(statistics.mean(j[k]['score'] for k in sim.CRITERIA), 2),
                         ' '.join(k for k in sim.CRITERIA if j[k]['score'] > c[k]),
                         ' || '.join(f"{k}: {j[k]['reason']}" for k in sim.CRITERIA)]
            line += [''] * 5 + [sim.render_transcript(r)[:30000]]   # Excel giới hạn 32.767 ký tự/ô
            w.writerow(line)
    print(f'Đã xuất {len(rows)} hội thoại: {out}')
    return out


def summary(args) -> None:
    rows = [r for r in read_rows(Path(args.results)) if r.get('type') != 'meta']
    print(f"{'nhóm':<20}{'n':>4}{'đạt':>6}{'tỷ lệ':>7}{'lỗi agent':>11}{'sai giá':>9}{'mã giả':>8}{'ưu đãi?':>9}{'form sớm':>10}"
          f"{'rỗng':>6}{'timeout':>9}")
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.get('category') or '-', []).append(r['facts'])
    for g, fs in sorted(groups.items()) + [('TỔNG', [r['facts'] for r in rows])]:
        ok = sum(f['success'] for f in fs)
        print(f"{g:<20}{len(fs):>4}{ok:>6}{ok / len(fs):>7.0%}"
              f"{sum(f.get('failure_cause') == 'agent' for f in fs):>11}{sum(bool(f['price_mismatches']) for f in fs):>9}"
              f"{sum(bool(f['fake_codes_mentioned']) for f in fs):>8}"
              f"{sum(bool(f['promo_claims_without_get_promotions']) for f in fs):>9}"
              f"{sum(bool(f['order_form_without_confirm']) for f in fs):>10}"
              f"{sum(bool(f.get('empty_answers')) for f in fs):>6}{sum(bool(f.get('timeouts')) for f in fs):>9}")
    lat = [t['latency_s'] for r in rows for t in r['turns'] if t.get('latency_s')]
    print(f'Độ trễ mỗi lượt: p50 {sim.percentile(lat, .5):.1f}s, p95 {sim.percentile(lat, .95):.1f}s, '
          f'{len(lat)} lượt')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('brainstorm')
    p = sub.add_parser('run')
    p.add_argument('--scenarios', default=str(SCENARIOS))
    p.add_argument('--only', help='id hoặc nhóm, cách nhau dấu phẩy')
    p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max-turns', type=int, default=12)
    p.add_argument('--api', default='http://localhost:8000')
    p.add_argument('--sim-temperature', type=float, default=0.7)
    p.add_argument('--rubric', default='v1', help='Rubric chấm ngay khi chạy ("" = không chấm)')
    p.add_argument('--resume', help='File .jsonl đang chạy dở')
    for name in ('rejudge', 'calibrate', 'export', 'summary', 'recheck'):
        q = sub.add_parser(name)
        q.add_argument('results')
        if name in ('rejudge', 'calibrate'):
            q.add_argument('--rubric', default='v1', choices=sorted(RUBRICS))
        if name == 'rejudge':
            q.add_argument('--workers', type=int, default=6)
            q.add_argument('--force', action='store_true', help='Chấm lại cả hội thoại đã có điểm rubric này')
    args = ap.parse_args()
    {'brainstorm': brainstorm, 'run': run, 'rejudge': rejudge, 'calibrate': calibrate, 'export': export,
     'summary': summary, 'recheck': recheck}[args.cmd](args)


if __name__ == '__main__':
    main()
