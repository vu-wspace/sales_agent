"""Tầng 4 của kiểm thử: user simulator end-to-end.

Một LLM (Gemini) đóng vai khách theo persona, có mục tiêu ẩn mà agent không biết (evals/personas.jsonl), chat với web
app đang chạy qua API thật: POST /session -> /chat nhiều lượt -> form đặt hàng (/order hoặc /order/cancel). Mỗi lần chạy
được chấm 2 lớp:

1. Kiểm tra xác định (ưu tiên), từ response của API và database:
   - có tạo đơn không, đơn có đúng các món trong giỏ lúc khách gửi form không;
   - mọi số tiền "...đ" agent nói có khớp database không (giá phiên bản, chênh lệch/tổng 2 giá, khuyến mãi, tổng giỏ);
   - form đặt hàng có hiện khi khách chưa có ý chốt không;
   - agent có nêu mã khuyến mãi không có trong bảng promotions, hay nhắc ưu đãi khi chưa tra get_promotions không;
   - mã giả khách nhập vào form có bị từ chối không; có chuyển nhân viên không.
2. LLM-as-judge chấm 1-5 theo rubric 5 tiêu chí, kèm lý do. Kiểm tra lại judge: điền điểm tự chấm vào file
   *-sim-human.csv rồi chạy --agreement để so độ lệch.

Cần web app (docker compose up -d), GOOGLE_API_KEY (khách giả lập, judge). Đơn tạo trong lúc chạy dùng số điện thoại
SIM_PHONE và được xóa khi chạy xong (trả lại tồn kho, lượt dùng mã), trừ khi có --keep-orders. Email gửi vào Mailpit.

    uv run python evals/simulate.py                               # mỗi persona chạy 1 lần
    uv run python evals/simulate.py --runs 3 --only sv_budget,angry_customer
    uv run python evals/simulate.py --no-judge                    # chỉ kiểm tra xác định, đỡ tốn quota
    uv run python evals/simulate.py --agreement evals/results/20261103-120000-sim-human.csv
"""
import os

# Tắt trace LangSmith của phía giả lập (phải đặt trước khi import phone_sales vì load_dotenv không ghi đè biến đã có)
os.environ['LANGSMITH_TRACING'] = 'false'

import phone_sales  # noqa: E402,F401 - phải import đầu tiên (xem __init__.py)
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from phone_sales import config, llm, orders, pii  # noqa: E402
from phone_sales.fallback import FallbackMiddleware  # noqa: E402

# gemini-3.5-flash-lite bỏ qua temperature và cảnh báo mỗi lần gọi
warnings.filterwarnings('ignore', message='.*uses fixed sampling defaults.*')

EVAL_DIR = Path(__file__).resolve().parent
PERSONAS = EVAL_DIR / 'personas.jsonl'
RESULTS_DIR = EVAL_DIR / 'results'
SIM_PHONE = '0900000003'            # số điện thoại của mọi đơn giả lập (xóa theo số này khi chạy xong)
SIM_EMAIL = 'khach.gialap@example.com'   # email xác nhận đi vào Mailpit

SIM_PROMPT = """Bạn đóng vai một khách hàng đang nhắn tin với nhân viên tư vấn của một cửa hàng điện thoại ở Việt Nam.
Persona: {persona}
Mục tiêu của bạn (không nói thẳng ra, hành xử tự nhiên để đạt được): {goal}

Quy tắc:
- Mỗi lần chỉ viết 1 tin nhắn ngắn (1-2 câu) như chat thật, đúng giọng của persona. Không nhận mình là AI.
- Chỉ dựa vào những gì cửa hàng đã nói; không tự bịa là cửa hàng đã hứa gì.
- Khi thấy dòng [FORM] (cửa hàng hiện form đặt hàng): trả lời đúng <FORM> nếu muốn điền và gửi form, hoặc <CANCEL> nếu
  không muốn đặt nữa.
- Khi thấy [FORM LỖI ...]: trả lời <FORM> để gửi lại (không dùng mã khuyến mãi) hoặc <CANCEL>.
- Khi đã đạt mục tiêu, hoặc thấy không đạt được và muốn rời đi: chỉ trả lời <END>."""

JUDGE_PROMPT = """Bạn chấm một cuộc hội thoại giữa khách hàng (do máy giả lập) và chatbot tư vấn bán điện thoại.
Quy tắc của chatbot: tiếng Việt, xưng "em", gọi "anh/chị", giọng miền Nam lịch sự; chỉ nêu giá, tồn kho, thông số,
chính sách, khuyến mãi lấy từ công cụ; không chê đối thủ, không bịa ưu đãi hay hứa khớp giá; gợi ý phụ kiện nhẹ nhàng,
không ép; khách khiếu nại, bực bội, muốn gặp người thật thì chuyển nhân viên.

Persona của khách: {persona}
Mục tiêu ẩn của khách: {goal}

Kết quả kiểm tra tự động (đáng tin, dùng để đối chiếu):
{facts}

Hội thoại (kèm công cụ chatbot đã gọi mỗi lượt):
{transcript}

Chấm 1-5 cho từng tiêu chí, mỗi tiêu chí kèm 1 câu lý do. 5 = tốt, không lỗi; 3 = được một phần / có lỗi nhỏ;
1 = sai nghiêm trọng. Tiêu chí không áp dụng (vd không có phản đối nào) thì cho 5 và ghi "không áp dụng"."""

CRITERIA = {
    'needs': 'Hiểu đúng nhu cầu khách',
    'recommendation': 'Gợi ý phù hợp, có lý do gắn với nhu cầu',
    'objection': 'Xử lý phản đối đúng chiến lược (không chê đối thủ, không bịa ưu đãi)',
    'tone': 'Giọng điệu lịch sự, xưng hô nhất quán, không hỏi dồn',
    'handoff': 'Biết dừng và chuyển nhân viên đúng lúc',
}


class Score(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(description='1 câu lý do')


class Rubric(BaseModel):
    needs: Score = Field(description=CRITERIA['needs'])
    recommendation: Score = Field(description=CRITERIA['recommendation'])
    objection: Score = Field(description=CRITERIA['objection'])
    tone: Score = Field(description=CRITERIA['tone'])
    handoff: Score = Field(description=CRITERIA['handoff'])


# ---------------------------------------------------------------- dữ liệu đối chiếu trong database

MONEY_RE = re.compile(r'(?<![\d.])(\d{1,3}(?:\.\d{3})+)\s*(?:đ|₫|vnđ|vnd|đồng)', re.I)
CODE_RE = re.compile(r'\bmã(?: khuyến mãi| giảm giá| ưu đãi)?\s*:?\s*\**\s*([A-Z][A-Z0-9-]{3,})\b')
PROMO_TALK_RE = re.compile(r'giảm giá|giảm còn|giảm (?:thêm )?\d|bớt|khuyến mãi|ưu đãi|quà tặng|tặng kèm|voucher', re.I)
# Câu phủ định hoặc lời mời tra cứu ("em có thể tra khuyến mãi") không phải là hứa ưu đãi
NEGATION_RE = re.compile(r'không có|chưa có|hiện không|hiện chưa|không áp dụng|chưa áp dụng|không còn|'
                         r'(?:tra|xem|kiểm tra)(?: thêm)?(?: giúp)? (?:\S+ ){0,3}(?:khuyến mãi|ưu đãi)|'
                         r'không (?:\S+ ){0,4}(?:bớt|giảm|tặng|khớp giá)|\(nếu có\)', re.I)
# Agent báo lỗi hệ thống / không làm được thao tác (vd add_to_cart thất bại liên tục)
# Không tính "xin lỗi vì sự cố này" (sự cố của khách: giao trễ, trừ tiền 2 lần), chỉ tính sự cố phía chatbot
SYSTEM_ERROR_RE = re.compile(r'lỗi hệ thống|hệ thống (?:đang )?(?:gặp|bị|lỗi|có)|(?:em|bên em|công cụ|tool) (?:đang )?'
                             r'(?:gặp|bị) (?:sự cố|trục trặc|lỗi)|trục trặc kỹ thuật|vướng mắc kỹ thuật|'
                             r'không thể thêm|chưa thêm được|không thêm được|chưa mở được form', re.I)
# Agent nói đã tạo/ghi nhận đơn: chỉ đúng sau khi khách gửi form (POST /order)
ORDER_CLAIM_RE = re.compile(r'đã (?:ghi nhận|lên|tạo) đơn|đơn hàng (?:của anh/chị |của cô |của chú )?'
                            r'(?:đã được|đã) (?:ghi nhận|tạo|lên)', re.I)
# Hứa gọi lại / liên hệ: chỉ có căn cứ khi đã có đơn (nhân viên gọi xác nhận) hoặc đã chuyển nhân viên
CALLBACK_RE = re.compile(r'sẽ (?:gọi|liên hệ|nhắn) lại|sẽ gọi (?:cho|điện cho) (?:anh|chị|cô|chú|bạn|mình)', re.I)
# Khách muốn đặt hàng (để phân biệt "không có đơn do agent" với "khách tự chọn không mua")
WANTS_ORDER_RE = re.compile(r'chốt|chot|đặt hàng|dat hang|lên đơn|len don|gửi form|gui form|xác nhận đặt|'
                            r'xac nhan dat|mua luôn|mua lun|lấy luôn|lay luon|<FORM', re.I)
ORDER_RULES = ('order_created', 'order_multi')
CONFIRM_RE = re.compile(r'xác nhận|xac nhan|chốt|chot|đặt|dat|mua|lấy|lay|ok|oke|đồng ý|dong y|được|duoc|\bdc\b|'
                        r'vâng|vang|ừ|\bu\b|có|\bco\b|luôn|luon|form|lên đơn|len don', re.I)


def reference_data() -> dict:
    """Giá phiên bản, khuyến mãi, mã khuyến mãi: số tiền agent nói phải lấy được từ đây."""
    with orders._connect() as conn:
        prices = [r['price'] for r in conn.execute('SELECT DISTINCT price FROM variants WHERE price > 0').fetchall()]
        promos = conn.execute('SELECT code, value, max_discount, min_order FROM promotions').fetchall()
    allowed = set(prices)
    # Chênh lệch (lên dung lượng, máy rẻ hơn) và tổng 2 món (máy + củ sạc) giữa các giá trong database
    for i, a in enumerate(prices):
        for b in prices[i:]:
            allowed.update((abs(a - b), a + b))
    for p in promos:
        allowed.update(v for v in (p['value'], p['max_discount'], p['min_order']) if v)
    for f in (config.ROOT / 'data_schema' / 'policies').glob('*.md'):   # phí giao, ví dụ trả góp trong chính sách
        allowed.update(money(f.read_text(encoding='utf-8')))
    return {'allowed': allowed, 'prices': set(prices), 'codes': {p['code'].upper() for p in promos if p['code']}}


def money(text: str) -> list[int]:
    return [int(m.replace('.', '')) for m in MONEY_RE.findall(text)]


BUDGET_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(triệu|trieu|tr|củ|cu|m|k|nghìn|nghin|ngàn|ngan)\b', re.I)


def customer_amounts(text: str) -> set[int]:
    """Số tiền khách nói ("tầm 9 triệu", "6-9 cu", "500k"): agent được nêu lại hoặc tính chênh lệch với giá máy."""
    out = set(money(text))
    for num, unit in BUDGET_RE.findall(text):
        value = float(num.replace(',', '.'))
        out.add(round(value * (1000 if unit.lower() in ('k', 'nghìn', 'nghin', 'ngàn', 'ngan') else 1_000_000)))
    # "6-9 cu": số đứng trước dấu gạch dùng chung đơn vị với số sau
    for lo, _hi, _unit in re.findall(r'(\d+)\s*-\s*(\d+)\s*(triệu|trieu|tr|củ|cu)\b', text, re.I):
        out.add(int(lo) * 1_000_000)
    return out


# ---------------------------------------------------------------- chạy 1 hội thoại

class Simulator:
    def __init__(self, temperature: float):
        # Gemini làm khách (khác model của agent); hết quota thì chuyển model dự phòng
        self.models = [m.model_copy(update={'temperature': temperature})
                       for m in (llm.google_llm, llm.google_llm_alt, llm.nvidia_llm, llm.groq_llm, llm.open_llm)]
        self.chain = FallbackMiddleware(*self.models[1:])

    def next_message(self, persona: dict, transcript: list[str]) -> str:
        messages = [SystemMessage(SIM_PROMPT.format(**persona)),
                    HumanMessage('Hội thoại tới giờ:\n' + '\n'.join(transcript) + '\n\nTin nhắn tiếp theo của bạn:')]
        text = self.chain.call(self.models[0], lambda m: m.invoke(messages)).text.strip()
        return text.strip('"').strip() or '<END>'


def simulate(api: httpx.Client, sim: Simulator, persona: dict, max_turns: int, phone: str = SIM_PHONE) -> dict:
    session = api.post('/session', json={}).json()
    thread = session['thread_id']
    transcript = [f"Cửa hàng: {session['greeting']}"]
    turns, order, form_shown, promo_code = [], None, False, persona.get('form', {}).get('promo_code')
    end = 'max_turns'
    for _ in range(max_turns):
        if form_shown:
            transcript.append('[FORM] (form đặt hàng: họ tên, số điện thoại, địa chỉ, email, mã khuyến mãi)')
        user = sim.next_message(persona, transcript)
        if '<END>' in user:
            end = 'customer_end'
            break
        if not form_shown and re.search(r'<FORM>|<CANCEL>|\[FORM', user):
            # Khách giả lập tưởng form đã hiện: đổi thành câu chat tương ứng thay vì gửi nguyên thẻ cho agent
            user = 'thôi không đặt nữa' if '<CANCEL>' in user else 'xác nhận đặt hàng giúp mình'
        if form_shown and re.search(r'<FORM|<CANCEL>|\[FORM', user):
            transcript.pop()
            if '<CANCEL>' in user:
                r = api.post('/order/cancel', json={'thread_id': thread}).json()
                turns.append({'user': '<CANCEL>', 'answer': r['answer'], 'tools': [], 'kind': 'cancel'})
                transcript += ['Bạn: (bấm Hủy đặt hàng)', f"Cửa hàng: {r['answer']}"]
                form_shown = False
                continue
            form = {'thread_id': thread, 'name': persona['form']['name'], 'phone': phone,
                    'address': persona['form']['address'], 'email': SIM_EMAIL, 'promo_code': promo_code,
                    'fulfillment_type': 'delivery', 'payment': 'cod'}
            cart = next((t['cart'] for t in reversed(turns) if t['kind'] == 'chat'), [])
            r = api.post('/order', json=form)
            if r.status_code != 200:
                detail = r.json().get('detail')
                turns.append({'user': f'<FORM mã={promo_code}>', 'answer': f'[lỗi {r.status_code}] {detail}',
                              'tools': [], 'kind': 'form_error', 'promo_code': promo_code})
                transcript += [f"Bạn: (gửi form{', mã ' + promo_code if promo_code else ''})", f'[FORM LỖI: {detail}]']
                promo_code = None   # gửi lại thì không dùng mã nữa
                continue
            data = r.json()
            order = {'order_id': data['order_id'], 'cart': cart, 'view': data['order'],
                     'promo_code_sent': form['promo_code']}
            turns.append({'user': '<FORM>', 'answer': data['answer'], 'tools': [], 'kind': 'order'})
            transcript += ['Bạn: (điền và gửi form đặt hàng)', f"Cửa hàng: {data['answer']}"]
            form_shown = False
            continue
        t0 = time.perf_counter()
        try:
            r = api.post('/chat', json={'message': user, 'thread_id': thread})
        except httpx.TimeoutException:
            # Agent quá chậm cũng là kết quả cần ghi lại, không bỏ cả hội thoại
            turns.append({'user': user, 'answer': '[hết thời gian chờ]', 'tools': [], 'kind': 'error',
                          'latency_s': time.perf_counter() - t0})
            end = 'api_timeout'
            break
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            turns.append({'user': user, 'answer': f'[lỗi {r.status_code}] {r.text[:200]}', 'tools': [],
                          'kind': 'error', 'latency_s': latency})
            end = 'api_error'
            break
        d = r.json()
        turns.append({'user': user, 'answer': d['answer'], 'tools': d['tools'], 'kind': 'chat', 'latency_s': latency,
                      'stage': d.get('stage'), 'cart': d.get('cart') or [], 'handoff': d.get('handoff'),
                      'order_form': d.get('order_form')})
        transcript += [f'Bạn: {user}', f"Cửa hàng: {d['answer']}"]
        form_shown = bool(d.get('order_form') and d.get('cart'))
    return {'thread_id': thread, 'greeting': session['greeting'], 'turns': turns, 'order': order, 'end': end}


# ---------------------------------------------------------------- lớp 1: kiểm tra xác định

def check(run: dict, persona: dict, ref: dict) -> dict:
    chats = [t for t in run['turns'] if t['kind'] == 'chat']
    allowed = set(ref['allowed'])
    for t in chats:   # tổng giỏ và từng món trong giỏ (số lượng > 1) cũng là số đúng
        allowed.add(sum(i['price'] * i['quantity'] for i in t['cart']))
        allowed.update(i['price'] * i['quantity'] for i in t['cart'])
    order = run['order']
    if order:
        v = order['view']
        allowed.update(x for x in (v.get('subtotal'), v.get('discount'), v.get('total')) if x)
    bad_money, fake_codes, unverified_promo, form_unconfirmed = [], [], [], []
    empty = [i for i, t in enumerate(run['turns'], 1) if t['kind'] == 'chat' and not t['answer'].strip()]
    promotions_checked = False
    said_by_customer: set[str] = set()   # mã khách tự nêu: agent nhắc lại để trả lời không phải là bịa mã
    for t in run['turns']:   # ngân sách khách nói và chênh lệch giữa ngân sách với giá trong database
        for b in customer_amounts(t['user']):
            allowed.add(b)
            allowed.update(abs(p - b) for p in ref.get('prices', ()) if abs(p - b) < 50_000_000)
    for i, t in enumerate(run['turns'], 1):
        said_by_customer.update(c.upper() for c in re.findall(r'\b[A-Z][A-Z0-9-]{3,}\b', t['user']))
        if t['kind'] not in ('chat', 'order', 'cancel'):
            continue
        if 'get_promotions' in t['tools']:
            promotions_checked = True
        bad_money += [{'turn': i, 'amount': a} for a in money(t['answer']) if a not in allowed]
        fake_codes += [{'turn': i, 'code': c} for c in CODE_RE.findall(t['answer'])
                       if c.upper() not in ref['codes'] and c.upper() not in said_by_customer]
        if t['kind'] == 'chat' and not promotions_checked:
            for sentence in re.split(r'(?<=[.!?\n])\s*', t['answer']):
                if PROMO_TALK_RE.search(sentence) and not NEGATION_RE.search(sentence):
                    unverified_promo.append({'turn': i, 'sentence': sentence.strip()[:160]})
        if t['kind'] == 'chat' and t.get('order_form') and not CONFIRM_RE.search(t['user']):
            form_unconfirmed.append({'turn': i, 'user': t['user']})
    tools = [tool for t in chats for tool in t['tools']]
    items_match = None
    order_items = []
    if order:
        saved = orders.get_order(order['order_id'])
        order_items = [{'name': it['name'], 'storage': it['storage'], 'color': it['color'], 'qty': it['qty']}
                       for it in saved['items']]
        want = sorted((c['name'], c.get('storage'), c.get('color'), c['quantity']) for c in order['cart'])
        items_match = want == sorted((it['name'], it['storage'], it['color'], it['qty']) for it in order_items)
    fake_code_sent = persona.get('form', {}).get('promo_code')
    fake_code_rejected = None
    if fake_code_sent and fake_code_sent.upper() not in ref['codes']:
        tried = any(t.get('promo_code') == fake_code_sent for t in run['turns'] if t['kind'] == 'form_error') or \
            (order and order['promo_code_sent'] == fake_code_sent)
        fake_code_rejected = None if not tried else not (order and order['promo_code_sent'] == fake_code_sent)
    facts = {
        'order_created': bool(order), 'order_id': order and order['order_id'], 'items_match': items_match,
        'order_items': order_items,
        'order_categories': sorted(c.get('category') or '' for c in order['cart']) if order else [],
        'handoff': any(t.get('handoff') for t in chats),
        'price_mismatches': bad_money, 'fake_codes_mentioned': fake_codes,
        'promo_claims_without_get_promotions': unverified_promo,
        'order_form_without_confirm': form_unconfirmed, 'fake_code_rejected': fake_code_rejected,
        'empty_answers': empty, 'timeouts': sum(t['kind'] == 'error' for t in run['turns']),
        **agent_issues(run),
        'tools': tools, 'turns': len([t for t in run['turns'] if t['kind'] != 'error']), 'end': run['end'],
    }
    facts['success'] = success(persona['success'], facts)
    facts['failure_cause'] = failure_cause(persona['success'], facts, run)
    return facts


def agent_issues(run: dict) -> dict:
    """Lỗi chỉ thấy được khi đọc câu trả lời theo thứ tự lượt: báo lỗi hệ thống, gọi add_to_cart lặp, nói đã ghi
    nhận đơn khi chưa có đơn, hứa gọi lại khi chưa có đơn và chưa chuyển nhân viên."""
    system_errors, loops, fake_orders, promises = [], [], [], []
    ordered = handed_off = False
    for i, t in enumerate(run['turns'], 1):
        if t['kind'] == 'order':
            ordered = True
        if t['kind'] != 'chat':
            continue
        handed_off = handed_off or bool(t.get('handoff'))
        if SYSTEM_ERROR_RE.search(t['answer']):
            system_errors.append(i)
        if t['tools'].count('add_to_cart') >= 4:
            loops.append(i)
        if not ordered and ORDER_CLAIM_RE.search(t['answer']):
            fake_orders.append(i)
        if not ordered and not handed_off and CALLBACK_RE.search(t['answer']):
            promises.append(i)
    return {'system_errors': system_errors, 'tool_loops': loops, 'fake_order_claims': fake_orders,
            'unbacked_promises': promises}


def failure_cause(rule: str, f: dict, run: dict) -> str | None:
    """Trượt do agent ('agent') hay do khách tự chọn không mua / mục tiêu không khả thi ('customer_choice')."""
    agent_fault = (f.get('empty_answers') or f.get('timeouts') or f.get('system_errors') or f.get('tool_loops')
                   or f.get('fake_order_claims') or f['price_mismatches'] or f['fake_codes_mentioned']
                   or f['promo_claims_without_get_promotions'] or f['items_match'] is False
                   or f['fake_code_rejected'] is False)
    if f['success']:
        return 'agent' if agent_fault else None   # đạt mục tiêu nhưng vẫn có lỗi agent
    if agent_fault:
        return 'agent'
    wanted = any(WANTS_ORDER_RE.search(t['user']) for t in run['turns'] if t['kind'] in ('chat', 'order', 'form_error'))
    if rule in ORDER_RULES and wanted:
        return 'agent'   # khách đã muốn đặt mà không có đơn đúng
    if rule == 'handoff':
        return 'agent'
    return 'customer_choice'


def success(rule: str, f: dict) -> bool:
    clean = not (f['price_mismatches'] or f['fake_codes_mentioned'] or f['promo_claims_without_get_promotions'])
    if f.get('empty_answers') or f.get('timeouts') or f.get('fake_order_claims'):   # trượt với mọi persona
        return False
    if rule == 'order_created':
        return f['order_created'] and bool(f['items_match'])
    if rule == 'order_multi':   # 1 máy + ít nhất 2 phụ kiện, đúng giỏ
        cats = f['order_categories']
        return bool(f['items_match']) and cats.count('phone') >= 1 and len(cats) - cats.count('phone') >= 2
    if rule == 'handoff':
        return f['handoff']
    if rule == 'no_fake_discount':
        return clean and f['fake_code_rejected'] is not False
    if rule == 'order_or_compare':   # đặt hàng, hoặc ít nhất đã tra chi tiết 2 máy để so sánh
        return (f['order_created'] and bool(f['items_match'])) or f['tools'].count('get_product_details') >= 2
    if rule == 'info_only':   # chỉ hỏi thông tin: trả lời không bịa, không mở form khi khách chưa chốt
        return clean and not f['order_form_without_confirm'] and f['items_match'] is not False
    raise ValueError(f'Không rõ tiêu chí thành công: {rule}')


# ---------------------------------------------------------------- lớp 2: LLM-as-judge

def render_transcript(run: dict) -> str:
    lines = [f"Cửa hàng: {run['greeting']}"]
    for t in run['turns']:
        tools = f" [công cụ: {', '.join(t['tools'])}]" if t['tools'] else ''
        lines += [f"Khách: {t['user']}", f"Cửa hàng{tools}: {t['answer']}"]
    return '\n'.join(lines)


def judge(models: list, run: dict, persona: dict, facts: dict, prompt_template: str = JUDGE_PROMPT,
          schema: type[BaseModel] = Rubric) -> dict:
    short = {k: v for k, v in facts.items() if k not in ('tools',)}
    prompt = prompt_template.format(persona=persona['persona'], goal=persona['goal'],
                                    facts=json.dumps(short, ensure_ascii=False, default=str),
                                    transcript=render_transcript(run))
    structured = [m.with_structured_output(schema) for m in models]
    chain = FallbackMiddleware(*structured[1:])
    verdict = chain.call(structured[0], lambda m: m.invoke(prompt))
    if verdict is None:   # model không gọi tool / trả JSON sai schema
        raise ValueError('judge không trả về điểm theo schema')
    return verdict.model_dump()


# ---------------------------------------------------------------- dọn dữ liệu, báo cáo

def cleanup(phone: str = SIM_PHONE) -> int:
    """Xóa đơn của số điện thoại giả lập: trả tồn kho, trả lượt dùng mã, xóa đơn và khách."""
    h = pii.phone_key(phone)
    with orders._connect() as conn:
        ids = [r['order_id'] for r in conn.execute(
            'SELECT o.order_id FROM orders o JOIN customers c USING (customer_id) WHERE c.phone_hash = %s', (h,))]
        if not ids:
            return 0
        conn.execute('UPDATE variants v SET stock = v.stock + oi.qty FROM order_items oi '
                     'WHERE oi.order_id = ANY(%s) AND v.variant_id = oi.variant_id', (ids,))
        conn.execute('UPDATE promotions p SET used_count = p.used_count - u.n FROM '
                     '(SELECT promo_id, count(*) AS n FROM promotion_usages WHERE order_id = ANY(%s) GROUP BY promo_id) u '
                     'WHERE p.promo_id = u.promo_id', (ids,))
        conn.execute('DELETE FROM promotion_usages WHERE order_id = ANY(%s)', (ids,))
        conn.execute('DELETE FROM order_items WHERE order_id = ANY(%s)', (ids,))
        conn.execute('DELETE FROM orders WHERE order_id = ANY(%s)', (ids,))
        conn.execute('DELETE FROM customers WHERE phone_hash = %s', (h,))
    return len(ids)


def versions(args, sim: Simulator) -> dict:
    def git(*cmd):
        try:
            return subprocess.run(['git', *cmd], capture_output=True, text=True, cwd=EVAL_DIR).stdout.strip()
        except OSError:
            return None
    return {
        'git_commit': git('rev-parse', '--short', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
        'agent_primary_model': f'9Router {config.ROUTER9_MODEL} (+ chuỗi dự phòng của server)',
        'simulator_model': llm.model_name(sim.models[0]), 'simulator_temperature': args.sim_temperature,
        'judge_model': llm.model_name(llm.google_llm), 'api': args.api,
        'personas_sha1': hashlib.sha1(PERSONAS.read_bytes()).hexdigest()[:10],
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))]


def report(results: list[dict], meta: dict) -> None:
    print(f"\nPhiên bản: commit {meta['git_commit']}{' (có sửa chưa commit)' if meta['git_dirty'] else ''} | "
          f"agent {meta['agent_primary_model']} | khách {meta['simulator_model']} | judge {meta['judge_model']}")
    header = (f"{'persona':<16}{'đạt':>7}{'đơn':>6}{'sai giá':>9}{'mã giả':>8}{'ưu đãi?':>9}{'form sớm':>10}"
              f"{'rỗng':>6}{'judge TB':>10}")
    print(header)
    print('-' * len(header))
    by_persona: dict[str, list] = {}
    for r in results:
        by_persona.setdefault(r['persona'], []).append(r)
    for pid, rs in by_persona.items():
        f = [r['facts'] for r in rs]
        judged = [statistics.mean(c['score'] for c in r['judge'].values()) for r in rs if r.get('judge')]
        print(f"{pid:<16}{sum(x['success'] for x in f):>4}/{len(f):<2}{sum(x['order_created'] for x in f):>6}"
              f"{sum(len(x['price_mismatches']) for x in f):>9}{sum(len(x['fake_codes_mentioned']) for x in f):>8}"
              f"{sum(len(x['promo_claims_without_get_promotions']) for x in f):>9}"
              f"{sum(len(x['order_form_without_confirm']) for x in f):>10}"
              f"{sum(len(x.get('empty_answers', [])) for x in f):>6}"
              f"{(f'{statistics.mean(judged):.2f}' if judged else '-'):>10}")
    lat = [t['latency_s'] for r in results for t in r['turns'] if t.get('latency_s')]
    ok = sum(r['facts']['success'] for r in results)
    print(f"\nTổng: đạt {ok}/{len(results)} ({ok / len(results):.0%}) | độ trễ mỗi lượt p50 "
          f"{percentile(lat, .5) or 0:.1f}s, p95 {percentile(lat, .95) or 0:.1f}s")
    judged = [r for r in results if r.get('judge')]
    if judged:
        print('Judge TB theo tiêu chí: ' + ', '.join(
            f"{k} {statistics.mean(r['judge'][k]['score'] for r in judged):.2f}" for k in CRITERIA))
    for r in results:
        f = r['facts']
        problems = ([f"sai giá {[x['amount'] for x in f['price_mismatches']]}"] if f['price_mismatches'] else []) + \
            ([f"mã giả {[x['code'] for x in f['fake_codes_mentioned']]}"] if f['fake_codes_mentioned'] else []) + \
            ([f"nhắc ưu đãi chưa tra: \"{f['promo_claims_without_get_promotions'][0]['sentence']}\""]
             if f['promo_claims_without_get_promotions'] else []) + \
            ([f"form khi chưa chốt (lượt {f['order_form_without_confirm'][0]['turn']})"]
             if f['order_form_without_confirm'] else []) + \
            (['đơn không khớp giỏ'] if f['items_match'] is False else []) + \
            (['mã giả được nhận'] if f['fake_code_rejected'] is False else []) + \
            ([f"trả lời rỗng ở lượt {f['empty_answers']}"] if f.get('empty_answers') else []) + \
            ([f"lỗi/hết thời gian chờ API ({f['end']})"] if f.get('timeouts') else [])
        if not f['success'] or problems:
            print(f"\n[{r['run_id']}] {'ĐẠT' if f['success'] else 'TRƯỢT'} ({r['success_rule']}), kết thúc: {f['end']}, "
                  f"{f['turns']} lượt" + ("\n  - " + '\n  - '.join(problems) if problems else ''))


def write_outputs(results: list[dict], meta: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f'{datetime.now():%Y%m%d-%H%M%S}-sim'
    stem.with_suffix('.json').write_text(json.dumps({'meta': meta, 'results': results}, ensure_ascii=False,
                                                    indent=2, default=str), encoding='utf-8')
    # Bản đọc cho người duyệt (tầng 6) và file tự chấm để kiểm tra judge
    md = [f"# Hội thoại giả lập {stem.name}\n"]
    for r in results:
        md.append(f"## {r['run_id']} ({'đạt' if r['facts']['success'] else 'trượt'})\n")
        md.append('```\n' + render_transcript(r) + '\n```\n')
        if r.get('judge'):
            md += [f"- {CRITERIA[k]}: {v['score']} – {v['reason']}" for k, v in r['judge'].items()]
        md.append('')
    stem.with_suffix('.md').write_text('\n'.join(md), encoding='utf-8')
    with open(f'{stem}-human.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['run_id', 'persona', 'criterion', 'judge', 'human'])
        for r in results:
            for k in CRITERIA:
                if r.get('judge'):
                    w.writerow([r['run_id'], r['persona'], k, r['judge'][k]['score'], ''])
    return stem


def agreement(path: str) -> None:
    """So điểm judge với điểm người chấm (cột human). Lệch nhiều thì cần viết lại rubric."""
    with open(path, encoding='utf-8-sig') as fh:
        rows = [r for r in csv.DictReader(fh) if r['human'].strip()]
    if not rows:
        sys.exit('Chưa có điểm nào ở cột human.')
    print(f"{'tiêu chí':<16}{'n':>4}{'trùng':>8}{'lệch ≤1':>9}{'MAE':>6}")
    for k in [*CRITERIA, None]:
        rs = [r for r in rows if k is None or r['criterion'] == k]
        if not rs:
            continue
        d = [abs(int(r['judge']) - int(r['human'])) for r in rs]
        print(f"{k or 'tổng':<16}{len(rs):>4}{sum(x == 0 for x in d) / len(d):>8.0%}"
              f"{sum(x <= 1 for x in d) / len(d):>9.0%}{statistics.mean(d):>6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='Danh sách id persona, cách nhau dấu phẩy')
    ap.add_argument('--runs', type=int, default=1, help='Số lần chạy mỗi persona (báo tỷ lệ đạt)')
    ap.add_argument('--max-turns', type=int, default=12)
    ap.add_argument('--api', default='http://localhost:8000')
    ap.add_argument('--sim-temperature', type=float, default=0.0)
    ap.add_argument('--no-judge', action='store_true')
    ap.add_argument('--keep-orders', action='store_true', help='Không xóa đơn giả lập sau khi chạy')
    ap.add_argument('--agreement', metavar='CSV', help='So điểm judge với điểm tự chấm rồi thoát')
    args = ap.parse_args()
    # Console Windows mặc định cp1252, in tiếng Việt bị lỗi
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    if args.agreement:
        return agreement(args.agreement)

    personas = [json.loads(line) for line in PERSONAS.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.only:
        keep = {s.strip() for s in args.only.split(',')}
        personas = [p for p in personas if p['id'] in keep]
    sim = Simulator(args.sim_temperature)
    meta = {**versions(args, sim), 'started_at': datetime.now().isoformat(timespec='seconds'),
            'runs': args.runs, 'max_turns': args.max_turns}
    ref = reference_data()
    judges = [llm.google_llm, llm.google_llm_alt]
    results = []
    with httpx.Client(base_url=args.api, timeout=400) as api:
        try:
            api.get('/health').raise_for_status()
        except httpx.HTTPError as e:
            sys.exit(f'Không gọi được web app ở {args.api} ({e}). Chạy: docker compose up -d')
        try:
            for p in personas:
                for n in range(1, args.runs + 1):
                    run_id = f"{p['id']}#{n}"
                    print(f'… {run_id}', flush=True)
                    t0 = time.perf_counter()
                    try:
                        run = simulate(api, sim, p, args.max_turns)
                    except Exception as e:   # 1 hội thoại lỗi không dừng cả lần chạy
                        print(f'  lỗi khi giả lập: {type(e).__name__}: {e}')
                        continue
                    facts = check(run, p, ref)
                    result = {'run_id': run_id, 'persona': p['id'], 'success_rule': p['success'], **run,
                              'facts': facts, 'duration_s': round(time.perf_counter() - t0, 1)}
                    if not args.no_judge:
                        try:
                            result['judge'] = judge(judges, run, p, facts)
                        except Exception as e:
                            result['judge_error'] = f'{type(e).__name__}: {e}'
                    results.append(result)
                    print(f"  {'đạt' if facts['success'] else 'trượt'} | {facts['turns']} lượt | "
                          f"đơn {facts['order_id'] or '-'} | {result['duration_s']}s", flush=True)
        finally:
            if not args.keep_orders:
                print(f'\nĐã xóa {cleanup()} đơn giả lập (số {SIM_PHONE}).')
    if not results:
        sys.exit('Không có hội thoại nào chạy xong.')
    report(results, meta)
    stem = write_outputs(results, meta)
    print(f'\nKết quả: {stem}.json | hội thoại: {stem}.md | tự chấm judge: {stem}-human.csv')


if __name__ == '__main__':
    main()
