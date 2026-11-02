import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from phone_sales import conversation as conv
from phone_sales import llm, tools


# ---------------------------------------------------------------- nhận diện phản đối

@pytest.mark.parametrize('text, expected', [
    ('máy này đắt quá em', 'price'),
    ('hơi mắc so với túi tiền', 'price'),
    ('bên thế giới di động bán rẻ hơn 2 triệu', 'competitor'),
    ('shop khác rẻ hơn mà, mắc quá', 'competitor'),           # so giá nơi khác ưu tiên hơn chê giá
    ('để anh suy nghĩ thêm đã', 'hesitation'),
    ('em còn phân vân quá', 'hesitation'),
    ('phân vân giữa s26 với iphone 17 nên chọn cái nào', None),  # so sánh 2 máy, không phải do dự
    ('củ sạc để sau nha', None),                               # từ chối bán kèm, không phải do dự
    ('bên em còn màu đen không', None),                        # "bên em" là chính cửa hàng
    ('lấy iphone 17 256gb', None),
])
def test_objection_of(text, expected):
    assert conv.objection_of(text) == expected


# ---------------------------------------------------------------- hướng dẫn theo chiến lược

def test_price_objection_hint():
    text = conv.objection_hint({'objection': 'price', 'slots': {'purposes': ['chụp ảnh']}})
    assert 'min_drop_pct=0.15, max_drop_pct=0.35' in text and 'đánh đổi trung thực' in text
    assert 'gắn với nhu cầu khách đã nói (chụp ảnh)' in text and 'không tự giảm giá' in text
    assert 'không giới thiệu lại máy cũ' in conv.objection_hint({'objection': 'price', 'objection_step': 1})


def test_competitor_hint_never_matches_or_denies_price():
    text = conv.objection_hint({'objection': 'competitor'})
    assert 'không xác nhận hay phủ nhận giá' in text and 'không hứa khớp giá' in text
    assert 'search_policy' in text and 'find_cheaper_alternatives' not in text


def test_hesitation_ladder_then_stop():
    hint = lambda step: conv.objection_hint({'objection': 'hesitation', 'objection_step': step})
    # Bước 1: khuyến mãi thật từ get_promotions, không có thì không nhắc; trả góp 0% chỉ theo search_policy
    assert 'bước 1/3' in hint(0) and 'get_promotions' in hint(0) and 'không có thì không nhắc' in hint(0)
    assert 'trả góp 0%' in hint(0) and 'search_policy' in hint(0)
    assert 'bước 2/3' in hint(1) and 'get_product_details' in hint(1)
    assert 'bước 3/3' in hint(2) and 'min_drop_pct=0.07, max_drop_pct=0.15' in hint(2)
    assert 'không thuyết phục thêm' in hint(3) and 'nhân viên' in hint(3)


def test_no_hint_without_objection_and_cross_sell_pauses():
    assert conv.objection_hint({}) is None
    phone = {'name': 'Galaxy S26', 'category': 'phone', 'charger_in_box': False, 'price': 1, 'quantity': 1}
    state = {'stage': 'closing', 'cart': [phone]}
    assert conv.cross_sell_step(state) == 'charger'
    # Khách chê giá lượt này: không mời bán kèm (và không ghi là đã mời), lượt sau mời tiếp
    assert conv.cross_sell_step({**state, 'objection': 'price'}) is None
    assert 'Xử lý phản đối (chê giá)' in conv.describe({**state, 'objection': 'price'})


# ---------------------------------------------------------------- tool máy rẻ hơn

PHONES = [
    {'product_id': 'p30', 'name': 'Galaxy S26 Ultra', 'brand': 'Samsung', 'storage': '256GB', 'color': 'Đen', 'price': 30_000_000, 'stock': 5},
    {'product_id': 'p30', 'name': 'Galaxy S26 Ultra', 'brand': 'Samsung', 'storage': '512GB', 'color': 'Đen', 'price': 34_000_000, 'stock': 5},
    {'product_id': 'p24', 'name': 'Galaxy S26', 'brand': 'Samsung', 'storage': '256GB', 'color': 'Đen', 'price': 24_000_000, 'stock': 5},
    {'product_id': 'p24', 'name': 'Galaxy S26', 'brand': 'Samsung', 'storage': '512GB', 'color': 'Đen', 'price': 25_500_000, 'stock': 5},
    {'product_id': 'i22', 'name': 'iPhone 17', 'brand': 'Apple', 'storage': '256GB', 'color': 'Đen', 'price': 22_000_000, 'stock': 5},
    {'product_id': 'p21', 'name': 'Galaxy S26 FE', 'brand': 'Samsung', 'storage': '256GB', 'color': 'Đen', 'price': 21_000_000, 'stock': 0},
    {'product_id': 'p12', 'name': 'Galaxy A57', 'brand': 'Samsung', 'storage': '128GB', 'color': 'Đen', 'price': 12_000_000, 'stock': 5},
    {'product_id': 'p7', 'name': 'Galaxy A27', 'brand': 'Samsung', 'storage': '128GB', 'color': 'Đen', 'price': 7_000_000, 'stock': 5},
    {'product_id': 'p2', 'name': 'Galaxy A07', 'brand': 'Samsung', 'storage': '64GB', 'color': 'Đen', 'price': 2_500_000, 'stock': 5},
]


@pytest.fixture
def phones_db(monkeypatch):
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): PHONES)


def cheaper(state=None, **args):
    cmd = tools.find_cheaper_alternatives.invoke(
        {'type': 'tool_call', 'name': 'find_cheaper_alternatives', 'id': 'c1', 'args': {**args, 'state': state or {}}})
    return cmd.update['messages'][0].content, cmd.update.get('offered_alternatives')


def test_cheaper_for_expensive_phone(phones_db):
    # Máy 30 triệu, rẻ hơn 15-35%: 19,5-25,5 triệu, còn hàng; cùng hãng trước, giá gần máy gốc trước
    text, offered = cheaper(product='galaxy s26 ultra', min_drop_pct=0.15, max_drop_pct=0.35)
    assert offered == ['Galaxy S26', 'iPhone 17']                       # S26 FE hết hàng: bỏ
    assert '- Galaxy S26 512GB: 25.500.000đ | rẻ hơn 4.500.000đ' in text  # mỗi máy 1 bản, dung lượng lớn nhất
    assert 'Galaxy A57' not in text                                      # không tụt phân khúc quá sâu


def test_cheap_phone_does_not_suggest_too_cheap(phones_db):
    # Máy 7 triệu: 4,55-5,95 triệu không có máy nào -> nói thật, không gợi ý máy 2,5 triệu
    text, offered = cheaper(product='galaxy a27', min_drop_pct=0.15, max_drop_pct=0.35)
    assert not offered and 'Không có máy khác còn hàng' in text and 'A07' not in text


def test_cheaper_respects_budget_and_skips_already_offered(phones_db):
    text, offered = cheaper({'slots': {'budget_max': 23_000_000}}, product='s26 ultra')
    assert offered == ['iPhone 17'] and 'không vượt ngân sách 23.000.000đ' in text
    text, offered = cheaper({'offered_alternatives': ['Galaxy S26']}, product='s26 ultra')
    assert offered == ['iPhone 17'] and '- Galaxy S26 ' not in text


def test_cheaper_lists_smaller_storage_of_same_phone(phones_db):
    text, _ = cheaper(product='galaxy s26 ultra', storage='512GB', min_drop_pct=0.07, max_drop_pct=0.15)
    assert 'Máy gốc: Galaxy S26 Ultra 512GB 34.000.000đ' in text
    assert 'Dung lượng thấp hơn của chính máy này: 256GB 30.000.000đ (rẻ hơn 4.000.000đ)' in text


def test_cheaper_unknown_product(phones_db):
    text, offered = cheaper(product='ốp lưng trong suốt')
    assert offered is None and 'chỉ áp dụng cho điện thoại' in text


# ---------------------------------------------------------------- graph: bậc thang qua nhiều lượt

class ScriptedModel(GenericFakeChatModel):
    model: str = 'scripted'

    def bind_tools(self, tools, **kwargs):
        return self


def test_graph_tracks_objection_steps(monkeypatch):
    monkeypatch.setattr(llm, 'open_llm', ScriptedModel(messages=iter([AIMessage('Dạ vâng.')] * 6)))
    from phone_sales.graph import build_graph
    g = build_graph(fallback=False, checkpointer=InMemorySaver())
    cfg = {'configurable': {'thread_id': 'o'}}
    say = lambda text: g.invoke({'messages': [HumanMessage(text)]}, cfg)
    s = say('để anh suy nghĩ thêm')
    assert (s['objection_type'], s['objection_step']) == ('hesitation', 1)
    s = say('chưa chắc lắm')
    assert s['objection_step'] == 2
    s = say('ok cảm ơn em')                                   # không phản đối: giữ bậc, không tăng
    assert s['objection'] is None and s['objection_step'] == 2
    s = say('mà đắt quá')                                     # đổi loại -> chiến lược mới từ bước đầu
    assert (s['objection_type'], s['objection_step']) == ('price', 1)
