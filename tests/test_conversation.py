import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from phone_sales import conversation as conv
from phone_sales import llm, tools


@pytest.mark.parametrize('text, expected', [
    ('tầm 15 triệu', 15_000_000), ('dưới 20tr', 20_000_000), ('khoảng 7,5 triệu', 7_500_000),
    ('từ 10 đến 15 triệu', 15_000_000), ('ngân sách 12.000.000', 12_000_000), ('20 củ', 20_000_000),
    ('iphone 17 bao nhiêu', None),
])
def test_parse_budget(text, expected):
    assert conv.parse_budget(text) == expected


def test_track_user_message_extracts_needs():
    info = conv.track_user_message('Tầm 30 triệu, mua iphone chụp ảnh đẹp, pin trâu, bản 256GB')
    assert info['slots'] == {'budget_max': 30_000_000, 'brand': 'Apple',
                             'purposes': ['chụp ảnh', 'pin lâu'], 'storage': '256GB'}
    assert info['sentiment_history'] == ['neutral']


@pytest.mark.parametrize('text', [
    'không biết chọn máy nào',   # "ông" trong "không" không phải "ông bà"
    'máy có chip gì',            # "ip" trong "chip" không phải iPhone
])
def test_no_false_keyword_matches(text):
    info = conv.track_user_message(text)
    assert info['slots'] == {}


def test_objections_and_sentiment():
    info = conv.track_user_message('máy này đắt quá, mà nghe nói hao pin')
    assert info['objections'] == ['chê giá cao', 'lo pin yếu']
    assert conv.sentiment_of('đặt hàng giao về Đà Nẵng') == 'neutral'      # "đặt" không phải "đắt"
    assert conv.track_user_message('đặt hàng giao về Đà Nẵng')['objections'] == []
    assert conv.sentiment_of('tức là máy này 256GB hả') == 'neutral'       # "tức là" không phải tức giận
    assert conv.sentiment_of('không thích màu này') == 'neutral'
    assert conv.sentiment_of('ok cảm ơn em') == 'positive'
    assert conv.sentiment_of('giao chậm quá, bực ghê') == 'negative'


def test_handoff_on_request_or_two_negative_turns():
    assert conv.needs_handoff({}, wants_handoff=True)
    assert conv.track_user_message('cho tôi gặp nhân viên')['wants_handoff']
    assert not conv.needs_handoff({'sentiment_history': ['neutral', 'negative']}, False)
    assert conv.needs_handoff({'sentiment_history': ['negative', 'negative']}, False)
    assert conv.needs_handoff({'handoff': True, 'sentiment_history': ['positive']}, False)  # đã bật thì giữ


@pytest.mark.parametrize('state, compare, close, expected', [
    ({}, False, False, 'discovery'),
    ({'shown_products': ['iPhone 17']}, False, False, 'recommending'),
    ({'shown_products': ['iPhone 17', 'Galaxy S26']}, True, False, 'comparing'),
    ({'shown_products': ['iPhone 17', 'Galaxy S26'], 'stage': 'comparing'}, False, False, 'comparing'),
    ({'shown_products': ['iPhone 17']}, False, True, 'closing'),
    ({'cart': [{'name': 'x'}]}, False, False, 'closing'),
    ({'cart': [{'name': 'x'}], 'handoff': True}, False, False, 'handoff'),
])
def test_decide_stage(state, compare, close, expected):
    assert conv.decide_stage(state, compare, close) == expected


def test_reducers():
    assert conv.merge_slots({'brand': 'Apple', 'purposes': ['chụp ảnh']},
                            {'brand': 'Samsung', 'purposes': ['chụp ảnh', 'pin lâu'], 'color': None}) == \
        {'brand': 'Samsung', 'purposes': ['chụp ảnh', 'pin lâu']}
    assert conv.append_unique(['A', 'B'], ['B', 'C']) == ['A', 'B', 'C']


def test_track_tool_results_reads_only_current_turn():
    msgs = [
        HumanMessage('lượt cũ'),
        AIMessage('', tool_calls=[{'name': 'get_product_details', 'args': {'product': 'x'}, 'id': 'old'}]),
        ToolMessage('iPhone 16 (Apple)\n...', tool_call_id='old'),
        HumanMessage('tầm 30 triệu'),
        AIMessage('', tool_calls=[
            {'name': 'search_products', 'args': {'max_price': 30_000_000, 'brand': 'Samsung'}, 'id': 'a'},
            {'name': 'get_product_details', 'args': {'product': 's26', 'storage': '256GB'}, 'id': 'b'},
            {'name': 'get_product_details', 'args': {'product': 'iphone'}, 'id': 'c'},
        ]),
        ToolMessage('Sản phẩm đang bán:\n- Galaxy A57 5G (Samsung): 12.490.000đ\n- Galaxy S26 FE (Samsung): 18.990.000đ\n'
                    '- Galaxy S26 (Samsung): 29.990.001đ\n- Galaxy S26+ (Samsung): 33.990.000đ', tool_call_id='a'),
        ToolMessage('Galaxy S26 (Samsung)\nMô tả', tool_call_id='b'),
        ToolMessage("Có nhiều máy khớp 'iphone': ...", tool_call_id='c'),
    ]
    assert conv.track_tool_results(msgs) == {'slots': {'budget_max': 30_000_000, 'brand': 'Samsung', 'storage': '256GB'}}

    # Chỉ tính máy model thật sự nhắc trong câu trả lời, không tính cả danh sách search_products;
    # "Galaxy S26" không bị tính nhầm khi câu trả lời nói "Galaxy S26+"
    reply = AIMessage('Em gợi ý **Galaxy S26 (256GB)** 29.990.001đ và Galaxy S26+ 33.990.000đ ạ.')
    assert conv.products_mentioned(msgs + [reply]) == ['Galaxy S26+', 'Galaxy S26']
    assert conv.products_mentioned(msgs + [AIMessage('Galaxy S26 Ultra thì sao ạ?')]) == []


def test_describe_lists_state_for_prompt():
    text = conv.describe({'stage': 'closing', 'slots': {'budget_max': 30_000_000, 'purposes': ['chụp ảnh']},
                          'shown_products': ['Galaxy S26'], 'objections': ['chê giá cao'],
                          'cart': [{'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen', 'price': 29990001, 'quantity': 2}],
                          'sentiment_history': ['negative']})
    assert text.startswith('Bước hiện tại: closing.')
    assert 'ngân sách tối đa 30.000.000đ; nhu cầu: chụp ảnh' in text
    assert 'Giỏ hàng: 2 x Galaxy S26 256GB Đen (29.990.001đ). Tổng 59.980.002đ' in text
    assert 'Khách đang lo ngại: chê giá cao' in text and 'Khách đang không vui' in text


PHONE_IN_CART = {'name': 'Galaxy S26', 'category': 'phone', 'storage': '256GB', 'color': 'Đen', 'price': 1,
                 'quantity': 1, 'charger_in_box': False}


def _find_acc(kind: str) -> list:
    return [HumanMessage('ok'),
            AIMessage('', tool_calls=[{'name': 'find_accessories', 'args': {'phone': 'S26', 'accessory_type': kind},
                                       'id': 'f'}]),
            ToolMessage('Phụ kiện tương thích với Galaxy S26:\n- Ốp: 490.320đ | còn hàng', tool_call_id='f')]


def test_cross_sell_charger_first_then_accessories_then_stop():
    state = {'stage': 'closing', 'cart': [PHONE_IN_CART]}
    hint = conv.cross_sell_hint
    # Máy không kèm củ sạc: bước 1 mời sạc + cáp
    assert hint(state).startswith('Bán kèm (bước 1)') and "accessory_type='sạc, cáp'" in hint(state)
    # Đã mời sạc: bước 2 ốp lưng, cường lực; khách vừa từ chối sạc thì mời nhẹ
    assert hint({**state, 'cross_sell': ['charger']}).startswith('Bán kèm (bước 2)')
    assert 'mời nhẹ' in hint({**state, 'cross_sell': ['charger'], 'declined': True})
    # Khách từ chối tiếp: không mời nữa, hỏi chốt đơn; không từ chối thì im lặng
    assert 'hỏi khách có chốt đơn' in hint({**state, 'cross_sell': ['charger', 'accessories'], 'declined': True})
    assert hint({**state, 'cross_sell': ['charger', 'accessories']}) is None


def test_cross_sell_only_chargers_when_cable_in_box():
    text = conv.cross_sell_hint({'stage': 'closing', 'cart': [{**PHONE_IN_CART, 'cable_in_box': True}]})
    assert "accessory_type='củ sạc'" in text and '2-3 củ sạc chính hãng' in text and 'rẻ tới đắt' in text
    assert 'đã có cáp sạc' in text and 'Không mời cáp' in text


def test_cross_sell_skips_charger_when_in_box_or_already_in_cart():
    state = {'stage': 'closing', 'cart': [{**PHONE_IN_CART, 'charger_in_box': True}]}
    assert conv.cross_sell_hint(state).startswith('Bán kèm (bước 2)')
    with_charger = [PHONE_IN_CART, {'name': 'Củ sạc', 'category': 'accessory', 'accessory_type': 'charger'}]
    assert conv.cross_sell_hint({'stage': 'closing', 'cart': with_charger}).startswith('Bán kèm (bước 2)')


def test_cross_sell_not_offered_outside_closing_twice_in_turn_or_when_unhappy():
    state = {'stage': 'closing', 'cart': [PHONE_IN_CART]}
    assert conv.cross_sell_hint({**state, 'stage': 'recommending'}) is None
    assert conv.cross_sell_hint({**state, 'messages': _find_acc('sạc, cáp')}) is None   # lượt này đã mời
    assert 'Bán kèm' in conv.describe(state)
    assert 'Bán kèm' not in conv.describe({**state, 'sentiment_history': ['negative']})


def test_cross_sell_offers_and_decline_detection():
    assert conv.cross_sell_offers(_find_acc('sạc, cáp')) == ['charger']
    assert conv.cross_sell_offers(_find_acc('ốp lưng, cường lực')) == ['accessories']
    assert conv.cross_sell_offers(_find_acc('')) == ['accessories']
    assert conv.track_user_message('thôi không cần sạc đâu em')['declined']
    assert not conv.track_user_message('ok lấy luôn thôi')['declined']


# ---------------------------------------------------------------- tool giỏ hàng

CATALOG = [{'product_id': 'samsung-galaxy-s26', 'name': 'Galaxy S26', 'category': 'phone'},
           {'product_id': 'acc-op', 'name': 'Ốp Lưng Trong Suốt Galaxy S26', 'category': 'accessory'}]
VARIANTS = {
    'samsung-galaxy-s26': [
        {'variant_id': 'S26-256-DEN', 'color': 'Đen Classic', 'storage': '256GB', 'price': 29990001, 'stock': 3},
        {'variant_id': 'S26-512-DEN', 'color': 'Đen Classic', 'storage': '512GB', 'price': 35990000, 'stock': 0},
    ],
    'acc-op': [{'variant_id': 'OP', 'color': None, 'storage': None, 'price': 490320, 'stock': 10}],
}


@pytest.fixture
def fake_catalog(monkeypatch):
    def query(sql, params=()):
        return VARIANTS[params[0]] if 'FROM variants' in sql else CATALOG
    monkeypatch.setattr(tools, '_query', query)


def call(tool, state, **args):
    cmd = tool.invoke({'type': 'tool_call', 'name': tool.name, 'id': 'c1', 'args': {**args, 'state': state}})
    return cmd.update.get('cart'), cmd.update['messages'][0].content


def test_add_to_cart_merges_same_variant_and_checks_stock(fake_catalog):
    cart, text = call(tools.add_to_cart, {}, product='galaxy s26', storage='256gb', color='đen')
    assert cart == [{'variant_id': 'S26-256-DEN', 'name': 'Galaxy S26', 'category': 'phone', 'storage': '256GB',
                     'color': 'Đen Classic', 'price': 29990001, 'quantity': 1, 'charger_in_box': None,
                     'cable_in_box': None}]
    assert text.endswith('Giỏ có 1 sản phẩm, tổng 29.990.001đ.')
    cart, _ = call(tools.add_to_cart, {'cart': cart}, product='s26', storage='256GB', quantity=2)
    assert cart[0]['quantity'] == 3
    cart2, text = call(tools.add_to_cart, {'cart': cart}, product='s26', storage='256GB')
    assert cart2 is None and 'không đủ số lượng' in text           # kho chỉ còn 3


def test_add_to_cart_asks_for_variant_and_rejects_out_of_stock(fake_catalog):
    cart, text = call(tools.add_to_cart, {}, product='galaxy s26')
    assert cart is None and text.startswith('Cần khách chọn rõ phiên bản của Galaxy S26: 256GB Đen Classic, 512GB')
    cart, text = call(tools.add_to_cart, {}, product='galaxy s26', storage='512gb')
    assert cart is None and 'hết hàng' in text


def test_add_accessory_and_remove(fake_catalog):
    cart, _ = call(tools.add_to_cart, {}, product='ốp lưng trong suốt s26')
    assert cart[0]['name'] == 'Ốp Lưng Trong Suốt Galaxy S26'
    cart, text = call(tools.remove_from_cart, {'cart': cart}, product='ốp lưng s26')
    assert cart == [] and text.startswith('Đã bỏ 1 sản phẩm')
    cart, text = call(tools.remove_from_cart, {'cart': []}, product='iphone')
    assert cart is None and text.startswith('Không có')


# ---------------------------------------------------------------- graph chạy nhiều lượt

class ScriptedModel(GenericFakeChatModel):
    model: str = 'scripted'

    def bind_tools(self, tools, **kwargs):
        return self


def test_graph_keeps_state_across_turns(monkeypatch, fake_catalog):
    script = iter([
        AIMessage('Dạ anh/chị cho em biết ngân sách ạ?'),                           # lượt 1: không gọi tool
        AIMessage('', tool_calls=[{'name': 'add_to_cart', 'id': 't1',
                                   'args': {'product': 'galaxy s26', 'storage': '256GB', 'color': 'đen'}}]),
        AIMessage('Dạ em đã thêm Galaxy S26 vào giỏ ạ.'),                            # lượt 2 sau khi gọi tool
    ])
    monkeypatch.setattr(llm, 'open_llm', ScriptedModel(messages=script))
    from phone_sales.graph import build_graph
    g = build_graph(fallback=False, checkpointer=InMemorySaver())
    cfg = {'configurable': {'thread_id': 't'}}

    s1 = g.invoke({'messages': [HumanMessage('tầm 30 triệu muốn máy samsung chụp ảnh đẹp')]}, cfg)
    assert s1['stage'] == 'discovery'
    assert s1['slots'] == {'budget_max': 30_000_000, 'brand': 'Samsung', 'purposes': ['chụp ảnh']}

    s2 = g.invoke({'messages': [HumanMessage('ok chốt galaxy s26 256GB màu đen')]}, cfg)
    assert s2['stage'] == 'closing'
    assert s2['cart'][0]['variant_id'] == 'S26-256-DEN'
    assert s2['slots']['budget_max'] == 30_000_000                    # nhu cầu lượt trước vẫn giữ
    assert s2['sentiment_history'] == ['neutral', 'positive']
    assert s2['messages'][-1].text == 'Dạ em đã thêm Galaxy S26 vào giỏ ạ.'


def test_show_order_form_needs_cart():
    cmd = tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c1',
                                        'args': {'state': {'cart': [PHONE_IN_CART]}}})
    assert cmd.update['order_form'] is True and 'Đã hiện form đặt hàng cho 1 sản phẩm' in cmd.update['messages'][0].content
    empty = tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c2', 'args': {'state': {}}})
    assert 'order_form' not in empty.update and 'Giỏ hàng đang trống' in empty.update['messages'][0].content


def test_show_order_form_not_in_same_turn_as_add_to_cart():
    turn = [HumanMessage('chốt s26 256gb đen'),
            AIMessage('', tool_calls=[{'name': 'add_to_cart', 'args': {}, 'id': 'a'}]),
            ToolMessage('Đã thêm', tool_call_id='a'),
            AIMessage('', tool_calls=[{'name': 'show_order_form', 'args': {}, 'id': 'c1'}])]
    cmd = tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c1',
                                        'args': {'state': {'cart': [PHONE_IN_CART], 'messages': turn}}})
    assert 'order_form' not in cmd.update and 'Chưa mở form' in cmd.update['messages'][0].content
    # Lượt sau khách xác nhận: mở form
    later = turn + [AIMessage('Anh/chị xác nhận đặt hàng không?'), HumanMessage('xác nhận'),
                    AIMessage('', tool_calls=[{'name': 'show_order_form', 'args': {}, 'id': 'c2'}])]
    cmd = tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c2',
                                        'args': {'state': {'cart': [PHONE_IN_CART], 'messages': later}}})
    assert cmd.update['order_form'] is True
