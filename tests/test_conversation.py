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


@pytest.mark.parametrize('text, expected', [
    # Từ simulator: không dấu, viết hoa, "quản lý", "người phụ trách", "staff", bị trừ tiền 2 lần
    ('khong can ho tro, anh muon gap quan ly ngay lap tuc', True),
    ('TOI MUON GAP NGUOI PHU TRACH NGAY', True),
    ('chị muốn gặp quản lý để làm rõ chính sách đổi trả', True),
    ('10tr không được à? gọi staff thật đi', True),
    ('em mua a07s bị trừ tiền 2 lần rồi', True),
    ('don hang bi giao tre 3 ngay roi', True),
    ('chuyển nv giúp mình', True),
    # Không phải xin gặp người
    ('iphone 17 quản lý pin tốt không', False),
    ('quan ly pin tot ko', False),
    ('nhớ nhắn nhân viên gọi xác nhận giúp chú', False),
    ('máy này có hay bị lỗi không', False),
])
def test_handoff_phrases_with_and_without_accents(text, expected):
    from phone_sales.textnorm import normalize_user_text
    assert conv.track_user_message(normalize_user_text(text))['wants_handoff'] is expected
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


CASE = {'name': 'Ốp lưng', 'category': 'accessory', 'accessory_type': 'case', 'price': 1, 'quantity': 1}
CHARGER = {'name': 'Củ sạc', 'category': 'accessory', 'accessory_type': 'charger', 'price': 1, 'quantity': 1}


def test_cross_sell_steps():
    state = {'stage': 'closing', 'cart': [PHONE_IN_CART]}
    hint = conv.cross_sell_hint
    # Bước 1 (lượt chọn máy): máy không kèm củ sạc -> chỉ mời củ sạc, chưa mời ốp lưng, cường lực
    assert conv.cross_sell_step(state) == 'charger'
    text = hint(state)
    assert text.startswith('Bán kèm (bước 1/3, củ sạc)') and "accessory_type='sạc, cáp', genuine_only=True" in text
    assert "accessory_type='ốp lưng, cường lực'" not in text and 'chưa gợi ý ốp lưng, cường lực' in text
    assert 'Gợi ý nhẹ, không ép' in text and 'không cần thì cũng không sao' in text
    # Bước 2: khách lấy củ sạc hoặc không -> mời ốp lưng, cường lực
    for cart in ([PHONE_IN_CART], [PHONE_IN_CART, CHARGER]):
        after = {**state, 'cart': cart, 'cross_sell': ['charger']}
        assert conv.cross_sell_step(after) == 'accessories'
        assert "accessory_type='ốp lưng, cường lực', genuine_only=True" in hint(after) and 'bước 2/3' in hint(after)
    # Bước 3: khách lấy phụ kiện (kể cả tự lấy ở bước 1) hoặc không -> xác nhận đơn, không mời thêm
    assert conv.cross_sell_step({**state, 'cart': [PHONE_IN_CART, CASE], 'cross_sell': ['charger']}) == 'confirm'
    third = {**state, 'cross_sell': ['charger', 'accessories']}
    assert conv.cross_sell_step(third) == 'confirm' and 'Đơn hàng của anh/chị:' in hint(third)
    assert 'không mời thêm phụ kiện' in hint(third)
    assert hint({**state, 'cross_sell': ['charger', 'accessories', 'confirm']}) is None


def test_cross_sell_only_chargers_when_cable_in_box():
    text = conv.cross_sell_hint({'stage': 'closing', 'cart': [{**PHONE_IN_CART, 'cable_in_box': True}]})
    assert "accessory_type='củ sạc', genuine_only=True" in text and 'Mời 2-3 củ sạc chính hãng' in text
    assert 'bên em có củ sạc chính hãng:' in text
    assert 'rẻ tới đắt' in text and 'đã có cáp sạc' in text and 'không mời cáp' in text


def test_cross_sell_starts_with_accessories_when_charger_in_box_or_cart():
    for cart in ([{**PHONE_IN_CART, 'charger_in_box': True}], [PHONE_IN_CART, CHARGER]):
        state = {'stage': 'closing', 'cart': cart}
        assert conv.cross_sell_step(state) == 'accessories'
        text = conv.cross_sell_hint(state)
        assert text.startswith('Bán kèm (bước 1/2, ốp lưng, cường lực)') and 'em đã thêm **[máy' in text and 'Gợi ý nhẹ, không ép' in text
    # Máy kèm củ sạc: mời phụ kiện xong là xác nhận đơn
    done = {'stage': 'closing', 'cart': [{**PHONE_IN_CART, 'charger_in_box': True}], 'cross_sell': ['accessories']}
    assert conv.cross_sell_step(done) == 'confirm'


def test_cross_sell_not_offered_outside_closing_or_when_unhappy():
    state = {'stage': 'closing', 'cart': [PHONE_IN_CART]}
    assert conv.cross_sell_hint({**state, 'stage': 'recommending'}) is None
    assert 'Bán kèm (bước' in conv.describe(state)
    assert 'Bán kèm (bước' not in conv.describe({**state, 'sentiment_history': ['negative']})
    assert conv.cross_sell_step({**state, 'sentiment_history': ['negative']}) is None  # không tính là đã mời


def test_cross_sell_steps_reset_after_order():
    assert conv.steps(['charger'], ['accessories']) == ['charger', 'accessories']
    assert conv.steps(['charger', 'accessories'], None) == []


def test_decline_detection():
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
    # Tool trả thao tác trên giỏ ({'add'/'remove'}); áp lên giỏ hiện tại như reducer của graph để ra giỏ mới
    cmd = tool.invoke({'type': 'tool_call', 'name': tool.name, 'id': 'c1', 'args': {**args, 'state': state}})
    change = cmd.update.get('cart')
    cart = None if change is None else conv.cart_update(state.get('cart'), change)
    return cart, cmd.update['messages'][0].content


def test_cart_update_applies_each_change():
    phone = {'variant_id': 'A', 'name': 'Galaxy S26', 'price': 10, 'quantity': 1}
    charger = {'variant_id': 'B', 'name': 'Củ sạc', 'price': 2, 'quantity': 1}
    cart = conv.cart_update(None, {'add': phone})
    cart = conv.cart_update(cart, {'add': charger})                  # 2 thao tác trong cùng 1 bước: giữ cả hai
    cart = conv.cart_update(cart, {'add': {**phone, 'quantity': 2}})  # cùng phiên bản: cộng số lượng
    assert [(i['variant_id'], i['quantity']) for i in cart] == [('A', 3), ('B', 1)]
    assert [i['variant_id'] for i in conv.cart_update(cart, {'remove': ['A']})] == ['B']
    assert conv.cart_update(cart, []) == []                          # api.py làm trống sau khi đặt/hủy đơn
    assert phone['quantity'] == 1                                    # không sửa món gốc


def test_graph_parallel_add_to_cart_keeps_both(monkeypatch, fake_catalog):
    # Lỗi cũ: model dự phòng gọi 2 add_to_cart song song -> InvalidUpdateError: At key 'cart'
    script = iter([
        AIMessage('', tool_calls=[
            {'name': 'add_to_cart', 'id': 't1', 'args': {'product': 'galaxy s26', 'storage': '256GB', 'color': 'đen'}},
            {'name': 'add_to_cart', 'id': 't2', 'args': {'product': 'ốp lưng trong suốt s26'}}]),
        AIMessage('Dạ em đã thêm máy và ốp lưng vào giỏ.'),
    ])
    monkeypatch.setattr(llm, 'openai_llm', ScriptedModel(messages=script))
    from phone_sales.graph import build_graph
    g = build_graph(fallback=False, checkpointer=InMemorySaver())
    res = g.invoke({'messages': [HumanMessage('lấy galaxy s26 256gb đen với ốp lưng trong suốt')]},
                   {'configurable': {'thread_id': 'p'}})
    assert sorted(i['variant_id'] for i in res['cart']) == ['OP', 'S26-256-DEN']


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
    monkeypatch.setattr(llm, 'openai_llm', ScriptedModel(messages=script))
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
    assert s2['messages'][-1].text == 'Dạ em đã thêm Galaxy S26 vào giỏ.'  # bỏ chữ "ạ"


def test_show_order_form_needs_cart():
    cmd = tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c1',
                                        'args': {'state': {'cart': [PHONE_IN_CART], 'cross_sell': ['charger', 'accessories']}}})
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
                                        'args': {'state': {'cart': [PHONE_IN_CART], 'messages': later,
                                                           'cross_sell': ['charger', 'accessories']}}})
    assert cmd.update['order_form'] is True


def test_show_order_form_waits_for_cross_sell():
    # "Không cần sạc, xác nhận đặt hàng giúp mình" ở bước 1, 2: chưa mở form, mời ốp lưng, cường lực trước
    def form(state):
        return tools.show_order_form.invoke({'type': 'tool_call', 'name': 'show_order_form', 'id': 'c1',
                                             'args': {'state': {'stage': 'closing', **state}}}).update
    for done in ([], ['charger']):
        upd = form({'cart': [PHONE_IN_CART], 'cross_sell': done})
        assert 'order_form' not in upd and 'Chưa mở form: chưa xong bước bán kèm' in upd['messages'][0].content
    assert form({'cart': [PHONE_IN_CART], 'cross_sell': ['charger', 'accessories']})['order_form'] is True  # bước 3
    assert form({'cart': [PHONE_IN_CART, CASE], 'cross_sell': ['charger']})['order_form'] is True   # đã lấy ốp
    acc_only = {**CASE, 'name': 'Cáp', 'accessory_type': 'cable'}
    assert form({'cart': [acc_only]})['order_form'] is True                                          # giỏ không có máy


def test_primary_allows_parallel_tool_calls(monkeypatch):
    # State chịu được nhiều tool cập nhật cùng bước (cart_update...) nên model chính được gọi tool song song
    seen = {}

    class Recording(ScriptedModel):
        def bind_tools(self, tools, **kwargs):
            seen.update(kwargs)
            return self

    monkeypatch.setattr(llm, 'openai_llm', Recording(messages=iter([])))
    from phone_sales.graph import build_graph
    build_graph(fallback=False)
    assert seen == {'parallel_tool_calls': True}


# ---------------------------------------------------------------- khách tự yêu cầu phụ kiện (wanted)

@pytest.mark.parametrize('text, kinds', [
    ('tui muốn đặt ip17, củ sạc 40W, ốp lưng silicon, kính cường lực belkin', ['charger', 'case', 'screen-protector']),
    ('cho mình xin ốp lưng trong suốt', ['case']),
    ('lấy thêm sạc dự phòng', ['power-bank']),        # sạc dự phòng không phải củ sạc
    ('iphone 17 có củ sạc nào không', []),            # hỏi, chưa có ý mua
    ('không cần củ sạc đâu', []),                     # từ chối
])
def test_wanted_of(text, kinds):
    assert list(conv.wanted_of(text)) == kinds


def test_wanted_update_reset_and_drop():
    w = conv.wanted_update(None, {'charger': 'củ sạc 40W', 'case': 'ốp silicon'})
    assert conv.wanted_update(w, {'drop': ['case']}) == {'charger': 'củ sạc 40W'}   # khách bỏ ốp khỏi giỏ
    assert conv.wanted_update(w, None) == {}                                       # đặt/hủy đơn: làm lại


def test_wanted_items_are_added_not_offered():
    ask = 'tui muốn đặt ip17, củ sạc 40W, ốp lưng silicon, kính cường lực belkin'
    state = {'stage': 'closing', 'cart': [PHONE_IN_CART], 'wanted': conv.wanted_of(ask)}
    hint = conv.wanted_hint(state)
    assert 'củ sạc, ốp lưng, cường lực/miếng dán' in hint and 'add_to_cart luôn' in hint and ask in hint
    assert 'kể cả khi máy chưa đủ dung lượng, màu' in hint
    # Bán kèm không mời lại loại khách đã yêu cầu: bỏ qua cả củ sạc lẫn ốp/cường lực -> xác nhận đơn
    assert conv.cross_sell_step(state) == 'confirm'
    # Đã có đủ trong giỏ thì không nhắc nữa
    full = [PHONE_IN_CART, CHARGER, CASE, {**CASE, 'name': 'Kính', 'accessory_type': 'screen-protector'}]
    assert conv.wanted_hint({**state, 'cart': full}) is None
    # Chỉ yêu cầu củ sạc: vẫn mời nhẹ ốp lưng, cường lực
    only_charger = {**state, 'wanted': {'charger': 'lấy củ sạc 40W'}}
    assert conv.cross_sell_step(only_charger) == 'accessories'


def test_remove_wanted_accessory_forgets_request(fake_catalog):
    case = {'variant_id': 'OP', 'name': 'Ốp Lưng Trong Suốt Galaxy S26', 'category': 'accessory',
            'accessory_type': 'case', 'storage': None, 'color': None, 'price': 1, 'quantity': 1}
    cmd = tools.remove_from_cart.invoke({'type': 'tool_call', 'name': 'remove_from_cart', 'id': 'r1',
                                         'args': {'product': 'ốp lưng s26', 'state': {'cart': [case]}}})
    assert cmd.update['wanted'] == {'drop': ['case']} and cmd.update['cart'] == {'remove': ['OP']}


def test_cross_sell_templates_do_not_leak_instructions():
    # Lỗi cũ: chỉ dẫn "(2 món, từ rẻ tới đắt; dòng tool có ghi công suất...)" nằm trong dòng mẫu, model chép cho khách.
    # Phần mẫu (sau "Trả lời theo mẫu:") bỏ [chỗ điền] đi thì chỉ còn chữ gửi khách
    import re
    phone = {**PHONE_IN_CART, 'cable_in_box': True}
    states = [{'stage': 'closing', 'cart': [phone]},
              {'stage': 'closing', 'cart': [PHONE_IN_CART]},
              {'stage': 'closing', 'cart': [phone], 'cross_sell': ['charger']},
              {'stage': 'closing', 'cart': [phone], 'cross_sell': ['charger', 'accessories']}]
    for state in states:
        hint = conv.cross_sell_hint(state)
        template = re.sub(r'\[[^\]]*\]', '', hint.split('Trả lời theo mẫu:', 1)[1])
        for word in ('tool', 'find_accessories', 'genuine', 'từ rẻ tới đắt', 'Mời ', 'show_order_form', '2-3'):
            assert word not in template, (word, template)
        assert '**' in template                                       # tên món / tổng tiền in đậm theo quy tắc 11b


def _added_turn(*results):
    # 1 lượt: khách nhắn, model gọi add_to_cart song song, tool trả kết quả
    calls = [{'name': 'add_to_cart', 'args': {}, 'id': f'a{i}'} for i in range(len(results))]
    return [HumanMessage('bán cho tui ip17 512gb đen, củ sạc 40w, kính cường lực belkin'), AIMessage('', tool_calls=calls),
            *[ToolMessage(r, tool_call_id=f'a{i}') for i, r in enumerate(results)]]


def test_added_hint_lists_each_item_then_total():
    messages = _added_turn('Đã thêm 1 x iPhone 17 512GB Đen (35.499.000đ). Giỏ có 1 sản phẩm, tổng 35.499.000đ.',
                           'Đã thêm 1 x Bộ Tiếp Hợp Nguồn Động 40W  (999.000đ). Giỏ có 1 sản phẩm, tổng 999.000đ.',
                           'Đã thêm 2 x Củ Sạc Không Kèm Cáp 60W (Đen)  (1.089.720đ). Giỏ có 1 sản phẩm.',
                           'iPhone 17 không có phiên bản dung lượng \'1TB\'.')               # thêm lỗi: không liệt kê
    cart = [{'name': 'iPhone 17', 'category': 'phone', 'price': 35_499_000, 'quantity': 1},
            {'name': 'Củ sạc', 'category': 'accessory', 'accessory_type': 'charger', 'price': 999_000, 'quantity': 1},
            {'name': 'Củ sạc 60W', 'category': 'accessory', 'accessory_type': 'charger', 'price': 1_089_720, 'quantity': 2}]
    hint = conv.added_hint({'messages': messages, 'cart': cart, 'wanted': {'charger': 'củ sạc 40w'}})
    assert ('- **iPhone 17 512GB Đen**: 35.499.000đ\n- **Bộ Tiếp Hợp Nguồn Động 40W**: 999.000đ\n'
            '- 2 x **Củ Sạc Không Kèm Cáp 60W (Đen)**: 1.089.720đ\n\n'
            'Đã thêm đủ 3 món anh/chị yêu cầu vào giỏ. Tổng cộng **38.677.440đ**.') in hint
    # Khách yêu cầu cường lực mà chưa thêm được: không nói "đủ", nêu món còn thiếu
    missing = conv.added_hint({'messages': messages, 'cart': cart,
                               'wanted': {'charger': 'x', 'screen-protector': 'kính belkin'}})
    assert 'còn cường lực/miếng dán em chưa thêm được' in missing and 'đủ' not in missing.split('Trả lời theo mẫu')[1]
    # Chỉ 1 món: không cần mẫu này
    assert conv.added_hint({'messages': _added_turn('Đã thêm 1 x A (1đ). Giỏ có 1 sản phẩm.'), 'cart': cart}) is None


def test_added_hint_replaces_confirm_template_in_describe():
    messages = _added_turn('Đã thêm 1 x iPhone 17 512GB Đen (35.499.000đ). Giỏ.', 'Đã thêm 1 x Kính Belkin (1.177.000đ). Giỏ.')
    state = {'stage': 'closing', 'messages': messages, 'cart': [PHONE_IN_CART, CASE],
             'wanted': {'case': 'ốp'}, 'cross_sell': ['accessories']}
    text = conv.describe(state)
    assert 'Dạ, em đã thêm vào giỏ:' in text and 'Đơn hàng của anh/chị:' not in text    # không lặp danh sách


def test_customer_hint_for_returning_customer():
    profile = {'customer_id': 'KH1', 'name': 'Nguyễn Văn A', 'order_count': 1,
               'last_order': {'order_id': 'DH1', 'date': '02/11/2026', 'items': ['iPhone 17 512GB Đen']}}
    text = conv.describe({'customer': profile})
    assert 'Khách quen: Nguyễn Văn A' in text and 'iPhone 17 512GB Đen' in text
    assert 'Khách quen' not in conv.describe({})
