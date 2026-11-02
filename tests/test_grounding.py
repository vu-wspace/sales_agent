import unicodedata

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from phone_sales import llm
from phone_sales.conversation import track_user_message
from phone_sales.grounding import sources_of, ungrounded
from phone_sales.textnorm import clean_answer, normalize_user_text


# ---------------------------------------------------------------- chuẩn hóa tin nhắn của khách

@pytest.mark.parametrize('text, expected', [
    ('ko cần đâu', 'không cần đâu'),
    ('mik mún mua ip 17 tầm 20tr', 'mình muốn mua ip 17 tầm 20 triệu'),
    ('giá bn vậy e', 'giá bao nhiêu vậy em'),
    ('xem đc ko', 'xem được không'),
    ('huỷ đơn giúp, pin khoẻ ko', 'hủy đơn giúp, pin khỏe không'),   # dấu kiểu cũ -> kiểu mới
    ('Hoàng hoà', 'Hoàng hòa'),                                        # "oà" giữa từ giữ nguyên
    ('đẹppppp quááá', 'đẹp quá'),
    ('100 k thì sao', '100 k thì sao'),                               # "k" sau số là tiền, không phải "không"
    ('Galaxy Z Fold 7', 'Galaxy Z Fold 7'),
    ('  chip​  A19 \n ', 'chip A19'),
])
def test_normalize_user_text(text, expected):
    assert normalize_user_text(text) == expected


def test_normalize_composes_decomposed_accents():
    # Bộ gõ gửi dấu rời (NFD) thì luật từ khóa có dấu không khớp; chuẩn hóa xong mới nhận ra từ chối
    nfd = unicodedata.normalize('NFD', 'không cần đâu')
    assert not track_user_message(nfd)['declined']
    assert track_user_message(normalize_user_text(nfd))['declined']


def test_clean_answer_strips_thinking_and_blank_lines():
    assert clean_answer('<think>tính giá</think>\n\nDạ   \n\n\n\nem gửi nhé') == 'Dạ\n\nem gửi nhé'


# ---------------------------------------------------------------- số liệu không rõ nguồn

SOURCES = ['- 256GB | Đen | 28.999.000đ | còn hàng\nPin: 4900mAh\nĐổi trả 30 ngày', 'tầm 20 triệu']


def test_grounded_numbers_pass():
    # Cùng số khác cách viết (4.900 mAh / 4900mAh), làm tròn giá sang "triệu", ngân sách khách nói
    answer = 'Giá 28.999.000đ, pin 4.900 mAh, khoảng 29 triệu, đổi trong 30 ngày, vừa ngân sách 20 triệu.'
    assert ungrounded(answer, SOURCES) == []


def test_ungrounded_numbers_are_reported():
    answer = 'Giá 27.990.000đ, pin 5000mAh, camera 200MP, bảo hành 24 tháng, sạc 45W, RAM 12GB, chỉ 25 triệu.'
    assert ungrounded(answer, SOURCES) == ['27.990.000đ', '5000mAh', '200MP', '24 tháng', '45W', '12GB', '25 triệu']


def test_numbers_without_unit_are_ignored():
    assert ungrounded('Em gợi ý 3 máy, 2 dung lượng, mỗi máy 1 dòng.', []) == []


def test_old_answers_are_not_sources():
    # Số model đã bịa ở lượt trước không được coi là có nguồn
    messages = [HumanMessage('giá s26?'), AIMessage('Dạ 25.000.000đ ạ'), ToolMessage('29.990.001đ', tool_call_id='t')]
    assert ungrounded('Dạ 25.000.000đ ạ', sources_of(messages)) == ['25.000.000đ']


def test_web_search_is_not_a_source():
    # Thông số, giá chỉ theo dữ liệu cửa hàng: số lấy từ web (vd camera S26 Ultra 200 MP) phải viết lại
    messages = [HumanMessage('s26 camera bao nhiêu MP'),
                ToolMessage('Thông số: chưa có thông số chi tiết', tool_call_id='a', name='get_product_details'),
                ToolMessage('Galaxy S26 Ultra camera 200MP', tool_call_id='b', name='web_search')]
    assert ungrounded('Camera chính 200 MP ạ', sources_of(messages)) == ['200 MP']


class ScriptedModel(GenericFakeChatModel):
    model: str = 'scripted'

    def bind_tools(self, tools, **kwargs):
        return self


def test_graph_rewrites_answer_with_ungrounded_price(monkeypatch):
    script = iter([AIMessage('Dạ iPhone 17 giá 25.000.000đ ạ.'),           # bịa giá
                   AIMessage('Dạ em chưa tra giá, anh/chị chờ em chút ạ.')])  # viết lại
    monkeypatch.setattr(llm, 'open_llm', ScriptedModel(messages=script))
    from phone_sales.graph import build_graph
    g = build_graph(fallback=False, checkpointer=InMemorySaver())
    res = g.invoke({'messages': [HumanMessage('iphone 17 giá bn')]}, {'configurable': {'thread_id': 'g'}})
    texts = [m.text for m in res['messages']]
    assert texts == ['iphone 17 giá bn', 'Dạ em chưa tra giá, anh/chị chờ em chút.']  # bản nháp sai không lưu


def test_graph_keeps_grounded_answer(monkeypatch):
    script = iter([AIMessage('Dạ anh/chị cho em biết ngân sách khoảng 20 triệu đúng không ạ?')])
    monkeypatch.setattr(llm, 'open_llm', ScriptedModel(messages=script))
    from phone_sales.graph import build_graph
    g = build_graph(fallback=False, checkpointer=InMemorySaver())
    res = g.invoke({'messages': [HumanMessage('tầm 20tr')]}, {'configurable': {'thread_id': 'g'}})
    assert res['messages'][-1].text == 'Dạ anh/chị cho em biết ngân sách khoảng 20 triệu đúng không?'  # bỏ chữ "ạ"


def test_clean_answer_removes_a_particle():
    # Cửa hàng không dùng "ạ"; "Dạ", "lạ" giữ nguyên
    assert clean_answer('Dạ, anh/chị chọn màu nào ạ? Vâng ạ. Máy lạ ạ!') == 'Dạ, anh/chị chọn màu nào? Vâng. Máy lạ!'
