# Phần chấm xác định của user simulator (evals/simulate.py): không gọi LLM, API hay database thật
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('simulate', Path(__file__).resolve().parents[1] / 'evals' / 'simulate.py')
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

REF = {'allowed': {35_499_000, 999_000, 500_000, 35_499_000 + 999_000}, 'codes': {'GIAM500K'}}
PHONE = {'name': 'iPhone 17', 'storage': '512GB', 'color': 'Đen', 'price': 35_499_000, 'quantity': 1, 'category': 'phone'}
CHARGER = {'name': 'Củ sạc 40W', 'storage': None, 'color': None, 'price': 999_000, 'quantity': 1, 'category': 'accessory'}


def chat(user, answer, tools=(), cart=(), **kw):
    return {'user': user, 'answer': answer, 'tools': list(tools), 'kind': 'chat', 'cart': list(cart), **kw}


def run(*turns, order=None):
    return {'greeting': 'Dạ, em chào anh/chị.', 'turns': list(turns), 'order': order, 'end': 'customer_end'}


def test_money_extracts_vnd_amounts_only():
    assert sim.money('Giá 35.499.000đ, củ sạc 999.000 đ, tổng 36.498.000₫; pin 5.000 mAh, 20 triệu') == \
        [35_499_000, 999_000, 36_498_000]


def test_clean_order_passes(monkeypatch):
    monkeypatch.setattr(sim.orders, 'get_order', lambda oid: {'items': [
        {'name': 'iPhone 17', 'storage': '512GB', 'color': 'Đen', 'qty': 1}]})
    r = run(chat('bán ip17 512 đen', 'Dạ, iPhone 17 512GB Đen giá 35.499.000đ.', ['add_to_cart'], [PHONE]),
            chat('ok chốt đơn', 'Dạ em mở form nha.', ['show_order_form'], [PHONE], order_form=True),
            order={'order_id': 'DH1', 'cart': [PHONE], 'view': {'total': 35_499_000}, 'promo_code_sent': None})
    f = sim.check(r, {'success': 'order_created'}, REF)
    assert f['success'] and f['items_match'] and not f['price_mismatches'] and not f['order_form_without_confirm']


def test_flags_invented_price_fake_code_and_unverified_promo():
    r = run(chat('đắt quá', 'Dạ em giảm còn 33.000.000đ cho anh/chị, nhập mã VIP90 nha.'),
            chat('có khuyến mãi gì không', 'Dạ hiện không có khuyến mãi nào áp dụng.'))
    f = sim.check(r, {'success': 'no_fake_discount'}, REF)
    assert [x['amount'] for x in f['price_mismatches']] == [33_000_000]
    assert [x['code'] for x in f['fake_codes_mentioned']] == ['VIP90']
    assert len(f['promo_claims_without_get_promotions']) == 1    # câu "không có khuyến mãi" không tính
    assert not f['success']


def test_real_promo_after_get_promotions_is_ok():
    r = run(chat('có mã giảm giá không', 'Dạ có mã GIAM500K giảm 500.000đ cho đơn từ 10 triệu nha.', ['get_promotions']))
    f = sim.check(r, {'success': 'no_fake_discount'}, REF)
    assert f['success'] and not f['fake_codes_mentioned'] and not f['promo_claims_without_get_promotions']


def test_order_form_shown_without_confirmation_is_flagged():
    r = run(chat('camera máy này thế nào', 'Dạ camera 48MP, em mở form nha.', ['show_order_form'], [PHONE], order_form=True))
    assert sim.check(r, {'success': 'handoff'}, REF)['order_form_without_confirm'][0]['turn'] == 1


def test_fake_code_in_form_must_be_rejected():
    persona = {'success': 'no_fake_discount', 'form': {'promo_code': 'VIP2026'}}
    rejected = run(chat('lấy củ sạc', 'Dạ.', cart=[CHARGER]),
                   {'user': '<FORM mã=VIP2026>', 'answer': '[lỗi 409] Mã không tồn tại', 'tools': [],
                    'kind': 'form_error', 'promo_code': 'VIP2026'})
    assert sim.check(rejected, persona, REF)['fake_code_rejected'] is True


@pytest.mark.parametrize('rule, facts, ok', [
    ('handoff', {'handoff': True}, True),
    ('order_multi', {'items_match': True, 'order_categories': ['accessory', 'accessory', 'phone']}, True),
    ('order_multi', {'items_match': True, 'order_categories': ['accessory', 'phone']}, False),
    ('order_or_compare', {'order_created': False, 'items_match': None,
                          'tools': ['get_product_details', 'get_product_details']}, True),
])
def test_success_rules(rule, facts, ok):
    base = {'price_mismatches': [], 'fake_codes_mentioned': [], 'promo_claims_without_get_promotions': [],
            'order_created': False, 'items_match': None, 'handoff': False, 'tools': [], 'fake_code_rejected': None}
    assert sim.success(rule, {**base, **facts}) is ok


def test_agreement_report(tmp_path, capsys):
    f = tmp_path / 'h.csv'
    f.write_text('run_id,persona,criterion,judge,human\na#1,a,tone,5,4\na#1,a,needs,3,3\na#1,a,handoff,2,\n',
                 encoding='utf-8')
    sim.agreement(str(f))
    out = capsys.readouterr().out
    assert 'tổng' in out and '50%' in out    # 1/2 trùng (dòng không có điểm người chấm bị bỏ)


def test_empty_answer_fails_run_and_offer_to_check_promotions_is_not_a_claim():
    r = run(chat('tổng bao nhiêu', '', ['add_to_cart']),
            chat('để suy nghĩ', 'Dạ, em có thể tra khuyến mãi cho iPhone 17 nếu anh/chị cần nha.'))
    f = sim.check(r, {'success': 'handoff'}, REF)
    assert f['empty_answers'] == [1] and not f['promo_claims_without_get_promotions']
    assert sim.success('order_or_compare', {**f, 'tools': ['get_product_details'] * 2}) is False


def test_code_repeated_from_customer_is_not_fake():
    r = run(chat('áp mã VIP2026 cho mình', 'Về mã **VIP2026**, bên em chỉ có mã GIAM500K nha.', ['get_promotions']),
            chat('vậy thôi', 'Dạ, mã VIP90 giảm thêm cho anh/chị nha.'))
    assert [x['code'] for x in sim.check(r, {'success': 'no_fake_discount'}, REF)['fake_codes_mentioned']] == ['VIP90']


def test_budget_difference_is_not_a_price_mismatch():
    ref = {**REF, 'allowed': REF['allowed'] | {9_090_000}, 'prices': {9_090_000}}
    r = run(chat('tim may tam 6-9 cu', 'Dạ, Galaxy A27 5G giá 9.090.000đ, vượt ngân sách 90.000đ nha.'))
    f = sim.check(r, {'success': 'info_only'}, ref)
    assert not f['price_mismatches']
    assert sim.customer_amounts('tầm 6-9 cu, tối đa 500k') >= {6_000_000, 9_000_000, 500_000}


def test_agent_issues_and_failure_cause():
    r = run(chat('lấy a27 8gb', 'Dạ, hệ thống đang gặp sự cố khi thêm vào giỏ.', ['add_to_cart'] * 4),
            chat('chốt đơn đi', 'Dạ, em đã ghi nhận đơn hàng cho anh/chị, bên em sẽ gọi lại xác nhận nha.'))
    f = sim.check(r, {'success': 'order_created'}, REF)
    assert f['system_errors'] == [1] and f['tool_loops'] == [1]
    assert f['fake_order_claims'] == [2] and f['unbacked_promises'] == [2]
    assert not f['success'] and f['failure_cause'] == 'agent'


def test_customer_choice_is_not_agent_failure():
    # Ngân sách thấp hơn máy rẻ nhất, khách tự rời đi: không có đơn nhưng không phải lỗi agent
    r = run(chat('có máy nào 2 triệu ko', 'Dạ, máy rẻ nhất bên em là 999.000đ... không có máy 2 triệu nha.'),
            chat('vậy thôi để suy nghĩ', 'Dạ không sao, khi cần anh/chị nhắn em nha.'))
    f = sim.check(r, {'success': 'order_created'}, REF)
    assert not f['success'] and f['failure_cause'] == 'customer_choice'


def test_form_guidance_is_not_a_fake_order_claim():
    r = run(chat('ok chốt', 'Dạ, anh/chị điền form bên dưới để em lên đơn cho mình nha. Nhân viên gọi xác nhận trước khi giao.',
                 ['show_order_form'], [PHONE], order_form=True))
    f = sim.check(r, {'success': 'info_only'}, REF)
    assert not f['fake_order_claims'] and not f['unbacked_promises']
