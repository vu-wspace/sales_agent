import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from phone_sales import tools

PHONES = [
    {'product_id': 'apple-iphone-18-pro', 'name': 'iPhone 18 Pro', 'brand': 'Apple'},
    {'product_id': 'apple-iphone-18-pro-max', 'name': 'iPhone 18 Pro Max', 'brand': 'Apple'},
    {'product_id': 'apple-iphone-17e', 'name': 'iPhone 17e', 'brand': 'Apple'},
    {'product_id': 'samsung-galaxy-s26', 'name': 'Galaxy S26', 'brand': 'Samsung'},
    {'product_id': 'samsung-galaxy-s26-plus', 'name': 'Galaxy S26+', 'brand': 'Samsung'},
    {'product_id': 'samsung-galaxy-s26-ultra', 'name': 'Galaxy S26 Ultra', 'brand': 'Samsung'},
    {'product_id': 'samsung-galaxy-a57-5g', 'name': 'Galaxy A57 5G', 'brand': 'Samsung'},
]


def names(rows):
    return [r['name'] for r in rows]


@pytest.mark.parametrize('query, expected', [
    ('iPhone 18 Pro', ['iPhone 18 Pro']),               # tên trùng khớp không kéo theo bản Pro Max
    ('ip 18 promax', ['iPhone 18 Pro Max']),
    ('iphone 18 pro max 256gb', ['iPhone 18 Pro Max']),  # dung lượng không tính vào tên
    ('samsung s26', ['Galaxy S26']),
    ('S26+', ['Galaxy S26+']),
    ('điện thoại a57', ['Galaxy A57 5G']),
    ('iphone 17e', ['iPhone 17e']),
    ('nokia', []),
])
def test_match(query, expected):
    assert names(tools._match(PHONES, query)) == expected


def test_match_family_without_exact():
    assert names(tools._match(PHONES, 's26', prefer_exact=False)) == ['Galaxy S26', 'Galaxy S26+', 'Galaxy S26 Ultra']


def test_storage_key_orders_by_capacity():
    labels = ['1TB', '256GB - RAM 12GB', '64GB', '2TB', '256GB - RAM 8GB']
    assert sorted(labels, key=tools._storage_key) == ['64GB', '256GB - RAM 8GB', '256GB - RAM 12GB', '1TB', '2TB']


@pytest.mark.parametrize('storage, expected', [
    ('', ['256GB - RAM 12GB', '256GB - RAM 8GB', '512GB - RAM 12GB', '128GB']),
    ('256GB', ['256GB - RAM 12GB', '256GB - RAM 8GB']),            # chỉ dung lượng: mọi RAM
    ('256GB - RAM 8GB', ['256GB - RAM 8GB']),                       # nêu cả RAM: đúng bản đó (lỗi từ simulator)
    ('12GB/256GB', ['256GB - RAM 12GB']),                           # RAM viết trước
    ('RAM 8GB', ['256GB - RAM 8GB']),                               # chỉ nêu RAM
    ('256GB - RAM 16GB', ['256GB - RAM 12GB', '256GB - RAM 8GB']),  # RAM không có: quay về so dung lượng
    ('1TB', []),
])
def test_filter_storage_distinguishes_ram(storage, expected):
    variants = [{'storage': s} for s in ['256GB - RAM 12GB', '256GB - RAM 8GB', '512GB - RAM 12GB', '128GB']]
    assert [v['storage'] for v in tools._filter_storage(variants, storage)] == expected


@pytest.mark.parametrize('text, expected', [
    ('ốp lưng', 'case'), ('cường lực', 'screen-protector'), ('sạc dự phòng', 'power-bank'),
    ('củ sạc', 'charger'), ('cáp', 'cable'), ('case', 'case'), ('bàn phím', None),
])
def test_accessory_type(text, expected):
    assert tools._accessory_type(text) == expected


def test_price_and_stock_format():
    assert tools._price(38999000) == '38.999.000đ'
    assert [tools._stock(n) for n in (0, 3, 20)] == ['hết hàng', 'sắp hết', 'còn hàng']  # không lộ số lượng


@pytest.fixture
def fake_db(monkeypatch):
    calls = []

    def query(sql, params=()):
        calls.append((sql, params))
        if 'array_agg' in sql:  # search_products
            return [
                {'product_id': 'samsung-galaxy-a57-5g', 'name': 'Galaxy A57 5G', 'brand': 'Samsung',
                 'min_price': 12490000, 'max_price': 16490280, 'stock': 40, 'storages': ['256GB - RAM 8GB', '128GB']},
                {'product_id': 'samsung-galaxy-s26', 'name': 'Galaxy S26', 'brand': 'Samsung',
                 'min_price': 29990001, 'max_price': 35990000, 'stock': 0, 'storages': ['512GB', '256GB']},
            ]
        if 'FROM compatibility' in sql:
            return [{'name': 'Ốp Lưng Trong Suốt Galaxy S26', 'kind': 'case', 'price': 490320, 'stock': 12}]
        if 'FROM variants' in sql:
            return [
                {'variant_id': 'B', 'color': 'Đen', 'storage': '512GB', 'price': 35990000, 'stock': 0},
                {'variant_id': 'A', 'color': 'Đen', 'storage': '256GB', 'price': 29990001, 'stock': 9},
            ]
        return [{**p, 'description': 'Mô tả', 'attributes': {
            'ram_by_storage': {'256GB': '12GB'}, 'specs': {'Chip': 'Snapdragon'}, 'estimated_price_skus': ['B']}}
            for p in PHONES]
    monkeypatch.setattr(tools, '_query', query)
    return calls


def test_search_products_filters_and_formats(fake_db):
    out = tools.search_products.invoke({'max_price': 15000000})
    assert out == ('Sản phẩm đang bán (giá đã gồm VAT):\n'
                   '- Galaxy A57 5G (Samsung): 12.490.000đ - 16.490.280đ | 128GB, 256GB - RAM 8GB | còn hàng\n'
                   + tools.NO_SPECS_IN_LIST)


def test_search_products_lists_slightly_over_budget(fake_db):
    # Up-sell: S26 (29.990.001đ) vượt ngân sách 28 triệu chưa tới 10% -> liệt kê riêng, kèm số tiền vượt
    out = tools.search_products.invoke({'max_price': 28_000_000})
    listed, above = out.split('\nNhỉnh hơn ngân sách')
    assert 'Galaxy A57 5G' in listed and 'Galaxy S26' not in listed
    assert '- Galaxy S26 (Samsung): 29.990.001đ - 35.990.000đ | 256GB, 512GB | hết hàng | vượt 1.990.001đ' in above
    assert 'Nhỉnh hơn' not in tools.search_products.invoke({'max_price': 20_000_000})   # S26 vượt quá 10%


def test_search_products_exact_name_points_to_variants(fake_db):
    # "muốn mua Galaxy S26": đúng tên 1 máy -> nhắc model gọi get_product_details để liệt kê phiên bản
    assert "get_product_details('Galaxy S26')" in tools.search_products.invoke({'query': 'galaxy s26'})
    assert 'get_product_details' not in tools.search_products.invoke({'query': 'galaxy'})


def test_search_products_no_match(fake_db):
    assert tools.search_products.invoke({'query': 'xiaomi'}).startswith('Không có sản phẩm khớp')


def test_product_details(fake_db):
    out = tools.get_product_details.invoke({'product': 'galaxy s26'})
    assert 'RAM: 256GB: 12GB' in out and 'Chip: Snapdragon' in out
    assert out.endswith('- 256GB | Đen | 29.990.001đ | còn hàng\n- 512GB | Đen | 35.990.000đ (giá tham khảo) | hết hàng')


def test_product_details_says_when_specs_missing(monkeypatch, fake_db):
    # Dòng Galaxy S/Z không có thông số: tool phải nói rõ, không thì model tự bịa chip, pin...
    assert tools.NO_SPECS not in tools.get_product_details.invoke({'product': 'galaxy s26'})
    fake = tools._query
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): [
        {**p, 'attributes': {}} if 'attributes' in p else p for p in fake(sql, params)])
    out = tools.get_product_details.invoke({'product': 'galaxy s26'})
    assert tools.NO_SPECS in out and 'Chip' not in out


def test_product_details_filters_storage(fake_db):
    out = tools.get_product_details.invoke({'product': 'galaxy s26', 'storage': '512 GB'})
    assert '256GB |' not in out and '512GB | Đen' in out


def test_product_details_lists_bigger_storage_for_upsell(monkeypatch, fake_db):
    # 512GB hết hàng thì không gợi ý; còn hàng thì ghi giá và số tiền chênh so với bản khách chọn
    assert 'Dung lượng lớn hơn' not in tools.get_product_details.invoke({'product': 'galaxy s26', 'storage': '256gb'})
    fake = tools._query
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): [
        {**v, 'stock': 4} if v.get('variant_id') == 'B' else v for v in fake(sql, params)])
    out = tools.get_product_details.invoke({'product': 'galaxy s26', 'storage': '256gb'})
    assert out.endswith('Dung lượng lớn hơn còn hàng: 512GB từ 35.990.000đ (thêm 5.999.999đ)')
    assert 'Dung lượng lớn hơn' not in tools.get_product_details.invoke({'product': 'galaxy s26'})  # chưa chọn bản


def test_product_details_ambiguous_asks_back(fake_db):
    out = tools.get_product_details.invoke({'product': 'iphone 18'})
    assert out.startswith("Có nhiều máy khớp 'iphone 18': iPhone 18 Pro, iPhone 18 Pro Max")


def test_find_accessories(fake_db):
    out = tools.find_accessories.invoke({'phone': 's26', 'accessory_type': 'ốp lưng'})
    assert out == 'Phụ kiện tương thích với Galaxy S26:\n- Ốp Lưng Trong Suốt Galaxy S26: 490.320đ | còn hàng'
    assert fake_db[-1][1] == ('samsung-galaxy-s26', ['case'], ['case'])


ACCESSORY_ROWS = [
    {'name': 'Bộ sạc USB-C 20W', 'brand': 'Apple', 'kind': 'charger', 'price': 539000, 'stock': 10},
    {'name': 'Bộ Sạc Gắn Tường 2 Cổng Belkin BoostCharge (42W)', 'brand': 'Apple', 'kind': 'charger', 'price': 100, 'stock': 10},
    {'name': 'Củ Sạc Kèm Cáp 45W', 'brand': 'Samsung', 'kind': 'charger', 'price': 1168363, 'stock': 10},
    {'name': 'Sạc Không Dây Nam Châm Magnet', 'brand': 'Samsung', 'kind': 'charger', 'price': 990360, 'stock': 10},
    {'name': 'Cáp Sạc USB-C 60W (1m)', 'brand': 'Apple', 'kind': 'cable', 'price': 539000, 'stock': 10},
    {'name': 'Cáp Bện USB-C sang USB-C của Beats', 'brand': 'Apple', 'kind': 'cable', 'price': 519000, 'stock': 10},
    {'name': 'Ốp Lưng MagSafe', 'brand': 'Apple', 'kind': 'case', 'price': 1499000, 'stock': 10},
]
IPHONE_BOX = {'specs': {'Trong Hộp Sản Phẩm': 'iPhone với iOS 27 Cáp Sạc USB‑C (1 m) Tài liệu iPhone Và Môi Trường...'}}


@pytest.fixture
def fake_accessories(monkeypatch):
    phones = [{**p, 'attributes': IPHONE_BOX if p['brand'] == 'Apple' else {}} for p in PHONES]

    def query(sql, params=()):
        if 'compatibility' not in sql:
            return phones
        kinds = params[1]
        return [dict(r) for r in ACCESSORY_ROWS if kinds is None or r['kind'] in kinds]
    monkeypatch.setattr(tools, '_query', query)


def test_in_box_reads_apple_specs_and_samsung_default():
    # iPhone: "Cáp Sạc USB-C" có chữ "sạc" nhưng là cáp, không phải củ sạc
    assert tools._in_box({'brand': 'Apple', 'attributes': IPHONE_BOX}) == {
        'cable': True, 'charger': False, 'text': 'có cáp sạc, không kèm củ sạc'}
    assert tools._in_box({'brand': 'Samsung', 'attributes': {}})['charger'] is False
    assert tools._in_box({'brand': 'Xiaomi', 'attributes': {}}) is None


def test_find_charger_and_cable_genuine_first_with_box_contents(fake_accessories):
    out = tools.find_accessories.invoke({'phone': 'iphone 18 pro', 'accessory_type': 'sạc, cáp'})
    lines = out.splitlines()
    assert lines[0] == 'Trong hộp iPhone 18 Pro: có cáp sạc, không kèm củ sạc.'
    # Chính hãng Apple trước hàng Belkin/Beats dù đắt hơn; mỗi loại 2 món
    assert lines[2:] == [
        '- Bộ sạc USB-C 20W: 539.000đ | chính hãng Apple | còn hàng',
        '- Bộ Sạc Gắn Tường 2 Cổng Belkin BoostCharge (42W): 100đ | còn hàng',
        '- Cáp Sạc USB-C 60W (1m): 539.000đ | chính hãng Apple | còn hàng',
        '- Cáp Bện USB-C sang USB-C của Beats: 519.000đ | còn hàng',
        '(còn 2 phụ kiện khác, xem tiếp bằng page=2)',
    ]
    samsung = tools.find_accessories.invoke({'phone': 'galaxy s26', 'accessory_type': 'sạc'})
    assert samsung.splitlines()[2] == '- Củ Sạc Kèm Cáp 45W: 1.168.363đ | chính hãng Samsung | còn hàng'
    # 'củ sạc' bỏ sạc không dây; hộp Galaxy đã có cáp nên bỏ cả củ sạc kèm cáp
    wall = tools.find_accessories.invoke({'phone': 'galaxy s26', 'accessory_type': 'củ sạc'})
    assert 'Nam Châm' in samsung and 'Nam Châm' not in wall
    assert 'Kèm Cáp 45W' in samsung and 'Kèm Cáp 45W' not in wall
    assert 'Củ Sạc' not in wall
    # Hàng chính hãng trước, tách mục hãng khác
    assert samsung.splitlines()[4] == 'Hãng khác (không phải chính hãng Samsung):'
    # Tìm ốp lưng thì không cần dòng "Trong hộp"
    assert 'Trong hộp' not in tools.find_accessories.invoke({'phone': 'iphone 18 pro', 'accessory_type': 'ốp lưng'})


def test_find_accessories_pages_of_five(monkeypatch):
    rows = [{'name': f'Ốp {i}', 'kind': 'case', 'price': 100000 * i, 'stock': 10} for i in range(1, 9)]
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): rows if 'compatibility' in sql else PHONES)
    page1 = tools.find_accessories.invoke({'phone': 's26'})
    assert page1.count('\n- ') == 5 and page1.endswith('(còn 3 phụ kiện khác, xem tiếp bằng page=2)')
    page2 = tools.find_accessories.invoke({'phone': 's26', 'page': 2})
    assert 'Ốp 6' in page2 and 'Ốp 8' in page2 and 'xem tiếp' not in page2
    assert tools.find_accessories.invoke({'phone': 's26', 'page': 3}).startswith('Đã xem hết 8 phụ kiện')


def test_find_accessories_unknown_type(fake_db):
    assert tools.find_accessories.invoke({'phone': 's26', 'accessory_type': 'bàn phím'}).startswith('Không rõ loại phụ kiện')


@pytest.mark.parametrize('tool, args', [
    (tools.search_products, {}),
    (tools.get_product_details, {'product': 's26'}),
    (tools.find_accessories, {'phone': 's26'}),
])
def test_db_tools_handle_backend_error(monkeypatch, tool, args):
    def boom(*a, **k):
        raise ConnectionError('postgres down')
    monkeypatch.setattr(tools, '_query', boom)
    assert tool.invoke(args).startswith('Lỗi hệ thống khi tra cứu kho hàng (ConnectionError)')


def test_search_policy_handles_backend_error(monkeypatch):
    def boom():
        raise ConnectionError('qdrant down')
    monkeypatch.setattr(tools, '_policy_store', boom)
    out = tools.search_policy.invoke({'query': 'đổi trả'})
    assert out.startswith('Lỗi hệ thống khi tra cứu dữ liệu nội bộ (ConnectionError)')


def doc(source: str, text: str) -> Document:
    return Document(page_content=text, metadata={'source': source, 'title': 'T', 'section': 'S'})


def fake_policy_store(monkeypatch, scored):
    class Store:
        def similarity_search_with_score(self, query, k):
            return scored[:k]
    monkeypatch.setattr(tools, '_policy_store', Store)


def test_search_policy_groups_by_file_with_its_prompt(monkeypatch):
    fake_policy_store(monkeypatch, [
        (doc('doi-tra.md', 'Đổi trả - Lỗi 30 ngày\nĐổi mới'), 0.60),
        (doc('bao-hanh-apple.md', 'Bảo hành - 12 tháng'), 0.57),
        (doc('doi-tra.md', 'Đổi trả - Hoàn tiền\n7-15 ngày'), 0.54),
    ])
    out = tools.search_policy.invoke({'query': 'máy lỗi'})
    assert out.startswith(tools.POLICY_GUIDE)
    doi_tra = out.index('=== [doi-tra.md] ===')
    assert doi_tra < out.index('=== [bao-hanh-apple.md] ===')     # file có đoạn liên quan nhất đứng trước
    assert out.count('=== [doi-tra.md] ===') == 1                 # 2 đoạn cùng file gom chung 1 nhóm
    assert f"Hướng dẫn trả lời: {tools.POLICY_PROMPTS['doi-tra.md']}" in out
    assert out.index('Đổi mới') < out.index('7-15 ngày') < out.index('=== [bao-hanh-apple.md] ===')


def test_search_policy_drops_weak_trailing_chunks(monkeypatch):
    fake_policy_store(monkeypatch, [
        (doc('tra-gop.md', 'Trả góp - CCCD'), 0.69),
        (doc('giao-hang.md', 'Giao hàng - phí'), 0.50),   # kém top 1 quá 0.15 -> bỏ
    ])
    out = tools.search_policy.invoke({'query': 'không có thẻ tín dụng trả góp được không'})
    assert 'tra-gop.md' in out and 'giao-hang.md' not in out


def test_search_policy_off_topic_returns_not_found(monkeypatch):
    fake_policy_store(monkeypatch, [(doc('tra-gop.md', 'Trả góp'), 0.11)])
    out = tools.search_policy.invoke({'query': 'có nên đầu tư bitcoin'})
    assert out.startswith('Không tìm thấy chính sách liên quan') and 'Trả góp' not in out


def test_every_policy_file_has_a_prompt():
    from phone_sales.config import POLICY_DIR
    assert set(tools.POLICY_PROMPTS) == {f.name for f in POLICY_DIR.glob('*.md')}


def test_web_search_handles_backend_error(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError()
    monkeypatch.setattr(tools.tavily_client, 'search', boom)
    out = tools.web_search.invoke({'query': 'đánh giá iphone 18'})
    assert out.startswith('Lỗi hệ thống khi tìm kiếm trên web (TimeoutError)')


def test_agent_main_has_all_tools():
    assert [t.name for t in tools.get_agent_tools('agent_main')] == [
        'search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search', 'get_promotions']


def test_find_accessories_genuine_only(monkeypatch):
    # Bán kèm: loại có hàng chính hãng thì bỏ hàng hãng khác (Belkin, củ sạc Apple cho Galaxy)
    rows = [{'name': 'Củ Sạc Không Kèm Cáp 60W', 'brand': 'Samsung', 'kind': 'charger', 'price': 1089720, 'stock': 10},
            {'name': 'Bộ Sạc Gắn Tường Belkin (42W)', 'brand': 'Apple', 'kind': 'charger', 'price': 951000, 'stock': 10},
            {'name': 'Bộ Tiếp Hợp Nguồn 35W', 'brand': 'Apple', 'kind': 'charger', 'price': 1471000, 'stock': 10}]
    phone = {'product_id': 'samsung-galaxy-s26', 'name': 'Galaxy S26', 'brand': 'Samsung', 'attributes': {}}
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): rows if 'compatibility' in sql else [phone])
    out = tools.find_accessories.invoke({'phone': 'galaxy s26', 'accessory_type': 'củ sạc', 'genuine_only': True})
    assert '60W' in out and 'Belkin' not in out and '35W' not in out and 'Hãng khác' not in out
    assert 'Belkin' in tools.find_accessories.invoke({'phone': 'galaxy s26', 'accessory_type': 'củ sạc'})


def test_find_accessories_genuine_only_keeps_kind_without_genuine(monkeypatch):
    # Apple không bán miếng dán màn hình: cường lực cho iPhone chỉ có Belkin -> vẫn giữ, không làm mất cả loại
    rows = [{'name': 'Ốp Lưng Trong Suốt MagSafe cho iPhone 17', 'brand': 'Apple', 'kind': 'case', 'price': 1499000, 'stock': 10},
            {'name': 'Ốp Lưng Beats cho iPhone 17', 'brand': 'Apple', 'kind': 'case', 'price': 1299000, 'stock': 10},
            {'name': 'Miếng Dán Màn Hình Belkin UltraGlass 2 cho iPhone 17', 'brand': 'Apple', 'kind': 'screen-protector',
             'price': 1177000, 'stock': 10}]
    phone = {'product_id': 'apple-iphone-17', 'name': 'iPhone 17', 'brand': 'Apple', 'attributes': {}}
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): rows if 'compatibility' in sql else [phone])
    out = tools.find_accessories.invoke({'phone': 'iphone 17', 'accessory_type': 'ốp lưng, cường lực', 'genuine_only': True})
    assert 'Ốp Lưng Trong Suốt MagSafe' in out and 'Beats' not in out   # có ốp chính hãng: bỏ ốp Beats
    assert 'Belkin UltraGlass 2' in out                                 # cường lực chỉ có Belkin: giữ


IPHONE17_SPECS = {'Pin Và Nguồn Điện': 'Khả năng sạc nhanh: Sạc lên đến 50% trong 20 phút với bộ tiếp hợp 40W trở lên '
                                       '(được bán riêng) sử dụng cáp sạc USB‑C Sạc lên đến 50% trong 30 phút với bộ tiếp '
                                       'hợp 30W trở lên sử dụng Bộ Sạc MagSafe',
                  'MagSafe và Sạc Không Dây': 'Sạc không dây MagSafe lên đến 25W Sạc không dây Qi2 lên đến 25W'}


def test_charging_spec_and_charger_fit():
    c = tools._charging({'name': 'iPhone 17', 'attributes': {'specs': IPHONE17_SPECS}})
    assert (c['wired'], c['wireless']) == (40, 25)            # 40W qua cáp, không lấy 30W của MagSafe
    assert 'sạc nhanh 50% trong 20 phút cần bộ tiếp hợp từ 40W trở lên' in c['text']
    fit = lambda name: tools._charger_fit({'name': name, 'kind': 'charger'}, c)
    assert fit('Bộ Tiếp Hợp Nguồn Động 40W Với Công Suất Tối Đa 60W') == ' | đủ công suất sạc nhanh cho máy (từ 40W)'
    assert fit('Bộ Tiếp Hợp Nguồn Cổng Kép 35W') == ' | dưới 40W: sạc được nhưng không đạt sạc nhanh'
    assert fit('Bộ Sạc MagSafe (1 m)') == ' | sạc không dây tối đa 25W trên máy này'
    assert tools._charger_fit({'name': 'Cáp Sạc USB-C 60W', 'kind': 'cable'}, c) == ''   # chỉ ghi cho củ sạc


def test_no_charging_note_without_specs():
    # Galaxy chưa có thông số sạc: không ghi công suất cần, không để model tự đoán
    assert tools._charging({'name': 'Galaxy S26', 'attributes': {}}) == {}
    assert tools._charger_fit({'name': 'Củ Sạc 60W', 'kind': 'charger'}, {}) == ''
    wireless_only = tools._charging({'name': 'iPhone 18 Pro', 'attributes': {'specs': {
        'MagSafe': 'Sạc không dây MagSafe lên đến 25W'}}})
    assert 'wired' not in wireless_only and 'Nói rõ củ sạc' not in wireless_only['text']


def test_find_chargers_puts_fast_enough_first(monkeypatch):
    phone = {'product_id': 'apple-iphone-17', 'name': 'iPhone 17', 'brand': 'Apple',
             'attributes': {'specs': IPHONE17_SPECS}}
    rows = [{'name': 'Bộ Tiếp Hợp Nguồn Cổng Kép 35W', 'brand': 'Apple', 'kind': 'charger', 'price': 500_000, 'stock': 5},
            {'name': 'Bộ Tiếp Hợp Nguồn Động 40W', 'brand': 'Apple', 'kind': 'charger', 'price': 999_000, 'stock': 5}]
    monkeypatch.setattr(tools, '_query', lambda sql, params=(): rows if 'compatibility' in sql else [phone])
    out = tools.find_accessories.invoke({'phone': 'iphone 17', 'accessory_type': 'củ sạc'}).splitlines()
    assert out[0].startswith('Công suất sạc của iPhone 17 (theo thông số)')
    assert '40W' in out[2] and 'đủ công suất sạc nhanh' in out[2]       # đủ công suất xếp trước, dù đắt hơn
    assert '35W' in out[3] and 'không đạt sạc nhanh' in out[3]


# ---------------------------------------------------------------- phụ kiện trùng tên, khác máy

BELKIN = [
    {'product_id': 'belkin-17', 'name': 'Miếng Dán Màn Hình Belkin UltraGlass 2 cho iPhone 17 | iPhone 16 Pro',
     'brand': 'Belkin', 'category': 'accessory', 'attributes': {'accessory_type': 'screen-protector'}},
    {'product_id': 'belkin-pro', 'name': 'Miếng Dán Màn Hình Belkin UltraGlass 2 cho iPhone 18 Pro | iPhone 17 Pro',
     'brand': 'Belkin', 'category': 'accessory', 'attributes': {'accessory_type': 'screen-protector'}},
    {'product_id': 'belkin-max', 'name': 'Miếng Dán Màn Hình Belkin UltraGlass 2 cho iPhone 18 Pro Max | iPhone 17 Pro Max',
     'brand': 'Belkin', 'category': 'accessory', 'attributes': {'accessory_type': 'screen-protector'}},
]
CATALOG = [{'product_id': 'iphone-17', 'name': 'iPhone 17', 'brand': 'Apple', 'category': 'phone', 'attributes': {}},
           {'product_id': 'iphone-18-pro-max', 'name': 'iPhone 18 Pro Max', 'brand': 'Apple', 'category': 'phone',
            'attributes': {}}, *BELKIN]
FITS = {'iphone-17': {'belkin-17'}, 'iphone-18-pro-max': {'belkin-max'}}


@pytest.fixture
def belkin_db(monkeypatch):
    def query(sql, params=()):
        if 'FROM compatibility' in sql:
            phones, accs = params
            return [{'accessory_id': a} for p in phones for a in FITS.get(p, ()) if a in accs]
        if 'FROM variants' in sql:
            return [{'variant_id': f'V-{params[0]}', 'color': None, 'storage': None, 'price': 1177000, 'stock': 5}]
        if "category = 'phone'" in sql:
            return [p for p in CATALOG if p['category'] == 'phone']
        return CATALOG
    monkeypatch.setattr(tools, '_query', query)


def _add(args, state):
    r = tools.add_to_cart.invoke({'type': 'tool_call', 'id': 't1', 'name': 'add_to_cart', 'args': {**args, 'state': state}})
    return r.update['messages'][0].content


@pytest.mark.parametrize('args, state, expected', [
    # Khách đã chốt iPhone 17: không hỏi lại bản cho 17 Pro / Pro Max (lỗi 03/11/2026)
    ({'for_phone': 'iPhone 17'}, {'cart': [], 'messages': []}, 'iPhone 17 | iPhone 16 Pro'),
    ({}, {'cart': [{'variant_id': 'X', 'name': 'iPhone 17', 'category': 'phone', 'price': 1, 'quantity': 1}],
          'messages': []}, 'iPhone 17 | iPhone 16 Pro'),
    # Gọi song song với máy trong cùng lượt: giỏ chưa có máy, lấy máy từ lệnh add_to_cart bên cạnh
    ({}, {'cart': [], 'messages': [HumanMessage('ip17 512 đen, kính belkin'), AIMessage('', tool_calls=[
        {'id': 'a', 'name': 'add_to_cart', 'args': {'product': 'iPhone 17', 'storage': '512GB', 'color': 'Đen'}},
        {'id': 'b', 'name': 'add_to_cart', 'args': {'product': 'Miếng Dán Màn Hình Belkin UltraGlass 2'}}])]},
     'iPhone 17 | iPhone 16 Pro'),
    ({'for_phone': 'iPhone 18 Pro Max'}, {'cart': [], 'messages': []}, 'iPhone 17 Pro Max'),
])
def test_add_to_cart_picks_accessory_for_customer_phone(belkin_db, args, state, expected):
    text = _add({'product': 'Miếng Dán Màn Hình Belkin UltraGlass 2', **args}, state)
    assert text.startswith('Đã thêm 1 x') and expected in text


def test_add_to_cart_still_asks_when_phone_unknown(belkin_db):
    assert _add({'product': 'Miếng Dán Màn Hình Belkin UltraGlass 2'}, {'cart': [], 'messages': []}).startswith('Có nhiều')
