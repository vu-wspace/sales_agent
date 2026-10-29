import pytest
from langchain_core.documents import Document

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
                   '- Galaxy A57 5G (Samsung): 12.490.000đ - 16.490.280đ | 128GB, 256GB - RAM 8GB | còn hàng')


def test_search_products_no_match(fake_db):
    assert tools.search_products.invoke({'query': 'xiaomi'}).startswith('Không có sản phẩm khớp')


def test_product_details(fake_db):
    out = tools.get_product_details.invoke({'product': 'galaxy s26'})
    assert 'RAM: 256GB: 12GB' in out and 'Chip: Snapdragon' in out
    assert out.endswith('- 256GB | Đen | 29.990.001đ | còn hàng\n- 512GB | Đen | 35.990.000đ (giá tham khảo) | hết hàng')


def test_product_details_filters_storage(fake_db):
    out = tools.get_product_details.invoke({'product': 'galaxy s26', 'storage': '512 GB'})
    assert '256GB |' not in out and '512GB | Đen' in out


def test_product_details_ambiguous_asks_back(fake_db):
    out = tools.get_product_details.invoke({'product': 'iphone 18'})
    assert out.startswith("Có nhiều máy khớp 'iphone 18': iPhone 18 Pro, iPhone 18 Pro Max")


def test_find_accessories(fake_db):
    out = tools.find_accessories.invoke({'phone': 's26', 'accessory_type': 'ốp lưng'})
    assert out == 'Phụ kiện tương thích với Galaxy S26:\n- Ốp Lưng Trong Suốt Galaxy S26: 490.320đ | còn hàng'
    assert fake_db[-1][1] == ('samsung-galaxy-s26', 'case', 'case')


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
        'search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search']
