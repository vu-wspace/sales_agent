from bank_faqs.config import CHUNK_MAX_CHARS
from bank_faqs.data_prep import load_policy_chunks, split_markdown

MD = """# Chính sách A

> Giới thiệu chung.

## 1. Phạm vi

Áp dụng cho mọi đơn.

### 1.1. Chi tiết

Nội dung chi tiết.

## 2. Câu hỏi thường gặp

**Hỏi một?**
Đáp một.

**Hỏi hai?**
Đáp hai,
dòng thứ hai.
"""


def test_split_markdown_sections_and_faq():
    chunks = split_markdown(MD, 'a.md')
    assert [c.section for c in chunks] == [
        'Giới thiệu', '1. Phạm vi', '1. Phạm vi > 1.1. Chi tiết', 'Hỏi: Hỏi một?', 'Hỏi: Hỏi hai?',
    ]
    assert all(c.source == 'a.md' and c.title == 'Chính sách A' for c in chunks)
    assert chunks[0].text == '> Giới thiệu chung.'
    assert chunks[4].text == 'Đáp: Đáp hai,\ndòng thứ hai.'
    assert chunks[2].page_content == 'Chính sách A - 1. Phạm vi > 1.1. Chi tiết\nNội dung chi tiết.'


def test_long_table_is_split_with_header_repeated():
    rows = '\n'.join(f'| dòng {i} | {"x" * 80} |' for i in range(30))
    chunks = split_markdown(f'# T\n\n## Bảng\n\n| Cột | Giá trị |\n|---|---|\n{rows}\n', 't.md')
    assert len(chunks) > 1
    assert all(c.text.startswith('| Cột | Giá trị |\n|---|---|') for c in chunks)
    assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)
    assert sum(c.text.count('| dòng ') for c in chunks) == 30


def test_policy_files_load():
    chunks = load_policy_chunks()
    sources = {c.source for c in chunks}
    assert sources == {'doi-tra.md', 'giao-hang.md', 'tra-gop.md', 'bao-hanh-apple.md', 'bao-hanh-samsung.md'}
    assert all(c.text.strip() for c in chunks)
    assert all(len(c.text) <= CHUNK_MAX_CHARS for c in chunks)
