# Tách các file chính sách (.md) thành chunk nhỏ theo heading để nạp vào vector DB
import re
from dataclasses import dataclass
from pathlib import Path
from bank_faqs.config import CHUNK_MAX_CHARS, POLICY_DIR

@dataclass
class Chunk:
    source: str    # tên file, vd 'doi-tra.md'
    title: str     # tiêu đề tài liệu (heading #)
    section: str   # 'Mục 3 > Mục 3.1', hoặc 'Hỏi: ...' nếu là FAQ
    text: str      # nội dung chunk (không gồm tiêu đề)

    @property
    def page_content(self) -> str:
        # Gắn tiêu đề + mục vào nội dung để embedding hiểu ngữ cảnh (model bkai chỉ đọc ~256 token đầu)
        return f'{self.title} - {self.section}\n{self.text}'

def _split_long(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    # Gộp các đoạn (cách nhau dòng trống) tới max_chars; bảng quá dài thì cắt theo dòng, lặp lại header bảng
    blocks = []
    for block in re.split(r'\n\s*\n', text.strip()):
        lines = block.splitlines()
        if len(block) > max_chars and len(lines) > 3 and lines[0].startswith('|'):
            header, cur = lines[:2], []
            for row in lines[2:]:
                if cur and len('\n'.join(header + cur + [row])) > max_chars:
                    blocks.append('\n'.join(header + cur))
                    cur = []
                cur.append(row)
            blocks.append('\n'.join(header + cur))
        else:
            blocks.append(block)

    parts, cur = [], ''
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > max_chars:
            parts.append(cur)
            cur = b
        else:
            cur = f'{cur}\n\n{b}' if cur else b
    if cur:
        parts.append(cur)
    return parts

def _faq_items(text: str) -> list[tuple[str, str]]:
    # FAQ viết dạng '**Câu hỏi?**' rồi câu trả lời ở dòng sau -> mỗi cặp hỏi/đáp là 1 chunk
    items = re.findall(r'^\*\*(.+?)\*\*[ \t]*\n(.*?)(?=^\*\*|\Z)', text, flags=re.M | re.S)
    return [(q.strip(), a.strip()) for q, a in items if a.strip()]

def split_markdown(md: str, source: str) -> list[Chunk]:
    title_m = re.search(r'^# (.+)$', md, flags=re.M)
    title = title_m.group(1).strip() if title_m else source
    # Tách theo heading ## và ###; phần trước heading ## đầu tiên là phần giới thiệu
    parts = re.split(r'^(#{2,3}) (.+)$', md, flags=re.M)
    intro = parts[0].replace(title_m.group(0), '') if title_m else parts[0]
    sections, h2 = [('Giới thiệu', intro)], ''
    for level, heading, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        heading = heading.strip()
        if level == '##':
            h2 = heading
            sections.append((heading, body))
        else:
            sections.append((f'{h2} > {heading}', body))

    chunks = []
    for section, body in sections:
        if not body.strip():
            continue
        faq = _faq_items(body)
        if faq and 'thường gặp' in section.lower():
            chunks += [Chunk(source, title, f'Hỏi: {q}', f'Đáp: {a}') for q, a in faq]
        else:
            chunks += [Chunk(source, title, section, t) for t in _split_long(body)]
    return chunks

def load_policy_chunks(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    files = sorted(policy_dir.glob('*.md'))
    if not files:
        raise FileNotFoundError(f'Không có file chính sách .md trong {policy_dir}')
    return [c for f in files for c in split_markdown(f.read_text(encoding='utf-8'), f.name)]

def main() -> None:
    chunks = load_policy_chunks()
    by_file: dict[str, int] = {}
    for c in chunks:
        by_file[c.source] = by_file.get(c.source, 0) + 1
    print(f'Đã tách {len(chunks)} chunk từ {len(by_file)} file trong {POLICY_DIR}')
    for name, n in by_file.items():
        print(f'  {name}: {n} chunk')
    longest = max(chunks, key=lambda c: len(c.page_content))
    print(f'\nChunk dài nhất ({len(longest.page_content)} ký tự):\n{longest.page_content}')

if __name__ == '__main__':
    main()
