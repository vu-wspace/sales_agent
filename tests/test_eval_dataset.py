import json
from pathlib import Path

from phone_sales.config import POLICY_DIR
from phone_sales.data_prep import load_policy_chunks
from phone_sales.tools import TOOLS

# Không import evals/run_eval.py: script đó tắt LangSmith và đổi stdout khi import
DATASET = Path(__file__).resolve().parents[1] / 'evals' / 'dataset.jsonl'
FIELDS = {'id', 'category', 'question', 'tools', 'tools_optional', 'policy', 'must_contain', 'escalate', 'refuse', 'criteria'}


def load() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_cases_have_required_fields_and_unique_ids():
    cases = load()
    assert cases
    for c in cases:
        assert set(c) <= FIELDS, c['id']  # gõ sai tên trường thì script lặng lẽ bỏ qua
        assert c['question'].strip() and c['criteria'].strip() and c['category']
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids))


def test_expected_tools_exist():
    for c in load():
        assert set(c.get('tools', [])) <= set(TOOLS), c['id']


def test_policy_files_exist():
    known = {f.name for f in POLICY_DIR.glob('*.md')}
    for c in load():
        if 'policy' in c:
            assert c['policy'] in known, c['id']


def test_questions_are_paraphrased_not_copied_from_data():
    # Câu eval trùng nguyên văn câu FAQ trong Qdrant thì truy xuất luôn đúng, chỉ số hit@k mất ý nghĩa
    stored = {c.section.removeprefix('Hỏi: ').strip().lower() for c in load_policy_chunks()}
    for c in load():
        assert c['question'].strip().lower() not in stored, c['id']


def test_must_contain_is_list_of_keyword_groups():
    for c in load():
        for group in c.get('must_contain', []):
            assert isinstance(group, list) and group and all(isinstance(k, str) and k for k in group), c['id']
