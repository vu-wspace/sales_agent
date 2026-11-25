-- Chính sách cửa hàng (bản gốc), be quản lý (services/be, phone_api/policies.py) và đẩy bản đã xuất bản sang agent
-- để chunk + embed vào Qdrant. be chạy file này mỗi lần khởi động; bảng rỗng thì nạp dữ liệu mẫu data_schema/policies/.
-- Quy trình: sửa bản nháp (policy_drafts) -> kiểm tra bằng câu hỏi mẫu (policy_checks) -> xuất bản (policies, lưu
-- vào policy_versions) -> đẩy sang agent. Khách chỉ thấy bản đã xuất bản.

-- Bản đang dùng (đã xuất bản)
CREATE TABLE IF NOT EXISTS public.policies (
    slug text PRIMARY KEY,                  -- vd 'doi-tra.md', cũng là source của chunk trong Qdrant
    content_md text NOT NULL,               -- markdown: # tiêu đề, ## mục, ### mục con (tách chunk theo heading)
    prompt text,                            -- hướng dẫn trả lời riêng, agent gửi kèm kết quả search_policy
    updated_at timestamp with time zone DEFAULT now()
);
ALTER TABLE public.policies ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;

-- Lịch sử: mỗi lần xuất bản (kể cả quay lại bản cũ) thêm 1 dòng, không sửa/xóa dòng cũ
CREATE TABLE IF NOT EXISTS public.policy_versions (
    slug text NOT NULL,
    version integer NOT NULL,
    content_md text NOT NULL,
    prompt text,
    published_by text,                      -- người xuất bản (admin tự ghi, chưa có tài khoản riêng)
    note text,                              -- lý do sửa
    published_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (slug, version)
);
-- Bản đang dùng của database tạo trước khi có lịch sử: ghi thành version hiện tại
INSERT INTO public.policy_versions (slug, version, content_md, prompt, note)
SELECT slug, version, content_md, prompt, 'bản có sẵn trước khi lưu lịch sử' FROM public.policies
ON CONFLICT DO NOTHING;

-- Bản nháp: tối đa 1 bản mỗi tài liệu; check_* là kết quả kiểm tra gần nhất và hash bản nháp lúc kiểm tra
CREATE TABLE IF NOT EXISTS public.policy_drafts (
    slug text PRIMARY KEY,
    content_md text NOT NULL,
    prompt text,
    updated_at timestamp with time zone DEFAULT now(),
    check_result jsonb,
    checked_hash text
);

-- Câu hỏi mẫu kiểm tra bản nháp: search_policy phải lấy ra đúng tài liệu, đoạn lấy ra phải có đủ must_contain
CREATE TABLE IF NOT EXISTS public.policy_checks (
    id serial PRIMARY KEY,
    slug text NOT NULL,
    question text NOT NULL,
    must_contain jsonb NOT NULL DEFAULT '[]'   -- [["30 ngày"], ["7", "bảy"]]: mỗi nhóm cần 1 cách viết
);
CREATE INDEX IF NOT EXISTS idx_policy_checks_slug ON public.policy_checks (slug);
