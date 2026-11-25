-- Chính sách cửa hàng (bản gốc): be (services/be, phone_api/policies.py) quản lý qua /admin/policies và đẩy sang agent
-- để chunk + embed vào Qdrant. be chạy file này mỗi lần khởi động; bảng rỗng thì nạp dữ liệu mẫu data_schema/policies/.
CREATE TABLE IF NOT EXISTS public.policies (
    slug text PRIMARY KEY,                  -- vd 'doi-tra.md', cũng là source của chunk trong Qdrant
    content_md text NOT NULL,               -- markdown: # tiêu đề, ## mục, ### mục con (tách chunk theo heading)
    prompt text,                            -- hướng dẫn trả lời riêng, agent gửi kèm kết quả search_policy
    updated_at timestamp with time zone DEFAULT now()
);
