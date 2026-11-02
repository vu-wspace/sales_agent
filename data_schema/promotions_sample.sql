-- Khuyến mãi mẫu để thử form đặt hàng và tool get_promotions (dữ liệu demo, không phải chương trình thật).
-- Nạp: docker exec -i phone_db psql -U postgres -d phone < data_schema/promotions_sample.sql
-- Chạy lại được: ghi đè theo promo_id, giữ used_count. Xóa: DELETE FROM promotions WHERE promo_id LIKE 'DEMO-%'
-- (chỉ xóa được khi chưa có đơn nào dùng mã).
INSERT INTO public.promotions (promo_id, code, description, discount_type, value, max_discount, min_order, scope,
                               scope_value, starts_at, ends_at, usage_limit, per_customer_limit)
VALUES
    -- Giảm cố định cả đơn, đơn từ 10 triệu, mỗi khách 1 lần, tối đa 100 lượt
    ('DEMO-GIAM500K', 'GIAM500K', 'Ưu đãi cuối năm', 'fixed', 500000, NULL, 10000000, 'all', NULL,
     '2026-11-01 00:00+07', '2026-12-31 23:59+07', 100, 1),
    -- Giảm 5% cho máy, phụ kiện Samsung, tối đa 2 triệu
    ('DEMO-SAMSUNG5', 'SAMSUNG5', 'Tuần lễ Samsung', 'percent', 5, 2000000, 0, 'brand', 'Samsung',
     '2026-11-01 00:00+07', '2026-11-30 23:59+07', NULL, 1),
    -- Tự động (không cần mã): giảm 10% phụ kiện, tối đa 300.000đ
    ('DEMO-PHUKIEN10', NULL, 'Mua kèm phụ kiện', 'percent', 10, 300000, 0, 'category', 'accessory',
     '2026-11-01 00:00+07', '2026-12-31 23:59+07', NULL, NULL),
    -- Đã hết hạn: để thử thông báo "mã đã hết hạn"
    ('DEMO-HETHAN', 'HETHAN', 'Ưu đãi tháng 10', 'fixed', 1000000, NULL, 0, 'all', NULL,
     '2026-10-01 00:00+07', '2026-10-31 23:59+07', NULL, 1)
ON CONFLICT (promo_id) DO UPDATE SET
    code = EXCLUDED.code, description = EXCLUDED.description, discount_type = EXCLUDED.discount_type,
    value = EXCLUDED.value, max_discount = EXCLUDED.max_discount, min_order = EXCLUDED.min_order,
    scope = EXCLUDED.scope, scope_value = EXCLUDED.scope_value, starts_at = EXCLUDED.starts_at,
    ends_at = EXCLUDED.ends_at, usage_limit = EXCLUDED.usage_limit, per_customer_limit = EXCLUDED.per_customer_limit;
