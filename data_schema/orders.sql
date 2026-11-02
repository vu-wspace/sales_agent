-- Khách hàng và đơn đặt hàng từ form đặt hàng trên khung chat (api.py: POST /order, orders.py).
-- Giống hệt schema đang có trong database phone (pg_dump -s), viết lại dạng IF NOT EXISTS:
-- API chạy file này mỗi lần khởi động, database mới tạo cũng có bảng, database đã có bảng thì không đổi gì.
-- Cột thêm sau (email) có thêm ALTER ... ADD COLUMN IF NOT EXISTS để database tạo trước đó cũng được bổ sung.

CREATE TABLE IF NOT EXISTS public.customers (
    customer_id text PRIMARY KEY,
    name text,
    phone text,
    default_address text,
    preferences jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    email text
);
ALTER TABLE public.customers ADD COLUMN IF NOT EXISTS email text;
-- 1 email chỉ thuộc 1 khách (không phân biệt hoa thường); khách không có email thì để NULL
CREATE UNIQUE INDEX IF NOT EXISTS uq_customers_email ON public.customers USING btree (lower(email)) WHERE (email IS NOT NULL);

CREATE TABLE IF NOT EXISTS public.orders (
    order_id text PRIMARY KEY,
    customer_id text REFERENCES public.customers(customer_id),
    thread_id text,
    recipient_name text NOT NULL,
    recipient_phone text NOT NULL,
    address text,
    fulfillment_type text DEFAULT 'delivery' NOT NULL,
    store_id text,
    payment text DEFAULT 'cod' NOT NULL,
    total integer NOT NULL,
    status text DEFAULT 'pending' NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    recipient_email text
);
ALTER TABLE public.orders ADD COLUMN IF NOT EXISTS recipient_email text;
CREATE INDEX IF NOT EXISTS idx_orders_customer ON public.orders USING btree (customer_id);

CREATE TABLE IF NOT EXISTS public.order_items (
    order_id text NOT NULL REFERENCES public.orders(order_id),
    variant_id text NOT NULL REFERENCES public.variants(variant_id),
    qty integer NOT NULL CHECK (qty > 0),
    unit_price integer NOT NULL,
    PRIMARY KEY (order_id, variant_id)
);
