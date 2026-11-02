-- Khách hàng, đơn đặt hàng, khuyến mãi từ form đặt hàng trên khung chat (api.py: POST /order, orders.py,
-- promotions.py) và giỏ hàng lưu theo khách (cart_item, code chưa dùng: giỏ hàng nằm trong state hội thoại).
-- Giống hệt schema đang có trong database phone (pg_dump -s), viết lại dạng IF NOT EXISTS:
-- API chạy file này mỗi lần khởi động, database mới tạo cũng có bảng, database đã có bảng thì không đổi gì.
-- Cột thêm sau (email, phone_hash, email_hash, subtotal, discount_amount, promo_id) có thêm ALTER ... ADD COLUMN IF NOT EXISTS để database tạo trước đó
-- cũng được bổ sung.
-- Số điện thoại, email (customers.phone, customers.email, orders.recipient_phone, orders.recipient_email) lưu dạng
-- mã hóa "enc:..." (pii.py, khóa PII_KEY); tra khách cũ, kiểm tra trùng email bằng phone_hash, email_hash (HMAC).

CREATE TABLE IF NOT EXISTS public.customers (
    customer_id text PRIMARY KEY,
    name text,
    phone text,
    default_address text,
    preferences jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    email text,
    phone_hash text,
    email_hash text
);
ALTER TABLE public.customers ADD COLUMN IF NOT EXISTS email text;
ALTER TABLE public.customers ADD COLUMN IF NOT EXISTS phone_hash text;
ALTER TABLE public.customers ADD COLUMN IF NOT EXISTS email_hash text;
-- Từ khi email được mã hóa, lower(email) là chuỗi mã hóa (mỗi lần mỗi khác) nên index này không còn chặn trùng;
-- giữ lại cho database cũ, việc chặn trùng chuyển sang uq_customers_email_hash
CREATE UNIQUE INDEX IF NOT EXISTS uq_customers_email ON public.customers USING btree (lower(email)) WHERE (email IS NOT NULL);
-- 1 email chỉ thuộc 1 khách (không phân biệt hoa thường); khách không có email thì để NULL
CREATE UNIQUE INDEX IF NOT EXISTS uq_customers_email_hash ON public.customers USING btree (email_hash) WHERE (email_hash IS NOT NULL);
CREATE INDEX IF NOT EXISTS idx_customers_phone_hash ON public.customers USING btree (phone_hash);

-- Khuyến mãi. scope: 'all' cả đơn, 'brand' (scope_value = 'Apple'/'Samsung'), 'product' (scope_value = product_id),
-- 'category' (scope_value = 'phone'/'accessory'). code NULL: khuyến mãi tự động, không cần nhập mã
CREATE TABLE IF NOT EXISTS public.promotions (
    promo_id text PRIMARY KEY,
    code text UNIQUE,
    description text NOT NULL,
    discount_type text NOT NULL CHECK (discount_type = ANY (ARRAY['percent'::text, 'fixed'::text])),
    value integer NOT NULL,
    max_discount integer,
    min_order integer DEFAULT 0,
    scope text DEFAULT 'all' NOT NULL,
    scope_value text,
    starts_at timestamp with time zone NOT NULL,
    ends_at timestamp with time zone NOT NULL,
    usage_limit integer,
    per_customer_limit integer DEFAULT 1,
    used_count integer DEFAULT 0 NOT NULL,
    is_active boolean DEFAULT true NOT NULL
);

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
    recipient_email text,
    subtotal integer,
    discount_amount integer DEFAULT 0 NOT NULL,
    promo_id text REFERENCES public.promotions(promo_id)
);
ALTER TABLE public.orders ADD COLUMN IF NOT EXISTS recipient_email text;
-- total = subtotal - discount_amount (đơn trước khi có khuyến mãi: subtotal NULL)
ALTER TABLE public.orders ADD COLUMN IF NOT EXISTS subtotal integer;
ALTER TABLE public.orders ADD COLUMN IF NOT EXISTS discount_amount integer DEFAULT 0 NOT NULL;
ALTER TABLE public.orders ADD COLUMN IF NOT EXISTS promo_id text REFERENCES public.promotions(promo_id);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON public.orders USING btree (customer_id);

CREATE TABLE IF NOT EXISTS public.order_items (
    order_id text NOT NULL REFERENCES public.orders(order_id),
    variant_id text NOT NULL REFERENCES public.variants(variant_id),
    qty integer NOT NULL CHECK (qty > 0),
    unit_price integer NOT NULL,
    PRIMARY KEY (order_id, variant_id)
);

CREATE TABLE IF NOT EXISTS public.promotion_usages (
    promo_id text NOT NULL REFERENCES public.promotions(promo_id),
    order_id text NOT NULL REFERENCES public.orders(order_id),
    customer_id text REFERENCES public.customers(customer_id),
    discount_amount integer NOT NULL,
    used_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (promo_id, order_id)
);
CREATE INDEX IF NOT EXISTS idx_promo_usages_customer ON public.promotion_usages USING btree (promo_id, customer_id);

CREATE TABLE IF NOT EXISTS public.cart_item (
    customer_id text NOT NULL REFERENCES public.customers(customer_id),
    variant_id text NOT NULL REFERENCES public.variants(variant_id),
    qty integer NOT NULL CHECK (qty > 0),
    added_at timestamp with time zone DEFAULT now(),
    PRIMARY KEY (customer_id, variant_id)
);
