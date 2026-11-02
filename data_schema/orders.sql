-- Khách hàng và đơn đặt hàng từ form đặt hàng trên khung chat (api.py: POST /order, orders.py).
-- Giống hệt schema đang có trong database phone (pg_dump -s), viết lại dạng IF NOT EXISTS:
-- API chạy file này mỗi lần khởi động, database mới tạo cũng có bảng, database đã có bảng thì không đổi gì.

CREATE TABLE IF NOT EXISTS public.customers (
    customer_id text PRIMARY KEY,
    name text,
    phone text,
    default_address text,
    preferences jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
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
    created_at timestamp with time zone DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON public.orders USING btree (customer_id);

CREATE TABLE IF NOT EXISTS public.order_items (
    order_id text NOT NULL REFERENCES public.orders(order_id),
    variant_id text NOT NULL REFERENCES public.variants(variant_id),
    qty integer NOT NULL CHECK (qty > 0),
    unit_price integer NOT NULL,
    PRIMARY KEY (order_id, variant_id)
);
