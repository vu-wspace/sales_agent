"""Cào dữ liệu điện thoại + phụ kiện Apple, Samsung (thị trường VN) cho database `phone`.

Nguồn: apple.com/vn và samsung.com/vn (chỉ các trang HTML công khai, dữ liệu nhúng sẵn
trong trang: PRODUCT_SELECTION_BOOTSTRAP của Apple, JSON-LD + buyingOptions của Samsung).

Đầu ra (thư mục output/):
  - products.json, variants.json, compatibility.json
  - phone_seed.sql  -> nạp vào Postgres:  psql -U postgres -d phone -f output/phone_seed.sql

Ghi chú dữ liệu (mục đích học tập / demo):
  - stock: website hãng không công khai tồn kho -> sinh số giả lập ổn định theo SKU
    (hết hàng trên web hãng -> stock = 0).
  - Samsung chỉ nhúng giá cho một số SKU; SKU thiếu giá được ước tính từ SKU cùng dung lượng
    hoặc từ giá bản thấp nhất + chênh lệch theo dung lượng (đánh dấu trong attributes).

Chạy:  python scrape_phone.py            (dùng cache .cache/ nếu có)
       python scrape_phone.py --refresh  (tải lại toàn bộ)
"""

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
OUT_DIR = BASE_DIR / "output"

UA = "Mozilla/5.0 (compatible; phone-demo-scraper/1.0; educational use)"
REQUEST_DELAY = 0.8  # giây giữa 2 request thật (không tính cache)

APPLE = "https://www.apple.com"
SAMSUNG = "https://www.samsung.com"
APPLE_ACCESSORY_CATEGORIES = ["cases-protection", "chargers-adapters", "magsafe"]
SAMSUNG_LISTING = f"{SAMSUNG}/vn/smartphones/all-smartphones/"
SAMSUNG_SERIES = ["galaxy-s", "galaxy-z", "galaxy-a"]
SAMSUNG_ACCESSORY_LISTING = f"{SAMSUNG}/vn/mobile-accessories/all-mobile-accessories/"

# Chênh lệch giá ước tính so với bản dung lượng thấp nhất (chỉ dùng khi web không có giá)
STORAGE_STEP_VND = {"128GB": 0, "256GB": 3_000_000, "512GB": 7_000_000, "1TB": 13_000_000, "2TB": 25_000_000}

_refresh = False
_last_request = 0.0


# ---------------------------------------------------------------- tiện ích

def fetch(url: str) -> str | None:
    """GET có cache trên đĩa; trả None nếu 404/lỗi."""
    global _last_request
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".html")
    if cache.exists() and not _refresh:
        text = cache.read_text(encoding="utf-8")
        return text or None
    wait = REQUEST_DELAY - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "vi-VN,vi;q=0.9"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            cache.write_text(text, encoding="utf-8")
            return text
        except urllib.error.HTTPError as e:
            log(f"  ! HTTP {e.code} {url}")
            if e.code == 404:
                cache.write_text("", encoding="utf-8")  # nhớ 404 để lần sau khỏi gọi lại
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"  ! {e} {url} (thử lại {attempt + 1})")
            time.sleep(3 * (attempt + 1))
    return None  # lỗi mạng: không ghi cache, lần chạy sau sẽ thử lại


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def clean(s: str | None) -> str:
    """Bỏ thẻ HTML, footnote, gộp khoảng trắng."""
    if not s:
        return ""
    s = re.sub(r"<as-footnote.*?</as-footnote>", "", s, flags=re.S)
    s = re.sub(r"<sup.*?</sup>", "", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", s).strip()


def json_after(text: str, marker: str, start: int = 0):
    """Decode giá trị JSON đứng ngay sau `marker`."""
    i = text.find(marker, start)
    if i < 0:
        return None
    return json.JSONDecoder().raw_decode(text[i + len(marker):].lstrip())[0]


def meta_description(page: str) -> str:
    m = re.search(r'<meta (?:name|property)="(?:og:)?description" content="([^"]*)"', page)
    return clean(m.group(1)) if m else ""


def jsonld_blocks(page: str):
    for m in re.finditer(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', page, re.S):
        try:
            yield json.loads(m.group(1))
        except json.JSONDecodeError:
            continue


def norm_storage(s: str) -> str:
    m = re.search(r"(\d+)\s*(GB|TB)", s, re.I)
    return f"{m.group(1)}{m.group(2).upper()}" if m else s.strip()


def demo_stock(sku: str, in_stock: bool = True) -> int:
    if not in_stock:
        return 0
    return int(hashlib.sha1(sku.encode()).hexdigest(), 16) % 46 + 5  # 5..50


def slugify(name: str) -> str:
    s = name.lower().replace("+", " plus")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------- Apple

def apple_specs(slug: str) -> dict[str, dict[str, str]]:
    """Trang /vn/<slug>/specs/ -> {tên model: {mục: nội dung}}; key '*' = áp dụng mọi model."""
    page = fetch(f"{APPLE}/vn/{slug}/specs/")
    if not page:
        return {}
    specs: dict[str, dict[str, str]] = {}
    # Thứ tự thuộc tính không cố định (role trước/sau class) -> tách theo class
    for sec in re.split(r'<div[^>]*class="techspecs-section [^"]*"[^>]*>', page)[1:]:
        hm = re.search(r'role="rowheader"[^>]*>(.*?)</div>', sec, re.S)
        if not hm:
            continue
        title = clean(hm.group(1))
        cells = re.split(r'<div[^>]*class="techspecs-column[^"]*"[^>]*>', sec)[1:]
        for cell in cells:
            heading = re.search(r'techspecs-small-heading[^"]*"[^>]*>(.*?)</strong>', cell, re.S)
            model = clean(heading.group(1)) if heading else "*"
            body = re.sub(r'<strong class="techspecs-small-heading.*?</strong>', "", cell, flags=re.S)
            body = re.sub(r'<div class="techspecs-appgrid.*', "", body, flags=re.S)
            text = clean(body)
            if not text:
                continue
            cur = specs.setdefault(model, {})
            cur[title] = (cur.get(title, "") + " " + text).strip()[:600]
    return specs


def scrape_apple_phones():
    products, variants = [], []
    listing = fetch(f"{APPLE}/vn/shop/buy-iphone")
    slugs = sorted(set(re.findall(r'/vn/shop/buy-iphone/(iphone-[a-z0-9-]+)', listing or "")))
    for slug in slugs:
        log(f"[apple] {slug}")
        page = fetch(f"{APPLE}/vn/shop/buy-iphone/{slug}")
        if not page or "productSelectionData:" not in page:
            log("  ! không có productSelectionData, bỏ qua")
            continue
        data = json_after(page, "productSelectionData:")
        dv = data["displayValues"]
        prices = dv.get("prices", {})
        title = clean(re.search(r"<title>(.*?)</title>", page, re.S).group(1))
        default_name = re.sub(r"^Mua\s+|\s+-\s+Apple.*$", "", title).strip()
        colors = {k: clean(v["value"]) for k, v in (dv.get("dimensionColor") or {}).items()
                  if isinstance(v, dict) and "value" in v}
        sizes = {}
        for k, v in (dv.get("dimensionScreensize") or {}).items():
            if isinstance(v, dict) and "value" in v:
                raw = v["value"].split("\n")
                sizes[k] = (clean(raw[0]), clean(raw[1]) if len(raw) > 1 else "")

        overview = fetch(f"{APPLE}/vn/{slug}/")
        description = meta_description(overview or "") or meta_description(page)
        specs = apple_specs(slug)
        groups: dict[str, dict] = {}
        for p in data["products"]:
            size_key = p.get("dimensionScreensize")
            name, size_label = sizes.get(size_key, (default_name, ""))
            pid = "apple-" + slugify(name)
            if pid not in groups:
                model_specs = dict(specs.get("*", {}))
                model_specs.update(specs.get(name, {}))
                attrs = {"source_url": f"{APPLE}/vn/shop/buy-iphone/{slug}", "family": p.get("familyType")}
                if size_label:
                    attrs["screen"] = size_label
                attrs["specs"] = model_specs
                attrs["magsafe"] = "MagSafe" in json.dumps(model_specs, ensure_ascii=False)
                attrs["port"] = "USB-C"
                groups[pid] = {"product_id": pid, "name": name, "brand": "Apple", "category": "phone",
                               "description": description, "attributes": attrs}
            price_info = prices.get(p.get("fullPrice") or "", {})
            price = price_info.get("amountBeforeTradeIn")
            if price is None:
                continue
            sku = p["partNumber"]
            variants.append({
                "variant_id": sku, "product_id": pid,
                "color": colors.get(p.get("dimensionColor"), p.get("dimensionColor")),
                "storage": norm_storage(p.get("dimensionCapacity", "")),
                "price": int(price), "stock": demo_stock(sku),
            })
        products.extend(groups.values())
    return products, variants


def scrape_apple_accessories():
    items = {}
    for cat in APPLE_ACCESSORY_CATEGORIES:
        log(f"[apple-acc] {cat}")
        page = fetch(f"{APPLE}/vn/shop/iphone/accessories/{cat}")
        tiles = json_after(page or "", '"tiles":')
        for t in (tiles or {}).get("items", []):
            v = t["value"]
            price = (v.get("productPrice") or {}).get("priceCurrent")
            if not price:
                continue
            items[v["partNumber"]] = {
                "sku": v["partNumber"], "name": clean(v["title"]),
                "price": int(re.sub(r"\D", "", price)),
                "url": APPLE + v["link"]["url"].split("?")[0], "category": cat,
            }
    return list(items.values())


# ---------------------------------------------------------------- Samsung

def samsung_listing_prices() -> dict[str, dict]:
    """JSON-LD ItemList ở trang danh sách: {modelCode: {name, price, url}}."""
    out = {}
    for url in [SAMSUNG_LISTING] + [f"{SAMSUNG}/vn/smartphones/{s}/" for s in SAMSUNG_SERIES]:
        page = fetch(url) or ""
        for block in jsonld_blocks(page):
            if not (isinstance(block, dict) and block.get("@type") == "ItemList"):
                continue
            for el in block.get("itemListElement", []):
                item = el.get("item", {})
                u = item.get("url", "")
                m = re.search(r"modelCode=([A-Z0-9-]+)", u) or re.search(r"-(sm-[a-z0-9]+)/?$", u)
                if not m:
                    continue
                out[m.group(1).upper()] = {"name": item.get("name", ""), "url": u,
                                           "price": int(float(item.get("offers", {}).get("price") or 0))}
    return out


def clean_samsung_name(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", " ", clean(name)).strip()


def fill_missing_prices(rows: list[dict], known: dict[str, int]) -> list[str]:
    """Điền giá cho SKU thiếu; trả về danh sách SKU được ước tính."""
    estimated = []
    by_storage = {}
    for r in rows:
        if r["price"] is None and r["variant_id"] in known:
            r["price"] = known[r["variant_id"]]
        if r["price"]:
            by_storage.setdefault(r["storage"], r["price"])
    for r in rows:
        if r["price"]:
            continue
        if r["storage"] in by_storage:
            r["price"] = by_storage[r["storage"]]
        elif by_storage:
            base_storage, base_price = min(by_storage.items(), key=lambda kv: kv[1])
            step = lambda s: STORAGE_STEP_VND.get(s.split(" ")[0], 0)  # "256GB - RAM 12GB" -> "256GB"
            delta = step(r["storage"]) - step(base_storage)
            r["price"] = base_price + max(delta, 0)
        else:
            continue
        estimated.append(r["variant_id"])
    return estimated


def scrape_samsung_buy_page(path: str, listing: dict[str, dict]):
    """Trang /buy/ của flagship (S, Z): có thể chứa nhiều model (vd. S26 và S26+)."""
    family_slug = path.strip("/").split("/")[-2]
    page = fetch(SAMSUNG + path)
    if not page:
        return [], []
    p = page.find('"products":{"SM-')
    b = page.rfind('"buyingOptions":[', 0, p)
    if p < 0 or b < 0:
        log(f"  ! không đọc được buyingOptions {path}")
        return [], []
    dec = json.JSONDecoder()
    skus = dec.raw_decode(page[p + len('"products":'):])[0]
    options = dec.raw_decode(page[b + len('"buyingOptions":'):])[0]

    opt_kind, opt_items = {}, {}
    for o in options:
        label = o.get("displayName", "").lower()
        kind = ("device" if "thiết bị" in label else
                "storage" if ("bộ nhớ" in label or "dung lượng" in label) else
                "color" if "màu" in label else None)
        if kind:
            opt_kind[o["typeCode"]] = kind
            opt_items[o["typeCode"]] = {i["itemId"]: clean(i["displayName"]) for i in o["optionItems"]}

    description = meta_description(page)
    known_prices = {k: v["price"] for k, v in listing.items() if v["price"]}
    groups: dict[str, dict] = {}
    rows_by_pid: dict[str, list] = {}
    for code, sku in skus.items():
        if not code.startswith("SM-") or sku.get("displayYn") == "N":
            continue  # bỏ bản bundle / special edition (mã F-...)
        sel = {}
        for m in sku["optionMappings"]:
            kind = opt_kind.get(m["typeCode"])
            if kind:
                sel[kind] = opt_items[m["typeCode"]].get(m["itemId"], "")
        name = clean_samsung_name(sel.get("device") or family_slug.replace("-", " ").title())
        if "special edition" in name.lower():
            continue
        pid = "samsung-" + slugify(name)
        storage_label = sel.get("storage", "")
        storage = norm_storage(storage_label.split("｜")[0])
        ram = norm_storage(storage_label.split("｜")[1]) if "｜" in storage_label else None
        if pid not in groups:
            groups[pid] = {"product_id": pid, "name": name, "brand": "Samsung", "category": "phone",
                           "description": description,
                           "attributes": {"source_url": SAMSUNG + path, "model_name": sku.get("modelName"),
                                          "ram_by_storage": {}, "port": "USB-C"}}
        if ram:
            groups[pid]["attributes"]["ram_by_storage"][storage] = ram
        price = int(sku["price"]) if sku.get("price") else None
        rows_by_pid.setdefault(pid, []).append({
            "variant_id": code, "product_id": pid, "color": sel.get("color"), "storage": storage,
            "price": price, "stock": demo_stock(code, sku.get("stockLevelStatus") != "outOfStock"),
        })
    variants = []
    for pid, rows in rows_by_pid.items():
        est = fill_missing_prices(rows, known_prices)
        if est:
            groups[pid]["attributes"]["estimated_price_skus"] = est
        variants.extend(r for r in rows if r["price"])
    return list(groups.values()), variants


PDP_SPEC_KEYS = [
    "Bộ vi xử lý", "Loại CPU", "Kích cỡ (Màn hình chính)", "Độ phân giải (Màn hình chính)",
    "Công nghệ màn hình (màn hình chính)", "Tần số làm tươi màn hình (Màn hình chính)",
    "Camera sau - Độ phân giải (Multiple)", "Camera trước - Độ phân giải", "Độ Phân Giải Quay Video",
    "Dung lượng pin (mAh, Typical)", "Kích thước (HxWxD, mm)", "Trọng lượng (g)", "Số lượng SIM",
    "Loại khe Sim", "Giao diện USB", "Phiên bản Bluetooth", "NFC", "Wi-Fi", "Hỗ trợ Samsung DeX",
]


def scrape_samsung_pdp(url: str, listing: dict[str, dict]):
    """Trang chi tiết (dòng A, FE): 1 trang = 1 model, biến thể nằm trong BV-buyingOptionData."""
    page = fetch(url)
    if not page:
        return [], []
    product_ld = next((b for b in jsonld_blocks(page) if isinstance(b, dict) and b.get("@type") == "Product"), {})
    name = clean_samsung_name(product_ld.get("name", ""))
    m = re.search(r'id="BV-buyingOptionData" value="([^"]*)"', page)
    if not m:
        return [], []
    options = json.loads(html.unescape(m.group(1)))
    # Tên trong JSON-LD kèm màu ("Galaxy A57 5G Xanh Đêm") -> lấy theo tên dòng ở listing
    url_sku = re.search(r"-(sm-[a-z0-9]+)/?$", url)
    current_sku = (product_ld.get("sku") or (url_sku.group(1) if url_sku else "")).upper()
    listing_name = next((clean_samsung_name(v["name"]) for k, v in listing.items() if k == current_sku), None)
    name = listing_name or re.sub(r"\s+(Xanh|Đen|Trắng|Tím|Hồng|Bạc|Vàng|Xám).*$", "", name)
    if not name:
        log(f"  ! không xác định được tên model {url}")
        return [], []
    pid = "samsung-" + slugify(name)

    specs_raw = re.findall(r'pdd32-product-spec__content-item-title">(.*?)</p>\s*'
                           r'<p class="pdd32-product-spec__content-item-desc">(.*?)</p>', page, re.S)
    specs = {}
    for k, v in specs_raw:
        k = clean(k)
        if k in PDP_SPEC_KEYS and k not in specs:
            specs[k] = clean(v)
    attrs = {"source_url": url, "port": "USB-C", "ram_by_storage": {}, "specs": specs}

    known = {k: v["price"] for k, v in listing.items() if v["price"]}
    offer = product_ld.get("offers") or {}
    if current_sku and offer.get("price"):
        known[current_sku] = int(float(offer["price"]))
    # Cùng dung lượng nhưng nhiều mức RAM (vd A57 256GB có 8GB và 12GB) -> ghi RAM vào storage
    # để mỗi phiên bản là 1 dòng phân biệt được trong bảng variants
    rams_per_storage: dict[str, set] = {}
    for opt in options.values():
        rams_per_storage.setdefault(norm_storage(opt.get("Lưu trữ", "")), set()).add(opt.get("Bộ nhớ"))
    rows = []
    for code, opt in options.items():
        storage = norm_storage(opt.get("Lưu trữ", ""))
        ram = norm_storage(opt.get("Bộ nhớ", "")) if opt.get("Bộ nhớ") else None
        if ram and len(rams_per_storage[storage]) > 1:
            storage = f"{storage} - RAM {ram}"
        if ram:
            attrs["ram_by_storage"].setdefault(storage, ram)
        in_stock = not (code == current_sku and "OutOfStock" in str(offer.get("availability", "")))
        rows.append({"variant_id": code, "product_id": pid, "color": opt.get("Màu Sắc"),
                     "storage": storage, "price": known.get(code), "stock": demo_stock(code, in_stock)})
    est = fill_missing_prices(rows, known)
    if est:
        attrs["estimated_price_skus"] = est
    product = {"product_id": pid, "name": name, "brand": "Samsung", "category": "phone",
               "description": meta_description(page), "attributes": attrs}
    return [product], [r for r in rows if r["price"]]


def scrape_samsung_phones():
    listing = samsung_listing_prices()
    pages = set()
    for url in [SAMSUNG_LISTING] + [f"{SAMSUNG}/vn/smartphones/{s}/" for s in SAMSUNG_SERIES]:
        page = fetch(url) or ""
        pages.update(re.findall(r'/vn/smartphones/galaxy-[a-z0-9-]+/buy/', page))
        pages.update(re.findall(r'/vn/smartphones/galaxy-[asz]/galaxy-[a-z0-9-]+-sm-[a-z0-9]+/', page))
    products, variants, seen = [], [], set()
    for path in sorted(pages):
        log(f"[samsung] {path}")
        if path.endswith("/buy/"):
            prods, vars_ = scrape_samsung_buy_page(path, listing)
        else:
            prods, vars_ = scrape_samsung_pdp(SAMSUNG + path, listing)
        for p in prods:
            if p["product_id"] in seen:  # PDP khác màu của cùng model
                continue
            if not any(v["product_id"] == p["product_id"] for v in vars_):
                log(f"  - bỏ {p['name']}: không có biến thể nào có giá")
                continue
            seen.add(p["product_id"])
            products.append(p)
            variants.extend(v for v in vars_ if v["product_id"] == p["product_id"])
    return products, variants


def scrape_samsung_accessories():
    page = fetch(SAMSUNG_ACCESSORY_LISTING) or ""
    items = []
    for block in jsonld_blocks(page):
        if not (isinstance(block, dict) and block.get("@type") == "ItemList"):
            continue
        for el in block.get("itemListElement", []):
            it = el.get("item", {})
            url = it.get("url", "")
            slug = url.rstrip("/").split("/")[-1]
            m = re.search(r"-((?:ef|gp|ep|eb|ej|ei|et)-[a-z0-9]+)$", slug)
            price = int(float((it.get("offers") or {}).get("price") or 0))
            if not m or not price:
                continue
            items.append({"sku": m.group(1).upper(), "name": clean(it.get("name")), "price": price,
                          "url": url, "slug": slug,
                          "in_stock": "InStock" in str((it.get("offers") or {}).get("availability"))})
    return items


# ---------------------------------------------------------------- phụ kiện & tương thích

def accessory_type(name: str) -> str:
    n = name.lower()
    rules = [("s pen", "s-pen"), ("dán", "screen-protector"), ("film", "screen-protector"),
             ("ốp", "case"), ("bao da", "case"), ("folio", "case"), ("case", "case"),
             ("pin magsafe", "power-bank"), ("dự phòng", "power-bank"), ("battery", "power-bank"),
             ("pin", "power-bank"), ("cáp", "cable"), ("sạc", "charger"), ("tiếp hợp", "charger"),
             ("adapter", "charger"), ("charger", "charger"), ("ví", "wallet"), ("wallet", "wallet"),
             ("dây đeo", "strap"), ("giá đỡ", "holder")]
    return next((t for k, t in rules if k in n), "other")


IPHONE_RE = re.compile(r"iPhone\s+(Duo|Air|\d+e?(?:\s+Pro\s+Max|\s+Pro|\s+Plus)?)(?![\w])")


def build_accessories(apple_acc, samsung_acc, phones):
    products, variants, compat = [], [], []
    iphones = {p["name"].lower(): p for p in phones if p["brand"] == "Apple"}
    galaxy_by_slug = {p["product_id"].removeprefix("samsung-"): p for p in phones if p["brand"] == "Samsung"}
    all_phone_ids = [p["product_id"] for p in phones]
    magsafe_ids = [p["product_id"] for p in iphones.values() if p["attributes"].get("magsafe")]

    def add(pid, name, brand, sku, price, url, compatible, in_stock=True, color=None):
        if not compatible:
            return
        products.append({"product_id": pid, "name": name, "brand": brand, "category": "accessory",
                         "description": name,
                         "attributes": {"accessory_type": accessory_type(name), "sku": sku, "source_url": url}})
        variants.append({"variant_id": sku, "product_id": pid, "color": color, "storage": None,
                         "price": price, "stock": demo_stock(sku, in_stock)})
        compat.extend({"accessory_id": pid, "phone_id": ph} for ph in sorted(set(compatible)))

    for a in apple_acc:
        name = a["name"]
        if "lightning" in name.lower():
            continue  # iPhone đang bán đều dùng USB-C
        mentioned = ["iphone " + m.lower() for m in IPHONE_RE.findall(name)]
        if mentioned:
            compatible = [iphones[m]["product_id"] for m in mentioned if m in iphones]
        elif re.search(r"magsafe|qi2|maggo", name, re.I):
            compatible = magsafe_ids
        elif re.search(r"usb-c|usb‑c|sạc|tiếp hợp|thunderbolt", name, re.I):
            compatible = all_phone_ids
        else:
            compatible = []
        color = name.rsplit(" - ", 1)[1] if " - " in name else None
        add("acc-apple-" + a["sku"].split("/")[0].lower(), name, "Apple", a["sku"], a["price"], a["url"],
            compatible, color=color)

    slugs_longest_first = sorted(galaxy_by_slug, key=len, reverse=True)
    wireless_galaxy = [p["product_id"] for s, p in galaxy_by_slug.items() if re.match(r"galaxy-[sz]", s)]
    for a in samsung_acc:
        slug = a["slug"]
        if re.search(r"tab-|watch|buds|smarttag|keyboard|book-cover", slug):
            continue
        target = next((s for s in slugs_longest_first if slug.startswith(s + "-")), None)
        if target:
            compatible = [galaxy_by_slug[target]["product_id"]]
        elif slug.startswith("galaxy-"):
            compatible = []  # phụ kiện cho model không còn bán trong DB
        elif slug.startswith("magnet-") or "holder" in slug or "wireless" in slug:
            compatible = wireless_galaxy  # sạc không dây / nam châm: dòng A không hỗ trợ
        elif re.search(r"power-adapter|battery-pack|charger", slug):
            compatible = all_phone_ids
        else:
            compatible = []
        add("acc-samsung-" + a["sku"].lower(), a["name"], "Samsung", a["sku"], a["price"], a["url"],
            compatible, a["in_stock"])
    return products, variants, compat


# ---------------------------------------------------------------- xuất SQL

def sql_lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(int(v))
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    return "'" + str(v).replace("'", "''") + "'"


def write_sql(products, variants, compat, path: Path):
    lines = [
        "-- Seed dữ liệu cho database phone (sinh tự động bởi scrape_phone.py)",
        "-- Nạp: psql -U postgres -d phone -f output/phone_seed.sql",
        "SET client_encoding = 'UTF8';",
        "BEGIN;",
        "TRUNCATE public.compatibility, public.variants, public.products;",
    ]
    for p in products:
        vals = ", ".join(sql_lit(p[k]) for k in ("product_id", "name", "brand", "category", "description", "attributes"))
        lines.append(f"INSERT INTO public.products (product_id, name, brand, category, description, attributes) VALUES ({vals});")
    for v in variants:
        vals = ", ".join(sql_lit(v[k]) for k in ("variant_id", "product_id", "color", "storage", "price", "stock"))
        lines.append(f"INSERT INTO public.variants (variant_id, product_id, color, storage, price, stock) VALUES ({vals});")
    for c in compat:
        lines.append(f"INSERT INTO public.compatibility (accessory_id, phone_id) VALUES ({sql_lit(c['accessory_id'])}, {sql_lit(c['phone_id'])});")
    lines.append("COMMIT;")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global _refresh
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="bỏ qua cache, tải lại trang")
    _refresh = ap.parse_args().refresh

    a_prod, a_var = scrape_apple_phones()
    s_prod, s_var = scrape_samsung_phones()
    phones = a_prod + s_prod
    acc_prod, acc_var, compat = build_accessories(scrape_apple_accessories(), scrape_samsung_accessories(), phones)

    products = phones + acc_prod
    variants = a_var + s_var + acc_var
    # khử trùng lặp khóa chính
    products = list({p["product_id"]: p for p in products}.values())
    variants = list({v["variant_id"]: v for v in variants}.values())
    compat = list({(c["accessory_id"], c["phone_id"]): c for c in compat}.values())

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in [("products", products), ("variants", variants), ("compatibility", compat)]:
        (OUT_DIR / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sql(products, variants, compat, OUT_DIR / "phone_seed.sql")
    log(f"Xong: {len(phones)} điện thoại, {len(acc_prod)} phụ kiện, {len(variants)} biến thể, "
        f"{len(compat)} dòng tương thích -> {OUT_DIR}")


if __name__ == "__main__":
    main()
