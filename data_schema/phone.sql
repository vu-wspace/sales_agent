--
-- PostgreSQL database dump
--

\restrict a4RJ7THsdv4xZaFF5tRqBQtZItz3v5EGDfKWAqOoAe0n9KUFBDzavOMcYvz30lf

-- Dumped from database version 18.6
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: compatibility; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.compatibility (
    accessory_id text NOT NULL,
    phone_id text NOT NULL
);


ALTER TABLE public.compatibility OWNER TO postgres;

--
-- Name: products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.products (
    product_id text NOT NULL,
    name text,
    brand text,
    category text,
    description text,
    attributes json
);


ALTER TABLE public.products OWNER TO postgres;

--
-- Name: variants; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.variants (
    variant_id text NOT NULL,
    product_id text,
    color text,
    storage text,
    price integer,
    stock integer
);


ALTER TABLE public.variants OWNER TO postgres;

--
-- Data for Name: compatibility; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.compatibility (accessory_id, phone_id) FROM stdin;
\.


--
-- Data for Name: products; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.products (product_id, name, brand, category, description, attributes) FROM stdin;
\.


--
-- Data for Name: variants; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.variants (variant_id, product_id, color, storage, price, stock) FROM stdin;
\.


--
-- Name: compatibility compatibility_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.compatibility
    ADD CONSTRAINT compatibility_pkey PRIMARY KEY (accessory_id, phone_id);


--
-- Name: products products_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_pkey PRIMARY KEY (product_id);


--
-- Name: variants variants_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.variants
    ADD CONSTRAINT variants_pkey PRIMARY KEY (variant_id);


--
-- Name: compatibility compatibility_accessory_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.compatibility
    ADD CONSTRAINT compatibility_accessory_id_fkey FOREIGN KEY (accessory_id) REFERENCES public.products(product_id);


--
-- Name: compatibility compatibility_phone_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.compatibility
    ADD CONSTRAINT compatibility_phone_id_fkey FOREIGN KEY (phone_id) REFERENCES public.products(product_id);


--
-- Name: variants variants_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.variants
    ADD CONSTRAINT variants_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(product_id);


--
-- PostgreSQL database dump complete
--

\unrestrict a4RJ7THsdv4xZaFF5tRqBQtZItz3v5EGDfKWAqOoAe0n9KUFBDzavOMcYvz30lf

