import scrapy, re, unicodedata
from urllib.parse import urldefrag, urljoin
from collections import OrderedDict
from bs4 import BeautifulSoup
from readability import Document


class NHSSpider(scrapy.Spider):
    name = "nhs_spider"
    allowed_domains = ["nhs.uk"]
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.75,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "TEST_MODE": False,          # set True in settings.py for quick runs
    }

    INDEX_URL = "https://www.nhs.uk/conditions/"
    SEL = {
        "article": "main#maincontent article",
        "title":   "main#maincontent h1, h1",
        "lede":    "main#maincontent .nhsuk-lede-text",
        "hub_ul":  "ul.nhsuk-hub-key-links a::attr(href)",
    }

    # ───────────────────────── bootstrap ─────────────────────────
    @classmethod
    def from_crawler(cls, crawler, *a, **kw):
        sp = super().from_crawler(crawler, *a, **kw)
        sp.test_mode = crawler.settings.getbool("TEST_MODE")
        return sp

    # ───────────────────────── seed request ──────────────────────
    def start_requests(self):
        yield scrapy.Request(self.INDEX_URL, callback=self.parse_index)

    # ─────────────────── parse A‑Z index page ────────────────────
    def parse_index(self, response):
        links = response.css("a[href^='/conditions/']::attr(href)").getall()
        seen, sent = set(), 0
        for href in links:
            href, _ = urldefrag(href)           # drop #a / #b anchors
            if not href or href in seen or "/conditions/#" in href:
                continue
            seen.add(href)
            yield response.follow(href, self.parse_condition)
            sent += 1
            if self.test_mode and sent >= 40:
                break

    # ─────────────────── landing / overview page ─────────────────
    def parse_condition(self, response):
        # ‑‑ title (strip “Overview …” prefix) ────────────────────
        title = response.css(self.SEL["title"]).xpath("normalize-space()").get("") or ""
        if title.lower().startswith("overview"):
            title = title[len("overview"):].lstrip(" :-").strip()
        title = re.sub(r'^[\s\u2010-\u2015\-–—:]+', '', title)

        # ‑‑ main article + lede ‑─────────────────────────────────
        lede = response.css(self.SEL["lede"]).xpath("normalize-space()").get(default="")
        body = self.extract_text(response, css=self.SEL["article"])
        if lede:
            body = f"{lede}\n\n{body}"
        body = self._dedupe_lines(body)

        # base item
        item = {"url": response.url, "title": title, "overview": body, "_todo": 0}

        # ‑‑ hub sub‑pages (symptoms, causes, …) ──────────────────
        hub_links = []
        for href in response.css(self.SEL["hub_ul"]).getall():
            full = urljoin(response.url, href.split("#")[0])
            slug = full.rstrip("/").split("/")[-1]
            if slug == "help-and-support":          # skip by request
                continue
            hub_links.append((full, slug))

        if not hub_links:                           # no extra pages
            item["type"] = "condition"
            item.pop("_todo")
            yield item
            return

        # schedule sub‑page requests
        item["_todo"] = len(hub_links)
        for full, slug in hub_links:
            yield scrapy.Request(
                full,
                callback=self.parse_hub_section,
                meta={"item": item, "slug": slug},
                dont_filter=True,
            )

    # ──────────────────── hub section page ──────────────────────
    def parse_hub_section(self, response):
        item  = response.meta["item"]
        slug  = response.meta["slug"]          # e.g. symptoms / causes
        text  = self.extract_text(response, css=self.SEL["article"])
        item[slug] = self._dedupe_lines(text)

        item["_todo"] -= 1
        if item["_todo"] == 0:                 # all sections fetched
            item.pop("_todo")
            item["type"] = "condition"
            yield item

    # ───────────────────── text extractor ───────────────────────
    def extract_text(self, response, *, css=None) -> str:
        soup = BeautifulSoup(response.text, "lxml")
        frag = soup.select_one(css) if css else soup
        frag = frag or soup.body or soup

        # strip nav/cards/images before running Readability
        BAD_SEL = (
            "header, nav, aside, form, script, style,"
            " .nhsuk-related-nav, .nhsuk-back-to-top,"
            " .nhsuk-prose__support-links, .nhsuk-card--care,"
            " .nhsuk-inset-text, figure, .nhsuk-image"
        )
        for bad in frag.select(BAD_SEL):
            bad.decompose()

        # Readability candidate
        read_txt = BeautifulSoup(Document(str(frag)).summary(), "lxml")\
                   .get_text(" ", strip=True)

        # Manual candidate: every p / li (including nested li li)
        parts = []
        for node in frag.select("p, li, li li"):
            txt_node = node.get_text(" ", strip=True)
            if not txt_node:
                continue
            if node.name == "p" and node.select_one("li"):
                # header text without trailing colon
                header = txt_node.rstrip(":,;")
                sub_parts = [li.get_text(" ", strip=True) for li in node.select("li")]
                parts.append(f"{header} " + ", ".join(sub_parts) + ".")
                continue
            if node.name == "li":                       # real list item
                parts.append("• " + txt_node)
            else:                                       # normal paragraph
                parts.append(txt_node)
        man_txt = "\n".join(parts)

        # Choose manual if ≥ 50‑word longer (Readability missed big chunk)
        txt = man_txt if len(man_txt.split()) - len(read_txt.split()) >= 25 else read_txt
        txt = self._flatten_bullets(txt)

        # Generic clean‑ups
        for pat in (
            r"Page last reviewed:.*?$",
            r"Next review due:.*?$",
            r"©\s*Crown copyright.*?$",
            r"Back to (?:Conditions A to Z|top)",
        ):
            txt = re.sub(pat, " ", txt, flags=re.I | re.M | re.S)

        txt = re.sub(r'https?://\S+', ' ', txt)                         # strip raw URLs
        txt = re.sub(r'\s{2,}', ' ', txt).replace('\u201c','"').replace('\u201d','"')
        txt = self._dedupe_sentences(txt)
        return txt.strip()

    # ────────────── helpers: de‑duplication methods ──────────────
    def _dedupe_sentences(self, s: str) -> str:
        """Remove exact/near‑exact duplicate sentences while preserving order."""
        seen, out = set(), []
        for sent in re.split(r'(?<=[.!?])\s+', s):
            canon = re.sub(r'\W+', '', sent.lower())
            if canon and canon not in seen:
                seen.add(canon); out.append(sent)
        return " ".join(out)

    def _dedupe_lines(self, text: str) -> str:
        """Line‑level de‑duplication for bullet lists."""
        norm_seen, uniq = set(), []
        for line in text.splitlines():
            canon = unicodedata.normalize("NFKD", line.lower())
            canon = re.sub(r'[•\u2022\-–—]', ' ', canon)
            canon = re.sub(r'\W+', '', canon)
            if canon and canon not in norm_seen:
                norm_seen.add(canon)
                uniq.append(line)
        return "\n".join(uniq)

    # ─────────── helper: flatten bullet lists ────────────
    def _flatten_bullets(self, txt: str) -> str:
        """
        * Collapse consecutive bullet lines into one comma‑separated
          sentence that ends with a period.
        * If the line *before* that list looks like a header
          (“Symptoms include:” / “Tests you may have include,”) merge it
          and drop the duplicate.
        """
        out, buf = [], []
        lines = txt.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.lstrip().startswith("• "):      # start of bullet block
                # Gather the whole block
                while i < len(lines) and lines[i].lstrip().startswith("• "):
                    bullet = lines[i].lstrip()[2:].strip()
                    if bullet:
                        buf.append(bullet.rstrip(",.;:"))
                    i += 1

                if not buf:
                    continue

                sent = ", ".join(buf).rstrip(",;") + "."
                out.append(sent)
                out.append("")  
                buf = []
            else:
                out.append(line)
                i += 1

        return "\n".join(filter(None, out))
