# spiders/webmd_spider.py
import re, html, unicodedata, scrapy, datetime as dt
from urllib.parse import urljoin, urldefrag, urlparse
from collections import defaultdict
from bs4 import BeautifulSoup
from readability import Document
from string import ascii_lowercase


class WebMDSpider(scrapy.Spider):
    name = "webmd_spider"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.75,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "TEST_MODE": False,          # set True in settings for the 40‑item smoke‑test
    }

    INDEX_TMPL = "https://www.webmd.com/a-to-z-guides/health-topics?pg={}"
    SEL = {
        "index_links": "main a::attr(href)",
        "title": "main h1, h1",
        "article": "main article, main .article-body, main section",
        # metadata helpers
        "published": (
            'meta[itemprop="datePublished"]::attr(content), '
            'meta[name="pubdate"]::attr(content)'
        ),
        "author": (
            '.byline__name::text, '
            '.article-author::text, '
            'meta[name="author"]::attr(content)'
        ),
    }

    # ─────────────────────────── bootstrap
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.test_mode = crawler.settings.getbool("TEST_MODE")
        spider._yielded = defaultdict(int)
        return spider

    # ─────────────────────────── seeds
    def start_requests(self):
        self.logger.info(f"Test mode: {self.test_mode}")
        letters = ascii_lowercase[:2] if self.test_mode else ascii_lowercase
        self.logger.info(f"Requesting A-Z index for {letters}")
        for lt in letters:
            self.logger.info(f"Requesting A-Z index for '{lt}'")
            url = self.INDEX_TMPL.format(lt)
            yield scrapy.Request(url, self.parse_index, meta={"letter": lt})

    # ─────────────────────────── A–Z index
    def parse_index(self, resp):
        lt, seen = resp.meta["letter"], set()
        for raw in resp.css(self.SEL["index_links"]).getall():
            # log the first 20 items in test mode
            if self._yielded[lt] < 20:
                self.logger.info(f"Found link: {raw}")
            if self.test_mode and self._yielded[lt] >= 20:
                break
            clean = self._norm_href(raw, resp.url)
            if not clean or clean in seen or "/health-topics?pg=" in clean:
                continue
            seen.add(clean)
            yield resp.follow(clean, self.parse_hub, meta={"letter": lt})

    # ─────────────────────────── hub page (parent + discover subs)
    def parse_hub(self, resp):
        lt = resp.meta["letter"]
        # 1️⃣ yield the hub/overview itself
        
        yield from self._finalize_item(
            resp,
            letter=lt,
            section_id="overview",
            url=resp.url,
        )

        # 2️⃣ discover sister pages and schedule
        path_parts = urlparse(resp.url).path.split("/")
        folder = "/".join(path_parts[:-1]) + "/"

        seen = set()
        for raw in resp.css("main a::attr(href)").getall():
            full = self._norm_href(raw, resp.url)
            if (
                not full
                or full.rstrip("/") == resp.url.rstrip("/")
                or not full.startswith("https://www.webmd.com" + folder)
            ):
                continue
            if full in seen:
                continue
            seen.add(full)
            slug = (
                full.split("/")[-1]
                .split(".")[0]
                .strip()
                .lower()
                .replace("-", "_")
            )
            if slug in {"default", "reference", ""}:
                continue
            yield resp.follow(
                full,
                self.parse_sub,
                meta={"letter": lt, "section_id": slug},
                dont_filter=True,
            )

    # ─────────────────────────── individual sibling page
    def parse_sub(self, resp):
        yield from self._finalize_item(
            resp,
            letter=resp.meta["letter"],
            section_id=resp.meta["section_id"],
            url=resp.url,
        )

    # ─────────────────────────── helpers
    # create & emit a single article item
    def _finalize_item(self, resp, *, letter, section_id, url):
        body = self._dedupe_lines(self._extract(resp, css=self.SEL["article"]))
        if not body:
            return  # skip empty entries

        item = {
            "url": url,
            "section_id": section_id,            # normalised slug
            "title": resp.css(self.SEL["title"]).xpath("normalize-space()").get("").lstrip(" -–—"),
            "text": body,
            "type": "condition",
            "published": self._get_date(resp),
            "author": resp.css(self.SEL["author"]).xpath("normalize-space()").get(""),
        }
        self._yielded[letter] += 1
        yield item

    # tidy URLs & filter unwanted asset links
    def _norm_href(self, href, base):
        href, _ = urldefrag((href or "").strip())
        if not href or href in {"/", "#"} or href.lower().startswith("javascript:"):
            return None
        href = html.unescape(href)
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("/"):
            href = urljoin(base, href)

        BAD = (
            "/video", "/slideshow", "/quiz", "symptoms.webmd.com",
            "/drugs/", "/news/", "/blogs.", "doctor.webmd.com", "/rx/",
        )
        return None if any(b in href for b in BAD) else href

    # capture YYYY‑MM‑DD if present
    def _get_date(self, resp):
        raw = resp.css(self.SEL["published"]).get("")
        if not raw:
            return ""
        try:
            # allow mm/dd/yyyy or iso
            raw = raw.strip()
            if "/" in raw:
                return dt.datetime.strptime(raw.split("T")[0], "%m/%d/%Y").date().isoformat()
            return dt.datetime.fromisoformat(raw.split("T")[0]).date().isoformat()
        except Exception:
            return ""

    # ----------------------------- text extraction & cleanup
    def _extract(self, resp, *, css=None):
        soup = BeautifulSoup(resp.text, "lxml")
        frag = BeautifulSoup("".join(map(str, soup.select(css))) or str(soup), "lxml")
        BAD = (
            "header, nav, aside, form, script, style, noscript, figure,"
            " .promo, .adslot, .sharebar, .article-recs, .recirc, .footer"
        )
        for n in frag.select(BAD):
            n.decompose()

        raw = BeautifulSoup(Document(str(frag)).summary(), "lxml").get_text(" ", strip=True)
        txt = self._flatten_bullets(raw)

        JUNK = [
            r"^Overview,.+?View Full Guide\.",           # top‑of‑page nav banner
            r"^###?\s*More on .*?$",
            r"Recommended (Videos|Slideshows).*?$",
            r"Sources\s+Update History.*?$",
            r"Need help navigating .*?Newsletter\.",    # promo call‑outs
        ]
        for p in JUNK:
            txt = re.sub(p, " ", txt, flags=re.I | re.M | re.S)

        txt = re.sub(r"https?://\S+", " ", txt)       # strip bare URLs
        txt = re.sub(r"\s{2,}", " ", txt)
        txt = self._dedupe_sentences(txt.strip())
        return txt

    def _flatten_bullets(self, s):
        out, buf = [], []
        for ln in s.splitlines():
            if ln.lstrip().startswith("•"):
                buf.append(ln.lstrip("• ").rstrip(" ,.;:"))
            else:
                if buf:
                    out.append(", ".join(buf) + ".")
                    buf.clear()
                out.append(ln)
        if buf:
            out.append(", ".join(buf) + ".")
        return "\n".join([l for l in out if l.strip()])

    def _dedupe_sentences(self, txt):
        seen, keep = set(), []
        for s in re.split(r"(?<=[.!?])\s+", txt):
            key = re.sub(r"\W+", "", s.lower())
            if key and key not in seen:
                seen.add(key)
                keep.append(s)
        return " ".join(keep)

    def _dedupe_lines(self, txt):
        seen, keep = set(), []
        for ln in txt.splitlines():
            key = re.sub(r"\W+", "", unicodedata.normalize("NFKD", ln.lower()))
            if key and key not in seen:
                seen.add(key)
                keep.append(ln)
        return "\n".join(keep)
