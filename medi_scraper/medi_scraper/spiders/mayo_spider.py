import scrapy, re
from string import ascii_uppercase
from readability import Document
from bs4 import BeautifulSoup

class MayoSpider(scrapy.Spider):
    name = "mayo_spider" 
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.75,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        # quick iteration toggle
        "TEST_MODE": False,
    }

    # ─── per‑site knobs ────────────────────────────────────────────
    SITE_CONFIGS = {
        "mayoclinic.org": {
            "a_z": "https://www.mayoclinic.org/diseases-conditions/index?letter={}",
            "selectors": {
                # <main id="mainContent"> since Jan‑2025; keep the older
                # one in the list just in case.
            # grab the whole visible article area, with graceful fallbacks
            "article": (
                ".content, article .aem-container,"     # 2025 markup
                " div#main-content, div#mainContent"       # legacy backup
            ),
                "title":    "main h1, h1[data-test='page-title']",
            },
        },
    }


    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # now crawler.settings is available
        spider.test_mode = crawler.settings.getbool("TEST_MODE")
        return spider
    
    def start_requests(self):
        letters = ascii_uppercase[6:8] if self.test_mode else ascii_uppercase
        cfg = self.SITE_CONFIGS["mayoclinic.org"]
        for letter in letters:
            url = cfg["a_z"].format(letter)
            
            self.logger.info(f"Requesting A-Z index for '{letter}' → {url}")
            yield scrapy.Request(url, callback=self.parse_index)

    def parse_index(self, response):
        self.logger.info(f"Parsing A-Z index: {response.url}")
        links = response.css("a[href*='/diseases-conditions/']::attr(href)").getall()
        seen, count = set(), 0
        for href in links:
            href = href.strip()
            if not href or href in seen:
                continue
            if "index?letter=" in href:
                # skip the generic index
                continue
            seen.add(href)
            full = response.urljoin(href)
            # skip the generic index
            if full.endswith(f"index?letter={response.url.split('=')[-1]}"):
                continue
            self.logger.info(f"Found overview: {full}")
            yield scrapy.Request(full, callback=self.parse_symptoms)
            count += 1
            if self.test_mode and count >= 5:
                self.logger.info("Test mode: limited to 5 overview links")
                break

    def parse_symptoms(self, response):
        """
        1. Pull the Symptoms & Causes article.
        2. If the first grab looks suspiciously short (≤ 600 chars) *or*
        doesn’t start with a typical leading heading, re‑run the extraction
        on a broader fragment taken from the same response ― no extra
        request needed.
        3. Yield the symptoms item, then queue the matching
        “/diagnosis‑treatment/…” page.
        """
        self.logger.info("Parsing symptoms page: %s", response.url)

        sel = self.SITE_CONFIGS["mayoclinic.org"]["selectors"]
        title = (
            response.css(sel["title"]).xpath("normalize-space()").get("")
            or response.css("h1").xpath("normalize-space()").get("")
        )

        # ── primary extraction (fast path) ───────────────────────────────
        body = self.extract_main_text(response, css_selector=sel["article"])

        # ── quick sanity‑check & in‑place fallback ──────────────────────
        looks_short  = len(body) < 600
        missing_head = not body.lstrip().startswith(
            ("Overview", "Symptoms", "Causes", "Definition")
        )

        if looks_short:
            self.logger.debug(
                "Primary grab looked thin (%s, %s) → retrying with broader selector",
                "short" if looks_short else "ok‑len",
                "missing‑head" if missing_head else "has‑head",
            )
            broader = self.extract_main_text(
                response,
                css_selector="main, .content, div#main-content, body",  # wide net
            )
            if len(broader) > len(body):
                body = broader

        # ── emit Symptoms & Causes item ─────────────────────────────────
        yield {
            "url":   response.url,
            "type":  "symptoms-causes",
            "title": title,
            "text":  body,
        }

        # ── queue the matching Diagnosis & Treatment page ───────────────
        link = response.css('a[href*="/diagnosis-treatment/"]::attr(href)').get()
        if link:
            diag_url = response.urljoin(link)
            self.logger.info("Queueing diagnosis page: %s", diag_url)
            yield scrapy.Request(
                diag_url,
                callback=self.parse_diagnosis,
                meta={"title": title},
                dont_filter=True,
            )




    def parse_diagnosis(self, response):
        self.logger.info(f"Parsing diagnosis page: {response.url}")
        title = response.meta.get("title", "")
        sel_cfg = self.SITE_CONFIGS["mayoclinic.org"]["selectors"]
        diagnosis = self.extract_main_text(response, css_selector=sel_cfg["article"])

        yield {
            "url":  response.url,
            "type": "diagnosis-treatment",
            "title": title,
            "text":  diagnosis,
        }

    
    def extract_main_text(self, response, *, css_selector=None) -> str:
        """
        1. Grab the first node that matches *css_selector* (or whole page if None)
        2. Try `readability-lxml` on that fragment
        3. If the result is < 50 chars, build text manually from every
           <p> / <li> **inside the same fragment**, excluding obvious noise.
        """
        soup = BeautifulSoup(response.text, "lxml")
        frag = soup.select_one(css_selector) if css_selector else soup

        # If we still didn’t find anything, fall back to the whole <body>.
        if frag is None:
            self.logger.debug("► selector %s not found on %s", css_selector, response.url)
            frag = soup.body or soup         # last‑ditch attempt
        # 1️⃣ prune obvious noise first  ─────────────────────────
        frag_copy = BeautifulSoup(str(frag), "lxml")      # work on a copy
        bad_sel = (
            "aside, form, nav, script, style,"
            " .mc-callout,"               # Care‑at‑Mayo advert
            " .requestappt,"              # ‘Request an appointment’ button
            " div.contentbox.newsletter," # <- be precise!
            " .myc-subscription-form,"
            " .acces-list-container,"
            " div[data-nosnippet='true']"
        )
        for bad in frag_copy.select(bad_sel):
            bad.decompose()

        # 2️⃣ now run Readability on the cleaned fragment
        readable = Document(str(frag_copy)).summary()
        txt = BeautifulSoup(readable, "lxml").get_text(" ", strip=True)
        if len(txt) >= 50:
            JUNK_PATTERNS = [
                r"Products\s*&\s*Services.*?$",
                r"From Mayo Clinic to your inbox.*?Retry",
                r"Show more products.*?$",
                r"Email field is required.*?Retry",
                r"^Request an appointment$"
                r"Products\s*&\s*Services.*?(?=Symptoms|$)"
                ]
            for pat in JUNK_PATTERNS:
                txt = re.sub(pat + r".*?(?=\n|$)", "", txt, flags=re.I|re.M|re.S)
            return txt
        # return ""
        # Manual fallback – strip banners / forms / ads / share bars
        for bad in frag.select(bad_sel):
            bad.decompose()
        parts = [n.get_text(" ", strip=True) for n in frag.select("p, li") if n.get_text(strip=True)]

        text =  " ".join(parts)
        # remove any junk patterns
        for pattern in JUNK_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.I|re.S)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
        



