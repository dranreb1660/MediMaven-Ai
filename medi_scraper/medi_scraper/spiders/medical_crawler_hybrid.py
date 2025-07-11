import scrapy
import logging
import re
import os
import json
from string import ascii_uppercase
from urllib.parse import urlparse
from twisted.internet.error import TimeoutError, DNSLookupError, ConnectionRefusedError
from scrapy.spidermiddlewares.httperror import HttpError
from functools import lru_cache
from w3lib.url import canonicalize_url
from scrapy_splash import SplashRequest
from scrapy import Selector

class MedicalSpider(scrapy.Spider):
    """
    Hybrid medical‑content spider
    --------------------------------
    *  JOBDIR & HTTPCACHE  →  resumable, polite long crawls
    *  Per‑domain SITE_CONFIGS  →  easy to extend
    *  Adaptive validation      →  keeps only useful pages
    *  Optional render_all per domain for heavy JS sites
    *  Optimized URL validation with caching
    *  Enhanced error handling with fallback mechanisms
    *  Memory-efficient processing for large crawls
    """
    name = "medical_crawler_hybrid"
    allowed_domains = [
        "mayoclinic.org",
        "nhs.uk",
        "webmd.com",
        "clevelandclinic.org",  # Changed from my.clevelandclinic.org for consistency
        "localhost"
    ]

    # ─────────────────────────── SETTINGS ──────────────────────────
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "DOWNLOAD_DELAY": 0.75,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "HTTPCACHE_ENABLED": False,
        "HTTPCACHE_EXPIRATION_SECS": 86400 * 3,          # 3 days
        # "JOBDIR": "crawls/medical_crawler_hybrid",       # resume support
        
        # Timeout settings
        "DOWNLOAD_TIMEOUT": 90,                          # Short timeout
        "SPLASH_COOKIES_DEBUG": True,                    # Debug cookies
        "SPLASH_LOG_400": True,                          # Log 400 errors
        
        # Retry settings - consolidated
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,                                # Default retry count
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],

        # Splash
        "SPLASH_URL": "http://localhost:8050",
        "DUPEFILTER_CLASS": "scrapy_splash.SplashAwareDupeFilter",
        "HTTPCACHE_STORAGE": "scrapy_splash.SplashAwareFSCacheStorage", # Required for Splash caching

        # Middlewares
        "SPIDER_MIDDLEWARES": {
            "scrapy_splash.SplashDeduplicateArgsMiddleware": 100,
        },
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy_splash.SplashCookiesMiddleware": 723,
            "scrapy_splash.SplashMiddleware": 725,
            "scrapy.downloadermiddlewares.httpcompression."
            "HttpCompressionMiddleware": 810,
            "scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware": 100,
        },

        # Memory management
        "MEMUSAGE_ENABLED": True,
        "MEMUSAGE_LIMIT_MB": 2048,
        "MEMUSAGE_WARNING_MB": 1536,

        # UA
        "USER_AGENT": "MediScraper/3.0 (+https://medimaven.ai) research bot",
        "TEST_MODE": False,
    }

        
    # ──────────────── PER‑DOMAIN CONFIG (easy to externalise) ───────────────
    SITE_CONFIGS = {
        "mayoclinic.org": {
            "a_z": "https://www.mayoclinic.org/diseases-conditions/index?letter={}",
           "url_patterns": [
            "/diseases-conditions/",
            "/symptoms-causes/",
            "/diagnosis-treatment/"
        ],
        "selectors": {
            "sections": "div.main-content article, div.content",
            "heading":  "h2::text, h3::text",
            #           ↓↓↓  added “ul li::text”
            "body":     "div.content p::text, article p::text, "
                        "div.mayoContent p::text, ul li::text",
            },
            "render_all": True,  # heavy JS
        },
        "nhs.uk": {
            "a_z": "https://www.nhs.uk/conditions/?char={}",     # <── FIX
            "url_patterns": ["/conditions/"],
            "selectors": {
                "sections": "main.nhsuk-width-container article,"
                            " main.nhsuk-main-wrapper article,"
                            " div.nhsuk-grid-column-two-thirds",
                "heading":   "h2::text, h3.nhsuk-heading-m::text",
                "body":      "article p::text, div.nhsuk-grid-column-two-thirds p::text, ul li::text",
            },
            "render_all": False,
        },
        "webmd.com": {
            # WebMD expects lowercase letters
            "a_z": "https://www.webmd.com/a-to-z-guides/health-topics?pg={}",
            "url_patterns": ["/a-to-z-guides/", "/vitamins/", "/drugs/", "/diabetes/"],
            "selectors": {
            "sections": "div#medical-reference, div.article-page, section.article-section",
            "heading": "h2::text, h3.article-header::text, h3.article-subtitle::text",
            "body": (
                "div#medical-reference p::text, div.article-content p::text, "
                "div.article-body p::text, section.article-section ul li::text"
            ),
            },
            "render_all": False,
        },
        "clevelandclinic.org": {
            "a_z": "https://my.clevelandclinic.org/AtoZ/HealthInformationLetterStatus",
            "url_patterns": [
                "/health/diseases/",
                "/health/symptoms/",
                "/health/diagnostics/",
            ],
            "render_all": True,
            "selectors": {
                # scope everything to the main content column
                "root": 'div[data-identity="main-article-content"]',
                # each top‐level section
                "sections": 'div[data-identity="article-section"]',
                # section title
                "heading": 'h2[data-identity="headline"]::text',
                # paragraphs, bullets, sub‑heads
                "body": (
                    'div[data-identity="rich-text"] '
                    'p[data-identity="paragraph-element"]::text, '
                    'div[data-identity="rich-text"] '
                    'ul[data-identity="unordered-list"] li::text, '
                    'div[data-identity="rich-text"] '
                    'h3[data-identity="headline"]::text, '
                    'div[data-identity="rich-text"] '
                    'h4[data-identity="headline"]::text'
                ),
            },
        },

    }
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # now crawler.settings is available
        spider.test_mode = crawler.settings.getbool("TEST_MODE")
        return spider

    @staticmethod
    def needs_splash(url: str) -> bool:
        """
        Return True only for pages that require JavaScript
        (tweak the heuristic for your project).
        """
        path = urlparse(url).path
        return path.endswith(("/videos", "/interactive"))

    def _make_request(self, url: str, *, callback, force_splash: bool = False,**extra):
        """
        Decide whether to return a SplashRequest or a plain Request.

        Any additional keyword arguments (dont_filter, meta, headers…)
        are forwarded to the underlying Request so callers don’t lose
        fine‑grained control.
        """
        url = canonicalize_url(url)
        domain = urlparse(url).netloc.replace("www.", "")
        # normalize any Cleveland subdomain
        if "clevelandclinic" in domain:
            domain = "clevelandclinic.org"

        cfg = self.SITE_CONFIGS.get(domain, {})
        render_all = cfg.get("render_all", False)

        if force_splash or render_all or self.needs_splash(url):
            # Allow the caller to over‑ride args/meta via **extra if they
            # really need to, but otherwise fall back to our splash().
            return self.splash(url, callback, render_all=True)

        # ---------- plain Request ----------
        default_headers = {
            "User-Agent": self.custom_settings["USER_AGENT"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        # Let callers inject/override headers if they passed them in **extra
        headers = extra.pop("headers", {})
        default_headers.update(headers)

        # Set a sane default meta, merge caller‑supplied one (if any)
        meta = {"direct_request": True}
        meta.update(extra.pop("meta", {}))

        return scrapy.Request(
            url=url,
            callback=callback,
            headers=default_headers,
            meta={**meta, "dont_redirect": True},
            **extra,
        )
    def _cc_start(self, letter=None):
        """
        Kick off Cleveland Clinic by POSTing to HealthInformationLetterStatus
        to get counts (or just POST directly for each letter if you don't care about counts).
        """
        form = {"category": "Diseases & Conditions"}
        # if you only want a single letter in test mode:
        if letter:
            form["letter"] = letter
            yield scrapy.FormRequest(
                "https://my.clevelandclinic.org/AtoZ/HealthInformationPages",
                formdata=form,
                callback=self._cc_parse_pages
            )
        else:
            # full‐blown: first get the count per letter
            yield scrapy.FormRequest(
                "https://my.clevelandclinic.org/AtoZ/HealthInformationLetterStatus",
                formdata=form,
                callback=self._cc_parse_counts
                
                
            )
            
    def _cc_parse_counts(self, response):
        counts = json.loads(response.text)  # e.g. {"A":123, …}
        for letter in counts:
            yield scrapy.FormRequest(
                "https://my.clevelandclinic.org/AtoZ/HealthInformationPages",
                formdata={"category": "Diseases & Conditions", "letter": letter},
                callback=self._cc_parse_pages
            )

    def _cc_parse_pages(self, response):
        data = json.loads(response.text)
        # Cleveland returns a top‑level list of { "title", "url", "description", "cat" }
        entries = data if isinstance(data, list) else data.get("items", [])
        for entry in entries:
            yield self._make_request(
                response.urljoin(entry["url"]),    # note lowercase “url”
                callback=self.parse_medical_page,
                meta={"direct_request": True},
            )
            
    # ────────────────────────── SPIDER STARTUP ──────────────────────────
    # ➊ ────────────────────────── START REQUESTS ──────────────────────────
    def start_requests(self):
        # 1) Kick off Cleveland Clinic via JSON POST (only once)
        #    — in TEST_MODE we limit to “A”, otherwise fetch counts for all letters
        yield from self._cc_start(
            letter=ascii_uppercase[0] if self.test_mode else None
        )

        # 2) All the other sites via their A–Z HTML index
        letters = ascii_uppercase[:1] if self.test_mode else ascii_uppercase
        for domain, cfg in self.SITE_CONFIGS.items():
            if domain == "clevelandclinic.org":
                continue
            tpl = cfg["a_z"]
            if "{}" in tpl:
                for L in letters:
                    val = L.lower() if "webmd.com" in domain else L
                    url = tpl.format(val)
                    self.logger.info(f"Creating direct request for A‑Z index: {url}")
                    yield self._make_request(url, callback=self.parse_az_index)
            else:
                self.logger.info(f"Creating direct request for A‑Z index: {tpl}")
                yield self._make_request(tpl, callback=self.parse_az_index)

        # 3) (Optional) Sitemaps, only in full run
        if not self.test_mode:
            for url in [
                "https://www.mayoclinic.org/sitemap",
                "https://www.nhs.uk/sitemap.xml",
                "https://www.webmd.com/sitemap.xml",
                "https://my.clevelandclinic.org/sitemap.xml",
            ]:
                yield self.splash(url, self.parse_sitemap)


    # ➋ ────────────────────────── ROUTER ──────────────────────────
    def parse(self, response):
        url = response.url
        if "sitemap" in url:
            yield from self.parse_sitemap(response)
        elif any(k in url for k in ["a-to-z", "index?letter="]):
            yield from self.parse_az_index(response)
        else:
            yield from self.parse_medical_page(response)

    # ➌ ────────────────────────── SITEMAP ─────────────────────────
    def parse_sitemap(self, response):
        # Handle sitemap index
        for loc in response.xpath("//sitemap/loc/text()").getall():
            if any(p in loc for p in ["condition", "disease", "health"]):
                yield self._make_request(loc, callback=self.parse_sitemap, force_splash=True)

        # URLs inside sitemap
        for loc in response.xpath("//url/loc/text()").getall():
            if self.is_medical_url(loc):
                yield self._make_request(loc, callback=self.parse_medical_page)

    # ➍ ─────────────────────── A‑Z INDEX PAGES ─────────────────────
    def parse_az_index(self, response):
        """Parse A-Z index pages."""
        try:
            self.logger.info(f"Processing A-Z index page: {response.url}")
            links_yielded = 0
            
            # Check for Splash JSON response - if so, immediately fall back to direct request
            if hasattr(response, 'selector') and getattr(response.selector, 'type', None) == 'json':
                self.logger.info(f"Detected Splash JSON response, falling back to direct request: {response.url}")
                # Properly yield the request instead of returning it
                yield self._make_request(
                    response.url,
                    callback=self.parse_az_index,
                    dont_filter=True,
                    meta={'direct_request': True},
                )
            
            # Process links from standard HTML response
            try:
                domain = urlparse(response.url).netloc.replace('www.', '')
                cfg = self.SITE_CONFIGS.get(domain, {})
                
                # Extract links using CSS selector
                raw_links = response.css("a::attr(href)").getall()
                self.logger.info(f"Found {len(raw_links)} total links on page {response.url}")
                seen = set()
                # Track medical URLs for logging
                medical_urls_count = 0
                processed_urls = 0
                
                # Process the links
                for href in raw_links:
                    href = href.strip()
                    # skip empty or duplicate hrefs
                    if not href or href in seen:
                        continue
                    seen.add(href)
                    
                    # normalize domain for lookups
                    dom = domain
                    if "clevelandclinic" in dom:
                        dom = "clevelandclinic.org"

                    # skip the bare hub page (e.g. “/conditions/”)
                    full = response.urljoin(href)
                    if full.rstrip("/").lower() == cfg.get("a_z", "").rstrip("/").lower():
                        continue

                    if self.is_medical_url(href, dom):
                        medical_urls_count += 1
                        url = response.urljoin(href)
                        if self.should_crawl(url):
                            processed_urls += 1
                            yield self._make_request(url,callback=self.parse_medical_page)
                            links_yielded += 1
                        if self.test_mode and links_yielded >= 5:
                            self.logger.info("Test mode: limiting to 5 links")
                            break
                            

                
                self.logger.info(f"Processed {processed_urls} out of {medical_urls_count} medical URLs found on {response.url}")
                
            except Exception as e:
                self.logger.error(f"Error extracting links from {response.url}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error processing A-Z index page {response.url}: {str(e)}", exc_info=True)

    # ➎ ───────────────────── MEDICAL CONTENT PAGES ─────────────────



    def parse_medical_page(self, response):
        """Universal medical‑page parser for Cleveland Clinic, Mayo, NHS, WebMD, etc."""
        try:
            url = response.url
            domain = urlparse(url).netloc.replace("www.", "")
            # normalize Cleveland sub‑domains
            if "clevelandclinic" in domain:
                domain = "clevelandclinic.org"
            cfg = self.SITE_CONFIGS.get(domain, {})
            sel_cfg = cfg.get("selectors")

            # 1) Bail out early on known A–Z/index pages for each site
            #    (so we don’t treat them as “content”)
            if domain == "webmd.com" and re.search(r"/(?:index|default\.htm)$", url, re.I):
                self.logger.info(f"Skipping WebMD index: {url}")
                return
            if domain == "nhs.uk" and re.match(r".*/conditions/?$", url):
                self.logger.info(f"Skipping NHS Conditions index: {url}")
                return
            if domain.endswith("mayoclinic.org") and re.search(r"/index(\?letter=[A-Z])?$", url):
                self.logger.info(f"Skipping Mayo Clinic index: {url}")
                return

            if not sel_cfg:
                self.logger.warning(f"No selectors for domain {domain}; skipping {url}")
                return

            # 2) If Splash’s Lua ‘execute’ endpoint returned JSON, re‑hydrate it
            if hasattr(response, "data") and isinstance(response.data, dict) and "html" in response.data:
                html = response.data["html"]
                selector = Selector(text=html)
                self.logger.info(f"Using Splash JSON→HTML for {url}")
            else:
                selector = response  # plain TextResponse or render.html

            # 3) Logging snippet
            snippet = selector.xpath("string()").get(default="").strip().replace("\n"," ")[:200]
            self.logger.info(f"Parsing {url} (domain={domain}) → {snippet!r}...")

            # 4) Build our item skeleton
            item = {
                "url": url,
                "domain": domain,
                "title": selector.css("title::text").get(default="").strip(),
                "last_updated": selector.css(
                    'meta[property="article:modified_time"]::attr(content)'
                ).get(),
                "content_sections": [],
                "medical_entities": [],
            }

            # 5) Scope to “root” if the config asks for it
            root_sel = sel_cfg.get("root")
            roots = selector.css(root_sel) if root_sel else [selector]
            if root_sel and not roots:
                self.logger.warning(f"Root selector {root_sel!r} found nothing on {url}")
                roots = [selector]

            # 6) Pull out each configured “section” block
            for root in roots:
                secs = root.css(sel_cfg["sections"])
                if not secs:
                    # fallback: any <p> under a “main-content” area
                    paras = selector.css("div.main-content p::text, article p::text").getall()
                    text = " ".join(p.strip() for p in paras if p.strip())
                    if len(text) > 100:
                        item["content_sections"].append({
                            "heading": "Content",
                            "body": self.clean_text(text)
                        })
                else:
                    for sec in secs:
                        heading = sec.css(sel_cfg["heading"]).get(default="Section").strip()
                        # skip newsletter/inbox boilerplate
                        if re.search(r"newsletter|inbox", heading, re.I):
                            continue

                        bits = sec.css(sel_cfg["body"]).getall()
                        if not bits:
                            bits = sec.css("p::text, ul li::text").getall()

                        raw = self._join_body_bits(bits)
                        clean = self.clean_text(raw)
                        clean = self._strip_mayo_noise(clean)
                        if len(clean) >= 50:
                            item["content_sections"].append({
                                "heading": heading,
                                "body": clean
                            })

            # 7) “Symptoms → Diagnosis” twin‑page queue re‑use exactly as before
            if "/symptoms-causes/" in url and not response.meta.get("alt_done"):
                link = (
                    selector.css('a#et_genericNavigation_diagnosis-treatment::attr(href)').get()
                    or selector.css('a[href*="/diagnosis-treatment/"]::attr(href)').get()
                )
                alt = response.urljoin(link) if link else url.replace(
                    'symptoms-causes', 'diagnosis-treatment'
                )
                if self.should_crawl(alt):
                    yield self._make_request(
                        alt,
                        callback=self.parse_medical_page,
                        meta={**response.meta, "alt_done": True}
                    )

            # 8) Validate & emit
            if self.validate_item(item):
                yield item

        except Exception:
            self.logger.exception(f"Error in parse_medical_page() for {response.url}")

    # ─────────────────────────── HELPERS ──────────────────────────
    def splash(self, url, callback, render_all=False):
        """Utility wrapper for SplashRequest with minimal rendering."""
        self.logger.info(f"Making Splash request for {url}")
        
        # For A-Z index pages, use a simpler approach with render.html endpoint
        # to avoid JSON response format issues
        if callback in (self.parse_az_index, self.parse_sitemap):
            args = {
                'wait': 1.0,        # Increased wait time
                'images': 0,        # Disable image loading
                'timeout': 12,      # Increased timeout
                'http_method': 'GET',
                'resource_timeout': 8  # Resource-specific timeout
            }
            
            return SplashRequest(
                url=url,
                callback=callback,
                errback=self.errback_handler,
                endpoint='render.html',  # Use render.html to get direct HTML, not JSON
                args=args,
                meta={
                    'splash_url': url,
                    'dont_retry': False,
                    'max_retry_times': 1
                },
                headers={
                    'User-Agent': self.custom_settings['USER_AGENT']
                }
            )
        
        # For other pages, use the optimized Lua script approach
        lua_script = """
            function main(splash, args)
            -- turn off cruft
            splash.images_enabled  = false
            splash.webgl_enabled   = false
            splash.plugins_enabled = false

            -- set a realistic UA
            splash:set_user_agent(args.user_agent)

            -- navigate to the URL
            local ok, reason = splash:go{
                url         = args.url,
                headers     = args.headers,
                http_method = args.http_method or "GET",
            }
            if not ok then
                return { error = reason, details = "Failed to load URL" }
            end

            -- progressive wait: break once page stops growing
            local start_time = splash:now()
            local last_size  = 0
            local stable     = 0
            while (splash:now() - start_time) < (args.max_wait or 3) do
                splash:wait(0.3)
                local new_size = splash:evaljs("document.body.innerHTML.length")
                local ready    = splash:evaljs(
                "['interactive','complete'].includes(document.readyState)"
                )
                if (new_size > 1000 and ready)
                or (new_size > 500 and new_size == last_size) then
                stable = stable + 1
                if stable >= 2 then break end
                else
                stable = 0
                end
                last_size = new_size
            end

            -- click cookie banner if present
            local dismissed = splash:evaljs([[
                (function(){
                var btn = Array.from(document.querySelectorAll('button, a, div'))
                                .find(el => {
                                var t = el.textContent.toLowerCase();
                                return t.includes('accept')
                                    && (t.includes('cookie') || t.includes('privacy'));
                                });
                if(btn){ btn.click(); return true; }
                return false;
                })();
            ]])
            if dismissed then
                splash:wait(0.3)
            end

            -- final wait for any trailing JS
            splash:wait(args.final_wait or 0.2)

            -- return the HTML, URL (in case of redirects), HTTP status, and how long it took
            return {
                html        = splash:html(),
                url         = splash:url(),
                status      = splash.status,
                render_time = math.floor(splash:now() - start_time),
            }
            end

        """
        
        # Enhanced args with better timeout configuration
        args = {
            'lua_source': lua_script,
            'url': url,
            'headers': {'User-Agent': self.custom_settings["USER_AGENT"]},
            'user_agent': self.custom_settings["USER_AGENT"],
            'timeout': 15,  # Increased timeout for better page loading
            'wait': 1.0     # Default wait time if needed
        }
        
        # Simple meta without callback objects
        meta = {
            'splash_url': url,  # Store original URL for fallback
            'dont_retry': False,  # Allow retries
            'max_retry_times': 1,  # Only retry once before falling back
            'handle_httpstatus_list': list(range(400, 405)) + list(range(407, 505))
        }
        
        return SplashRequest(
            url=url,
            callback=callback,
            errback=self.errback_handler,
            endpoint='execute',  # Use execute endpoint for Lua scripting
            args=args,
            meta=meta,
        )
        
    def errback_handler(self, failure):
        """Handle errors by falling back to direct requests with comprehensive error handling."""
        url = failure.request.meta.get('splash_url', failure.request.url)
        callback = failure.request.callback
        
        # More detailed error logging by error type
        if failure.check(HttpError):
            response = failure.value.response
            self.logger.error(f"HTTP error {response.status} on {url}")
        elif failure.check(DNSLookupError):
            self.logger.error(f"DNS lookup error on {url}")
        elif failure.check(TimeoutError):
            self.logger.error(f"Timeout error on {url}")
        elif failure.check(ConnectionRefusedError):
            self.logger.error(f"Connection refused on {url}")
        else:
            self.logger.error(f"Unhandled error on {url}: {repr(failure.value)}")
        
        # Fall back to direct request with robust headers
        self.logger.info(f"Falling back to direct request for {url}")
        return self._make_request(
            url,
            callback=callback,
            # dont_filter=True,
            meta={
                'direct_request': True,
                'dont_retry': True,
                'download_timeout': 20,
            },
        )

    # ────────────────────────── URL VALIDATION ──────────────────────────
    @lru_cache(maxsize=10000)
    def is_medical_url(self, url, domain=None):
        domain = domain or urlparse(url).netloc.replace("www.", "")

        # ——— Cleveland Clinic: only diseases, symptoms, diagnostics ———
        if domain.endswith("clevelandclinic.org"):
            return bool(re.search(r"/health/(?:diseases|symptoms|diagnostics)/", url))

        # ——— NHS ———
        if domain == "nhs.uk":
            return bool(re.match(r".*/conditions/[^/]+/?$", url))

        # ——— Mayo Clinic ———
        if "mayoclinic.org" in domain:
            if any(loc in url for loc in ("/zh-hans/", "/es/", "/ar/")):
                return False
            if re.search(r"/index(\?letter=[A-Z])?$", url):
                return False
            if url.lower().endswith((".pdf", ".jpg", ".png", ".gif", ".mp4", ".zip", ".doc", ".docx")):
                return False
            return any(p in url for p in self.SITE_CONFIGS["mayoclinic.org"]["url_patterns"])

        # ——— WebMD & fallback ———
        pats = self.SITE_CONFIGS.get(domain, {}).get("url_patterns", [])
        return any(p in url for p in pats)

    
    def should_crawl(self, url, domain=None):
        return self.is_medical_url(url, domain)
    
    def _yield_az_requests(self):
        for domain, cfg in self.SITE_CONFIGS.items():
            tpl = cfg["a_z"]
            # If the template still contains “{}” we generate 26 URLs,
            # otherwise we hit the single index page once.
            if "{}" in tpl:
                for letter in ascii_uppercase:
                    yield tpl.format(letter)
            else:
                yield tpl

    
    def _join_body_bits(self, texts):
        """Preserve bullets: first item plain, subsequent items prefixed with '• '."""
        joined = []
        bullet = False
        for t in texts:
            t = t.strip()
            if not t:
                continue
            if bullet:
                joined.append('• ' + t)
            else:
                joined.append(t)
                bullet = True if t.endswith(':') else bullet
        return ' '.join(joined)
    def validate_item(self, item):
        """Adaptive: ≥900 chars → length only, else ≥300 chars AND keyword"""
        if not item["title"]:
            self.logger.warning(f"Missing title for {item['url']}")
            return False
            
        # Check content sections
        if not item["content_sections"]:
            self.logger.warning(f"No content sections for {item['url']}")
            return False
            
        text = " ".join(sec["body"] for sec in item["content_sections"])
        if len(text) < 300 or len(text.split()) < 50:
            self.logger.warning(f"Insufficient content length ({len(text)} chars) for {item['url']}")
            return False

        KEYWORDS = [
            "symptoms",
            "causes",
            "treatment",
            "diagnosis",
            "overview",
            "condition",
            "disease",
            "health",
            "medical",
            "doctor",
            "patient",
            "therapy",
            "medicine",
            "clinical",
            "care"
        ]
        
        if len(text) >= 900:
            self.logger.info(f"Long content ({len(text)} chars) accepted for {item['url']}")
            return True
            
        found_keywords = [k for k in KEYWORDS if k in text.lower()]
        if not found_keywords:
            self.logger.warning(f"No medical keywords found in {item['url']}")
            return False
            
        self.logger.info(f"Content validated with {len(found_keywords)} keywords for {item['url']}")
        return True


    # ───────────────────────── TEXT UTILITIES ─────────────────────────
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Collapse runs of whitespace and strip leading / trailing spaces.
        """
        import re
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_mayo_noise(text: str) -> str:
        """
        Remove newsletter sign‑ups, copyright blocks, etc.
        """
        garbage = [
            r"From Mayo Clinic to your inbox.*?(unsubscribe link in the email\.)?",
            r"©\s*\d{4}\s*Mayo Foundation for Medical Education and Research\.",
            r"This site complies with the HONcode standard.*?Verify here\.",
            r"We do not endorse.*?non‑profit mission",
            r"Email field is required.*",
            r"Include a valid email address.*",
            r"Sign up for free.*?stay up to date on research advancements.*",
        ]
        for g in garbage:
            text = re.sub(g, "", text, flags=re.I | re.S)
        return text.strip()