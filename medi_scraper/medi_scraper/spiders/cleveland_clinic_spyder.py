import scrapy
import re
from string import ascii_uppercase
from readability import Document
from bs4 import BeautifulSoup
from scrapy_splash import SplashRequest  # Add SplashRequest

class ClevelandClinicSpider(scrapy.Spider):
    name = "cleveland_clinic_spider"
    allowed_domains = ["my.clevelandclinic.org"]
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.75,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "TEST_MODE": False,
        # Splash settings
        "SPLASH_URL": "http://localhost:8050",  # Ensure Splash is running
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy_splash.SplashCookiesMiddleware": 723,
            "scrapy_splash.SplashMiddleware": 725,
            "scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware": 810,
        },
        "SPIDER_MIDDLEWARES": {
            "scrapy_splash.SplashDeduplicateArgsMiddleware": 100,
        },
        "DUPEFILTER_CLASS": "scrapy_splash.SplashAwareDupeFilter",
    }

    SITE_CONFIGS = {
        "my.clevelandclinic.org": {
            "symptoms": {
                "index_url": "https://my.clevelandclinic.org/health/symptoms",
                "selectors": {
                    "links": "div.az-links-container h3 + ul a::attr(href)",  # Adjusted selector
                    "article": "main, div.page-body, article",
                    "title": "h1::text",
                },
            },
            "diagnostics": {
                "index_url": "https://my.clevelandclinic.org/health/diagnostics",
                "selectors": {
                    "links": "div.az-links-container h3 + ul a::attr(href)",  # Adjusted selector
                    "article": "main, div.page-body, article",
                    "title": "h1::text",
                },
            },
            "diseases": {
                "index_url": "https://my.clevelandclinic.org/health/diseases",
                "selectors": {
                    "links": "div.az-links-container h3 + ul a::attr(href)",  # Adjusted selector
                    "article": "main, div.page-body, article",
                    "title": "h1::text",
                },
            },
        }
    }

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider.test_mode = crawler.settings.getbool("TEST_MODE")
        return spider

    def start_requests(self):
        categories = ["symptoms", "diagnostics", "diseases"]
        site_config = self.SITE_CONFIGS["my.clevelandclinic.org"]
        for category in categories:
            url = site_config[category]["index_url"]
            yield SplashRequest(  # Use SplashRequest instead of scrapy.Request
                url,
                callback=self.parse_index,
                meta={"category": category},
                args={"wait": 2},  # Wait for JavaScript to render
            )

    def parse_index(self, response):
        category = response.meta["category"]
        config = self.SITE_CONFIGS["my.clevelandclinic.org"][category]["selectors"]
        self.logger.info(f"Parsing index for {category}: {response.url}")

        # Extract all links under each letter section
        links = response.css(config["links"]).getall()
        seen = set()
        count = 0
        for href in links:
            if href in seen:
                continue
            seen.add(href)
            full_url = response.urljoin(href)
            self.logger.debug(f"Found {category} link: {full_url}")
            yield SplashRequest(  # Use SplashRequest for detail pages
                full_url,
                callback=self.parse_detail,
                meta={"category": category},
                args={"wait": 1},
            )
            count += 1
            if self.test_mode and count >= 5:
                self.logger.info("Test mode: limiting to 5 links per category")
                break

    def parse_detail(self, response):
        category = response.meta["category"]
        config = self.SITE_CONFIGS["my.clevelandclinic.org"][category]["selectors"]
        title = response.css(config["title"]).get(default="").strip()
        body = self.extract_main_text(response, config["article"])

        yield {
            "url": response.url,
            "type": category,
            "title": title,
            "text": body,
        }

    def extract_main_text(self, response, css_selector):
        soup = BeautifulSoup(response.text, "lxml")
        frag = soup.select_one(css_selector) or soup

        # Remove unwanted elements
        for noise in frag.select("script, style, aside, footer, nav, .hidden, .sr-only"):
            noise.decompose()

        # Attempt Readability extraction
        readable = Document(str(frag)).summary()
        readable_soup = BeautifulSoup(readable, "lxml")
        text = readable_soup.get_text(" ", strip=True)

        # Fallback if content is too short
        if len(text) < 600:
            manual_text = " ".join(
                p.get_text(" ", strip=True)
                for p in frag.select("p, li, h2, h3")
                if p.get_text(strip=True)
            )
            text = manual_text if len(manual_text) > len(text) else text

        # Cleanup common patterns
        patterns = [
            r"Call Appointment Center 24\/7.*?\d+",
            r"To speak with someone directly.*?\d+",
            r"© \d+ Cleveland Clinic\.?",
            r"Last reviewed by a Cleveland Clinic professional on.*?\.?",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        text = re.sub(r"\s{2,}", " ", text).strip()
        return text