import logging, re, scrapy
from urllib.parse import urlparse
from scrapy_splash import SplashRequest
from twisted.internet.error import TimeoutError, TCPTimedOutError
from scrapy.spidermiddlewares.httperror import HttpError
from typing import Dict, List, Optional, Set

class MedicalSpider(scrapy.Spider):
    name = "medical_crawler_v2"
    allowed_domains = ["mayoclinic.org", "nhs.uk", "clevelandclinic.org", "webmd.com"]
    
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 16,  # Increased from 1
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,  # Increased from 1
        "DOWNLOAD_TIMEOUT": 30,  # Increased from 15
        "SPLASH_URL": "http://localhost:8050",
        "DUPEFILTER_CLASS": "scrapy_splash.SplashAwareDupeFilter",
        "SPIDER_MIDDLEWARES": {
            "scrapy_splash.SplashDeduplicateArgsMiddleware": 100,
        },
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy_splash.SplashCookiesMiddleware": 723,
            "scrapy_splash.SplashMiddleware": 725,
            "scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware": 810,
        },
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,  # Increased from 1
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "RETRY_PRIORITY_ADJUST": 1,  # Retry with higher priority
        "RETRY_BACKOFF_BASE": 2,  # Progressive backoff
        "RETRY_BACKOFF_MAX": 60,
        "COOKIES_ENABLED": False,
        "HTTPCACHE_ENABLED": False,
        "ITEM_PIPELINES": {
            "medi_scraper.pipelines.PydanticValidationPipeline": 300,
        },
    }

    # Site-specific configurations
    SITE_CONFIGS = {
        "mayoclinic.org": {
            "timeout": 20,
            "selectors": [
                "#main-content p::text",
                ".content p::text",
                "article p::text"
            ]
        },
        "nhs.uk": {
            "timeout": 20,
            "selectors": [
                "main p::text",
                ".nhsuk-main-wrapper p::text",
                ".nhsuk-grid-column p::text"
            ]
        },
        "clevelandclinic.org": {
            "timeout": 35,  # Extended timeout
            "selectors": [
                ".article-content p::text",
                ".js-article-content p::text",
                "#article-content p::text",
                "div.body-copy *::text",  # Added broader selector
                ".content-section__text p::text",
                "[class*='content'] p::text",
                ".section p::text",
                ".article-body p::text"  # Added fallback
            ]
        },
        "webmd.com": {
            "timeout": 25,
            "selectors": [
                "#mainContent p::text",
                ".article-content p::text",
                ".article-body p::text",
                "[class*='content'] p::text",
                "article p::text"
            ]
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retried_urls = set()
        self.failed_urls = set()

    def start_requests(self):
        """Test URLs with domain-specific settings."""
        test_urls = [
            "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
            "https://www.nhs.uk/conditions/diabetes/",
            "https://my.clevelandclinic.org/health/diseases/7104-diabetes-mellitus-an-overview",
            "https://www.webmd.com/diabetes/diabetes-types-insulin"
        ]
        
        for url in test_urls:
            domain = urlparse(url).netloc.replace('www.', '')
            timeout = self.SITE_CONFIGS.get(domain, {}).get('timeout', 20)
            
            self.logger.info(f"Starting request for {url} (timeout: {timeout}s)")
            yield self.make_request(url, use_splash=True, timeout=timeout)

    def make_request(self, url: str, use_splash: bool = True, timeout: int = 20) -> scrapy.Request:
        """Create either a Splash or regular request with appropriate settings."""
        common_meta = {
            'url': url,
            'try_without_splash': True,
            'dont_merge_cookies': True,  # Performance optimization
            'max_retry_times': 3,
            'download_timeout': timeout
        }
        
        if use_splash:
            return SplashRequest(
                url,
                self.parse_medical_page,
                endpoint='render.html',
                args={
                    'wait': 2,
                    'timeout': min(timeout, 20),  # Cap Splash timeout
                    'images': 0,
                    'resource_timeout': 5,
                    'js_source': "document.documentElement.scrollTop = document.documentElement.scrollHeight/2;"
                },
                dont_filter=True,
                errback=self.handle_error,
                meta=common_meta
            )
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            }
            return scrapy.Request(
                url,
                self.parse_medical_page,
                dont_filter=True,
                headers=headers,
                meta=common_meta
            )

    def handle_error(self, failure):
        """Handle errors with fallback and logging."""
        request = failure.request
        url = request.meta.get('url')
        
        if failure.check(TimeoutError, TCPTimedOutError, HttpError):
            if url not in self.retried_urls and request.meta.get('try_without_splash'):
                self.logger.info(f"Retrying {url} without Splash due to {failure.value}")
                self.retried_urls.add(url)
                timeout = request.meta.get('download_timeout', 20)
                return self.make_request(url, use_splash=False, timeout=timeout)
            elif url not in self.failed_urls:
                self.failed_urls.add(url)
                self.logger.error(f"Failed to fetch {url} after retries: {failure.value}")
                # Log failure for monitoring
                self.crawler.stats.inc_value(f'failed_urls/{failure.value.__class__.__name__}')

    def parse_medical_page(self, response):
        """Extract medical content with improved error handling."""
        url = response.meta.get('url')
        domain = urlparse(url).netloc.replace('www.', '')
        base_domain = next((d for d in self.allowed_domains if d in domain), None)
        
        if not base_domain:
            return
            
        self.logger.info(f"Processing {url}")
        
        # Get site-specific configuration
        config = self.SITE_CONFIGS.get(base_domain, {})
        
        # Initialize item
        item = {
            "url": url,
            "domain": domain,
            "title": self.clean_text(response.css("h1::text").get()),
            "last_updated": response.css('meta[property="article:modified_time"]::attr(content)').get(),
            "medical_entities": set(),
            "content_sections": []
        }

        # Extract content
        content = []
        seen_content = set()
        
        # Try all selectors
        for selector in config.get("selectors", ["p::text"]):
            for element in response.css(selector):
                # Get text from element and children
                texts = [self.clean_text(t) for t in element.css("::text").getall()]
                text = " ".join(t for t in texts if t)
                
                if text and len(text) > 20 and text not in seen_content:
                    content.append(text)
                    seen_content.add(text)

        if content:
            # Join unique paragraphs
            text = " ".join(content)
            if len(text) >= 200:
                self.logger.info(f"Found {len(content)} paragraphs with total length {len(text)}")
                
                item["content_sections"].append({
                    "heading": item["title"] or "Overview",
                    "body": text,
                    "type": "medical_content"
                })
                
                # Extract medical terms
                item["medical_entities"] = self.extract_medical_terms(text)
                
                if len(item["medical_entities"]) >= 2:
                    self.logger.info(f"Successfully extracted {len(text)} chars with {len(item['medical_entities'])} medical terms from {domain}")
                    item["medical_entities"] = list(item["medical_entities"])
                    return item
        
        self.logger.info(f"Insufficient content from {domain}")

    def clean_text(self, text: Optional[str]) -> str:
        """Clean text with improved handling of common patterns."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Subscribe|Newsletter|Sign up|Share|Email|Privacy|Copyright', '', text, flags=re.I)
        text = re.sub(r'Click here|Read more|Learn more', '', text, flags=re.I)
        text = re.sub(r'Last reviewed|Next review|Page updated', '', text, flags=re.I)
        return text.strip()

    def extract_medical_terms(self, text: str) -> Set[str]:
        """Extract medical terms with expanded vocabulary."""
        medical_terms = {
            # Core medical terms
            'symptoms', 'causes', 'treatment', 'diagnosis',
            'condition', 'disease', 'health', 'medical',
            'patient', 'therapy', 'chronic', 'medicine',
            
            # Common medical concepts
            'blood', 'risk', 'care', 'clinical', 'doctor',
            'hospital', 'surgery', 'prescription', 'medication',
            
            # Diabetes-specific terms
            'diabetes', 'insulin', 'glucose', 'sugar', 'pancreas',
            'type 1', 'type 2', 'blood sugar', 'hyperglycemia'
        }
        text_lower = text.lower()
        return {term for term in medical_terms if term in text_lower}

