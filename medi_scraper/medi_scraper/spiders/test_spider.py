import scrapy
import logging
from urllib.parse import urlparse
import re
from w3lib.html import remove_tags
from scrapy_splash import SplashRequest

class MedicalSpider(scrapy.Spider):
    name = "medical_crawler_v2"
    allowed_domains = ["mayoclinic.org", "my.clevelandclinic.org", "webmd.com", "nhs.uk", "localhost"]
    
    custom_settings = {
        # Core settings
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "DOWNLOAD_DELAY": 0.5,  # Increased delay for politeness
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,  # Reduced for politeness
        "AUTOTHROTTLE_START_DELAY": 1.0,  # Start with conservative delay
        "AUTOTHROTTLE_MAX_DELAY": 5.0,  # Max delay for slow responses
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,  # Target concurrency
        "HTTPCACHE_ENABLED": True,
        
        # Middleware configuration
        "SPIDER_MIDDLEWARES": {
            'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
        },
        "DOWNLOADER_MIDDLEWARES": {
            'scrapy_splash.SplashCookiesMiddleware': 723,
            'scrapy_splash.SplashMiddleware': 725,
            'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
            'scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware': 100,
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,  # Disable default
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': 500,  # Add retry middleware
        },
        
        # Splash configuration
        "DUPEFILTER_CLASS": 'scrapy_splash.SplashAwareDupeFilter',
        "SPLASH_URL": 'http://localhost:8050',
        
        # User Agent - identify the bot properly
        "USER_AGENT": "MediScraper/1.0 (+https://medicalscraper.example.com) Research Bot",
        
        # Retry settings
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,  # Retry failed requests up to 3 times
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],  # Include 429 (too many requests)
        
        # Cache settings
        "HTTPCACHE_EXPIRATION_SECS": 86400,  # 24 hours
        "HTTPCACHE_IGNORE_HTTP_CODES": [500, 502, 503, 504, 400, 401, 403, 404, 408, 429],
        
        # Pipeline settings
        "ITEM_PIPELINES": {
            # 'medi_scraper.pipelines.MediScraperPipeline': 300,
        }
    }
    
    # Site-specific configurations
    SITE_CONFIGS = {
        'mayoclinic.org': {
            'index_selectors': {
                'a_z_links': 'a[href*="index?letter="]::attr(href), a[href*="diseases-conditions/index"]::attr(href)',
                'condition_links': ('a[href*="/diseases-conditions/"][href*="/symptoms-causes/"]::attr(href), ' +
                                  'a[href*="/diseases-conditions/"][href*="/diagnosis-treatment/"]::attr(href), ' +
                                  'a[href*="/diseases-conditions/"][href*="/doctors-departments/"]::attr(href), ' +
                                  'a[href*="/diseases-conditions/"][href*="/syc-"]::attr(href)'),
                'content_sections': 'div#main-content, div.content, article, .row, .content-row',
                'section_selectors': 'div.mayo-row, section, .content, .webnav, .pod, .panel, .content-block',
                'heading_selectors': 'h2::text, h3::text, .header::text',
                'content_selectors': '.content, .section-content, .row, p'
            }
        },
        'nhs.uk': {
            'index_selectors': {
                'a_z_links': 'a[href*="/conditions/"][href*="-to-"]::attr(href)',
                'condition_links': ('a[href*="/conditions/"]::attr(href), ' +
                                  '.nhsuk-list--menu a::attr(href), ' +
                                  '.nhsuk-card__link::attr(href), ' +
                                  '.nhsuk-grid-column a::attr(href), ' +
                                  '.condition-list-item a::attr(href)'),
                'sitemap_content': 'https://www.nhs.uk/sitemaps/sitemap-content.xml',
                'content_sections': 'main, article, .nhsuk-main-wrapper',
                'section_selectors': ('section, .block, .nhsuk-grid-column, ' + 
                                    '.nhsuk-card, .nhsuk-expander, .nhsuk-details, ' +
                                    '.nhsuk-warning-callout, .nhsuk-inset-text, .panel'),
                'heading_selectors': 'h2::text, h3::text, .nhsuk-heading-l::text, .nhsuk-heading-m::text',
                'content_selectors': 'p, .nhsuk-body, li'
            }
        }
    }

    def start_requests(self):
        # Create A-Z index URLs for Mayo Clinic
        mayo_az_urls = [f"https://www.mayoclinic.org/diseases-conditions/index?letter={letter}" 
                        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        
        urls = [
            # Mayo Clinic URLs - directory and popular conditions
            "https://www.mayoclinic.org/diseases-conditions",  # Main diseases directory
            "https://www.mayoclinic.org/diseases-conditions/index",  # A-Z index main page
            
            # NHS URLs - sitemaps and directories
            "https://www.nhs.uk/sitemap.xml",  # Sitemap index (will follow to sitemap-content.xml)
            "https://www.nhs.uk/sitemaps/sitemap-content.xml",  # Direct sitemap content URL
            "https://www.nhs.uk/conditions/",  # Main conditions directory 
            "https://www.nhs.uk/conditions/a-to-z/",  # A-Z index for conditions
            
            # Direct entry points to popular conditions for faster crawling
            # "https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118",
            # "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
            # "https://www.nhs.uk/conditions/diabetes/",
            # "https://www.nhs.uk/conditions/heart-disease/",
        ]
        
        # Add Mayo Clinic A-Z index URLs 
        urls.extend(mayo_az_urls[:3])  # Limit to first 3 letters for now (can be adjusted)
        for url in urls:
            try:
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    self.logger.error(f"Invalid URL format: {url}. Skipping.")
                    continue
                
                self.logger.info(f"Starting request for URL: {url}")
                # Configure Splash request properly with endpoint and full args
                yield SplashRequest(
                    url, 
                    callback=self.parse,
                    endpoint='render.html',
                    args={
                        'wait': 2,
                        'timeout': 90,
                        'images': 0,
                        'render_all': 1
                    },
                    dont_filter=True,
                    errback=self.errback_handler
                )
            except Exception as e:
                self.logger.error(f"Error starting request for URL {url}: {str(e)}")
    
    def errback_handler(self, failure):
        """Handle request errors"""
        # Get the request URL that caused the error
        request = failure.request
        self.logger.error(f"Request failed: {request.url}")
        self.logger.error(f"Error: {failure.value}")

    def parse(self, response):
        # Register XML namespaces
        namespaces = [
            ('sm', 'http://www.sitemaps.org/schemas/sitemap/0.9'),
            ('xhtml', 'http://www.w3.org/1999/xhtml')
        ]
        try:
            for prefix, uri in namespaces:
                response.selector.register_namespace(prefix, uri)
        except Exception as e:
            self.logger.error(f"Error registering XML namespace: {str(e)}")
        
        # Get domain for site-specific handling
        domain = urlparse(response.url).netloc
        
        # Determine page type based on URL and contents
        if 'sitemap' in response.url:
            self.logger.info(f"Processing sitemap: {response.url}")
            yield from self.parse_sitemap(response)
        elif 'index' in response.url or any(path in response.url for path in ['/a-to-z/', '/conditions/']):
            self.logger.info(f"Processing index page: {response.url}")
            yield from self.parse_index_page(response)
        else:
            # Page appears to be a medical condition page
            self.logger.info(f"Processing medical page: {response.url}")
            yield from self.parse_medical_page(response)
            
    def parse_index_page(self, response):
        """Parse index pages that list multiple medical conditions"""
        try:
            self.logger.info(f"Parsing index page: {response.url}")
            domain = urlparse(response.url).netloc
            
            # Get domain-specific selectors
            if "mayoclinic.org" in domain and domain in self.SITE_CONFIGS:
                selectors = self.SITE_CONFIGS[domain]['index_selectors']
                
                # Extract condition links
                condition_links = response.css(selectors['condition_links']).getall()
                self.logger.info(f"Found {len(condition_links)} Mayo Clinic condition links")
                
                # Process condition links
                for link in condition_links:
                    if link.startswith('/'):
                        link = f"https://www.mayoclinic.org{link}"
                    
                    self.logger.info(f"Following Mayo condition link: {link}")
                    yield SplashRequest(
                        link,
                        callback=self.parse_medical_page,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                
                # Look for A-Z index links
                az_links = response.css(selectors['a_z_links']).getall()
                for link in az_links:
                    if link.startswith('/'):
                        link = f"https://www.mayoclinic.org{link}"
                    yield SplashRequest(
                        link,
                        callback=self.parse_az_index,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                    
            elif "nhs.uk" in domain and domain in self.SITE_CONFIGS:
                selectors = self.SITE_CONFIGS[domain]['index_selectors']
                
                # Look for condition links in HTML format
                condition_links = response.css(selectors['condition_links']).getall()
                self.logger.info(f"Found {len(condition_links)} NHS condition links")
                
                # Process condition links
                for link in condition_links:
                    if link.startswith('/'):
                        link = f"https://www.nhs.uk{link}"
                    elif not link.startswith(('http://', 'https://')):
                        continue
                        
                    self.logger.info(f"Following NHS condition link: {link}")
                    yield SplashRequest(
                        link,
                        callback=self.parse_medical_page,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                
                # Extract A-Z condition links
                az_links = response.css(selectors['a_z_links']).getall()
                for link in az_links:
                    if link.startswith('/'):
                        link = f"https://www.nhs.uk{link}"
                    elif not link.startswith(('http://', 'https://')):
                        continue
                        
                    self.logger.info(f"Following NHS A-Z index: {link}")
                    yield SplashRequest(
                        link,
                        callback=self.parse_az_index,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
            else:
                # Generic handling for unknown domains
                condition_links = response.css('a[href*="condition"], a[href*="disease"], a[href*="symptom"]').getall()
                for link in condition_links:
                    # Process the link
                    if link.startswith('/'):
                        base_url = f"https://{domain}"
                        link = f"{base_url}{link}"
                    
                    if not link.startswith(('http://', 'https://')):
                        continue
                        
                    yield SplashRequest(
                        link,
                        callback=self.parse_medical_page,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
        except Exception as e:
            self.logger.error(f"Error in parse_index_page: {str(e)}")
    
    def parse_sitemap(self, response):
        """Parse XML sitemaps to find medical condition URLs"""
        try:
            self.logger.info(f"Parsing sitemap at URL: {response.url}")
            self.logger.info(f"Response type: {response.headers.get('Content-Type', b'unknown').decode()}")
            
            # Debug - print the first 1000 characters of the response body
            body_sample = response.body[:1000].decode('utf-8', errors='replace')
            self.logger.info(f"Response body sample: {body_sample}")
            
            # Check if this is a sitemap index (XML format)
            sitemap_urls = response.xpath('//sm:sitemap/sm:loc/text()').getall()
            if sitemap_urls:
                self.logger.info(f"Found sitemap index with {len(sitemap_urls)} sitemaps")
                # Process sitemap index
                for sitemap_url in sitemap_urls:
                    # Filter out non-medical sitemaps if possible
                    if any(keyword in sitemap_url.lower() for keyword in ['disease', 'condition', 'symptom', 'health', 'medical', 'content']):
                        self.logger.info(f"Following medical sitemap: {sitemap_url}")
                        yield SplashRequest(
                            sitemap_url,
                            callback=self.parse_sitemap,
                            endpoint='render.html',
                            args={
                                'wait': 2,
                                'timeout': 90,
                                'images': 0
                            },
                            dont_filter=True,
                            errback=self.errback_handler
                        )
            
            # Extract URLs from regular XML sitemap
            urls = response.xpath('//sm:url/sm:loc/text()').getall()
            if not urls:
                # Fallback to non-namespaced XPath if needed
                urls = response.xpath('//loc/text()').getall()
            
            # For HTML sitemaps (like mayoclinic.org or nhs.uk conditions pages)
            html_urls = []
            
            # Handle different types of pages based on domain and content
            
            # Mayo Clinic specific patterns
            if "mayoclinic.org" in response.url:
                self.logger.info("Processing Mayo Clinic page")
                # Look for disease links in HTML sitemap
                html_urls.extend(response.css(
                    'a[href*="diseases-conditions/"]::attr(href), ' +
                    'a[href*="symptoms/"]::attr(href), ' +
                    '.index a::attr(href), ' +
                    '.cardlist a::attr(href), ' +
                    '.az-list a::attr(href)'
                ).getall())
                
                # Extract direct disease/condition links
                condition_links = response.css(
                    'a[href*="/diseases-conditions/"][href*="/symptoms-causes/"]::attr(href), ' +
                    'a[href*="/diseases-conditions/"][href*="/diagnosis-treatment/"]::attr(href), ' +
                    'a[href*="/diseases-conditions/"][href*="/doctors-departments/"]::attr(href), ' +
                    'a[href*="/diseases-conditions/"][href*="/syc-"]::attr(href)'
                ).getall()
                
                self.logger.info(f"Found {len(condition_links)} direct condition links on Mayo Clinic page")
                
                for link in condition_links:
                    if link.startswith('/'):
                        link = f"https://www.mayoclinic.org{link}"
                    self.logger.info(f"Found direct condition link: {link}")
                    yield SplashRequest(
                        link,
                        callback=self.parse_medical_page,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                
                # Look for A-Z index links
                az_links = response.css('a[href*="index?letter="]::attr(href)').getall()
                for link in az_links:
                    self.logger.info(f"Found Mayo Clinic A-Z index link: {link}")
                    if link.startswith('/'):
                        link = f"https://www.mayoclinic.org{link}"
                    yield SplashRequest(
                        link,
                        callback=self.parse_az_index,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                
            # NHS specific patterns
            if "nhs.uk" in response.url:
                # Handle sitemap-content.xml specifically
                if "sitemap-content.xml" in response.url:
                    # This should contain direct URLs to condition pages
                    content_urls = response.xpath('//sm:url/sm:loc/text()').getall()
                    for url in content_urls:
                        if '/conditions/' in url:
                            self.logger.info(f"Found NHS condition URL: {url}")
                            html_urls.append(url)
                
                # Look for condition links in HTML format
                html_urls.extend(response.css(
                    'a[href*="/conditions/"]::attr(href), ' +
                    '.nhsuk-list--menu a::attr(href), ' +
                    '.nhsuk-card__link::attr(href), ' +
                    '.nhsuk-grid-column a::attr(href), ' +
                    '.condition-list-item a::attr(href)'
                ).getall())
                
                # Extract A-Z condition links
                az_links = response.css('a[href*="/conditions/"][href*="-to-"]::attr(href)').getall()
                for link in az_links:
                    if not link.startswith(('http://', 'https://')):
                        link = f"https://www.nhs.uk{link}"
                    self.logger.info(f"Found NHS A-Z index: {link}")
                    yield SplashRequest(
                        link,
                        callback=self.parse_az_index,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
            # Combine XML and HTML URLs
            all_urls = urls + html_urls
            self.logger.info(f"Found {len(all_urls)} URLs in sitemap (XML: {len(urls)}, HTML: {len(html_urls)})")
            
            # Medical condition URL patterns to match
            medical_patterns = [
                # Mayo Clinic patterns
                '/diseases-conditions/', 
                '/symptoms/',
                '/diagnosis-treatment/',
                # NHS patterns
                '/conditions/',
                '/symptoms/',
                # Generic patterns
                '/disorders/', 
                '/health/', 
                '/medicine/',
                'condition-',
                'disease-',
                'symptom-'
            ]
            
            # Filter and process medical URLs
            medical_urls = [url for url in all_urls if any(pattern in url.lower() for pattern in medical_patterns)]
            self.logger.info(f"Found {len(medical_urls)} medical URLs")
            
            # Debug - print all medical URLs found
            for url in medical_urls[:10]:  # Limit to first 10 to avoid too much output
                self.logger.info(f"Medical URL: {url}")
            
            # Process URLs
            for url in medical_urls:
                try:
                    # Make sure it's an absolute URL
                    if url.startswith('/'):
                        if 'mayoclinic.org' in response.url:
                            url = f"https://www.mayoclinic.org{url}"
                        elif 'nhs.uk' in response.url:
                            url = f"https://www.nhs.uk{url}"
                        else:
                            self.logger.warning(f"Could not determine domain for relative URL: {url}")
                            continue
                    
                    # Validate URL format
                    if not url.startswith(('http://', 'https://')):
                        self.logger.warning(f"Invalid URL format after processing: {url}. Skipping.")
                        continue
                            
                    self.logger.info(f"Processing medical URL: {url}")
                    yield SplashRequest(
                        url, 
                        callback=self.parse_medical_page,
                        endpoint='render.html',
                        args={
                            'wait': 2,
                            'timeout': 90,
                            'images': 0
                        },
                        dont_filter=True,
                        errback=self.errback_handler
                    )
                except Exception as e:
                    self.logger.error(f"Error processing URL {url}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in parse_sitemap: {str(e)}")
    
    def parse_az_index(self, response):
        """
        Parse A-Z disease index pages that contain multiple links to medical conditions
        """
        try:
            self.logger.info(f"Parsing A-Z index: {response.url}")
            
            # Extract disease/condition links based on domain
            condition_links = []
            
            # Mayo Clinic specific selectors
            if "mayoclinic.org" in response.url:
                condition_links = response.css(
                    '.index a::attr(href), ' +
                    'a[href*="diseases-conditions/"]::attr(href), ' +
                    'a[href*="symptoms/"]::attr(href), ' +
                    '.cardlist a::attr(href), ' + 
                    '.card a::attr(href), ' +
                    'li.ari__feature a::attr(href)'
                ).getall()
                self.logger.info(f"Found {len(condition_links)} Mayo Clinic condition links")
            
            # NHS specific selectors
            elif "nhs.uk" in response.url:
                condition_links = response.css(
                    '.nhsuk-list a::attr(href), ' +
                    '.nhsuk-list--menu a::attr(href), ' +
                    '.nhsuk-card__link::attr(href), ' +
                    '.nhsuk-grid-column a::attr(href), ' +
                    '.condition-list-item a::attr(href)'
                ).getall()
                self.logger.info(f"Found {len(condition_links)} NHS condition links")
            
            # Generic fallback
            if not condition_links:
                condition_links = response.css('a[href*="condition"]::attr(href), a[href*="disease"]::attr(href), a[href*="symptom"]::attr(href)').getall()
            
            self.logger.info(f"Found {len(condition_links)} condition links in A-Z index")
            
            for link in condition_links:
                # Handle relative URLs based on domain
                if link.startswith('/'):
                    if "mayoclinic.org" in response.url:
                        link = f"https://www.mayoclinic.org{link}"
                    elif "nhs.uk" in response.url:
                        link = f"https://www.nhs.uk{link}"
                    else:
                        # Skip if we can't determine domain
                        self.logger.warning(f"Skipping relative URL without known domain: {link}")
                        continue
                
                # Skip URLs that are not absolute after processing
                if not link.startswith(('http://', 'https://')):
                    self.logger.warning(f"Skipping invalid URL: {link}")
                    continue
                    
                self.logger.info(f"Following condition link: {link}")
                yield SplashRequest(
                    link,
                    callback=self.parse_medical_page,
                    endpoint='render.html',
                    args={
                        'wait': 2,
                        'timeout': 90,
                        'images': 0
                    },
                    dont_filter=True,
                    errback=self.errback_handler
                )
        except Exception as e:
            self.logger.error(f"Error in parse_az_index: {str(e)}")
        
    def parse_medical_page(self, response):
        try:
            self.logger.info(f"Parsing medical page: {response.url}")
            
            item = {
                "url": response.url,
                "domain": urlparse(response.url).netloc,
                "title": response.css('title::text').get(),
                "last_updated": response.css('meta[property="article:modified_time"]::attr(content)').get(),
                "medical_entities": [],
                "content_sections": []
            }

            # Structured content extraction
            if "mayoclinic.org" in item['domain']:
                # Extract main content
                main_content = response.css('div#main-content, div.content, article, .row, .content-row')
                
                # Extract headline/title
                title = response.css('h1.headTitle::text, h1::text, .heading-1::text').get()
                if title:
                    item['title'] = title.strip()
                
                # Extract sections
                sections = main_content.css('div.mayo-row, section, .content, .webnav, .pod, .panel, .content-block')
                self.logger.info(f"Found {len(sections)} content sections on Mayo Clinic page")
                
                for section in sections:
                    heading = section.css('h2::text, h3::text, .header::text').get()
                    content = self.clean_text(section.css('.content, .section-content, .row, p'))
                    
                    if heading and content:
                        item['content_sections'].append({
                            "heading": heading,
                            "body": content,
                            "type": "medical_content"
                        })
                        
                # Mayo Clinic specific entity extraction
                symptom_list = response.css('.symptom-list li, .symptoms-list li::text').getall()
                if symptom_list:
                    item['medical_entities'].extend(symptom_list)
            # NHS specific content extraction
            elif "nhs.uk" in item['domain']:
                # Extract main content for NHS pages
                main_content = response.css('main, article, .nhsuk-main-wrapper')
                
                # Extract headline/title
                title = response.css('h1.nhsuk-heading-xl::text, h1::text').get()
                if title:
                    item['title'] = title.strip()
                
                # Extract sections from NHS pages
                sections = main_content.css(
                    'section, .block, .nhsuk-grid-column, ' + 
                    '.nhsuk-card, .nhsuk-expander, .nhsuk-details, ' +
                    '.nhsuk-warning-callout, .nhsuk-inset-text, .panel'
                )
                self.logger.info(f"Found {len(sections)} content sections on NHS page")
                
                for section in sections:
                    heading = section.css('h2::text, h3::text, .nhsuk-heading-l::text, .nhsuk-heading-m::text').get()
                    content = self.clean_text(section.css('p, .nhsuk-body, li'))
                    
                    # If no content found with selective CSS, try getting all text
                    if not content:
                        content = self.clean_text(section)
                    
                    if heading and content:
                        item['content_sections'].append({
                            "heading": heading,
                            "body": content,
                            "type": "medical_content"
                        })
            
            # Generic entity extraction for all sites
            item['medical_entities'].extend(response.css(
                'div.diagnosis-list li::text, table.conditions td:first-child::text, ul.condition-list li::text'
            ).getall())
            
            # Quality checks
            if self.validate_item(item):
                self.logger.info(f"Successfully scraped medical content: {item['url']}")
                yield item
            else:
                self.logger.info(f"Item failed validation: {item['url']}")
        except Exception as e:
            self.logger.error(f"Error in parse_medical_page for {response.url}: {str(e)}")
            
    def validate_item(self, item):
        try:
            min_content_length = 300
            # Make the validation more lenient to capture more content
            required_entities = ['symptoms', 'causes', 'treatment', 'diagnosis', 'overview', 
                               'condition', 'disease', 'health', 'medical', 'doctor']
            
            # If no content sections, item is invalid
            if not item['content_sections']:
                self.logger.warning(f"No content sections found in {item['url']}")
                return False
                
            content = " ".join([s.get('body', '') for s in item['content_sections'] if s.get('body')])
            
            # Log validation details
            self.logger.info(f"Validating {item['url']}")
            self.logger.info(f"Title: {item.get('title', 'N/A')}")
            self.logger.info(f"Content sections: {len(item['content_sections'])}")
            self.logger.info(f"Content length: {len(content)} characters")
            found_entities = [entity for entity in required_entities if entity in content.lower()]
            self.logger.info(f"Found {len(found_entities)}/{len(required_entities)} required entities: {found_entities}")
            
            # Make validation more lenient by using only the content length requirement
            # unless we have lots of content, in which case also check for entities
            if len(content) >= min_content_length * 3:  # For very long content
                result = any(entity in content.lower() for entity in required_entities)
            else:
                result = len(content) >= min_content_length  # Just check length for shorter content
            
            self.logger.info(f"Validation result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error in validate_item: {str(e)}")
            return False

    def clean_text(self, selector):
        try:
            if not selector:
                return ""
                
            text = selector.css('::text').getall()
            cleaned = ' '.join([t.strip() for t in text if t.strip()])
            return re.sub(r'\s+', ' ', cleaned)
        except Exception as e:
            self.logger.error(f"Error in clean_text: {str(e)}")
            return ""