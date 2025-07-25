# Custom middleware for MediMaven scrapers

from scrapy import signals

# for handling scraped items
from itemadapter import is_item, ItemAdapter


class MediScraperSpiderMiddleware:
    # Basic spider middleware - only implementing what we need

    @classmethod
    def from_crawler(cls, crawler):
        # Connect spider signals
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Process incoming responses
        return None

    def process_spider_output(self, response, result, spider):
        # Pass through results from spider
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Handle spider exceptions
        pass

    def process_start_requests(self, start_requests, spider):
        # Process initial requests
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class MediScraperDownloaderMiddleware:
    # Downloader middleware for request/response handling

    @classmethod
    def from_crawler(cls, crawler):
        # Set up downloader middleware
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Process outgoing requests
        return None

    def process_response(self, request, response, spider):
        # Process incoming responses
        return response

    def process_exception(self, request, exception, spider):
        # Handle download exceptions
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
