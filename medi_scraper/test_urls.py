urls = [
    # Mayo Clinic
    "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
    # NHS
    "https://www.nhs.uk/conditions/diabetes/",
    # Cleveland Clinic
    "https://my.clevelandclinic.org/health/diseases/7104-diabetes-mellitus-an-overview",
    # WebMD
    "https://www.webmd.com/diabetes/diabetes-types-insulin"
]

import scrapy
from scrapy.crawler import CrawlerProcess
from medi_scraper.spiders.medical_spider import MedicalSpider

class TestSpider(MedicalSpider):
    name = 'test_medical_crawler'
    
    def start_requests(self):
        for url in urls:
            yield self.make_splash_request(url, callback=self.parse_medical_page, dont_filter=True)

# Run the test spider
if __name__ == '__main__':
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (compatible; TestMediScraper/1.0)',
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': False,
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 3,
        'COOKIES_ENABLED': False,
        'HTTPCACHE_ENABLED': False,
    })
    
    process.crawl(TestSpider)
    process.start()
