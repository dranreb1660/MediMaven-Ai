from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from medi_scraper.models import MedicalItem


class PydanticValidationPipeline:
    def process_item(self, item, spider):
        try:
            validated = MedicalItem(**ItemAdapter(item).asdict())
            return validated.model_dump(mode="json")
        except Exception as exc:
            spider.logger.warning(f"Validation failed ‑ {exc} ({item.get('url')})")
            raise DropItem("Pydantic validation error")