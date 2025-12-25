# Загрузка документов
import os
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader

os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

SITEMAP_URL = "https://antarcticwallet.com/sitemap.xml"
ROOT_URL = "https://antarcticwallet.com/"

# 1) Загружаем все страницы из sitemap
sitemap_loader = SitemapLoader(
    web_path=SITEMAP_URL,
    filter_urls=[ROOT_URL],  # на всякий случай ограничиваем доменом
)
sitemap_docs = sitemap_loader.load()

docs = sitemap_docs

print(f"Total documents: {len(docs)}")
print(f"Total characters: {sum(len(doc.page_content) for doc in docs)}")

# Total documents: 8
# Total characters: 43650
