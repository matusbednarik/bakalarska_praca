import scrapy
from bs4 import BeautifulSoup
import logging
import os
import json

class CsoonlineSpider(scrapy.Spider):
    name = "cybersecurity_terms"
    allowed_domains = ["www.csoonline.com"]
    start_urls = [
        "https://www.csoonline.com/cybercrime/filter/feature/",
        "https://www.csoonline.com/security/",
        "https://www.csoonline.com/privacy/"
    ]
    
    visited_urls = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load previously visited URLs from existing file if it exists
        self.output_file = 'crawledWebsites.json'
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # Add all existing URLs to visited_urls set
                    for item in existing_data:
                        if 'link' in item:
                            self.visited_urls.add(item['link'])
                    logging.info(f"Loaded {len(self.visited_urls)} previously visited URLs")
            except json.JSONDecodeError:
                logging.error(f"Error parsing existing {self.output_file}. Will create a new file.")
        else:
            logging.info(f"No existing {self.output_file} found or file is empty. Will create a new file.")

    custom_settings = {
        'DEPTH_LIMIT': 10,
        # Remove the FEEDS setting as we'll handle writing to the file manually
        'ROBOTSTXT_OBEY': False,
        'AUTOTHROTTLE_ENABLED': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 1,
        'COOKIES_ENABLED': False,
    }

    def parse(self, response):
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)

        articles = response.css('div.content-listing-various__container div.content-listing-various__row')
        
        for article in articles:
            link = article.css('a.grid.content-row-article::attr(href)').get()
            title = article.css('h3.card__title::text').get()
            description = article.css('p.card__description::text').get()
            
            if link and title and description:
                url = response.urljoin(link)
                if url not in self.visited_urls:
                    print(f"Following article: {title.strip()}")  # Add more visible output
                    yield response.follow(
                        url,
                        callback=self.parse_details,
                        meta={
                            'link': url,
                            'title': title.strip(),
                            'description': description.strip()
                        },
                        # Add this line to fix depth tracking
                        priority=1  # Higher priority than pagination links
                    )

        next_page = response.css('nav.pagination a.next.pagination__link::attr(href)').get()
        if next_page:
            logging.info(f"Following next page: {next_page}")
            yield response.follow(next_page, callback=self.parse)

    def parse_details(self, response):
        title = response.meta.get('title', 'Unknown Title')
        print(f"🔍 PROCESSING ARTICLE: \"{title}\"")
        print(f"   URL: {response.url}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = []
        
        content_containers = soup.select('div.article-content, div.content-well__main, article')
        
        if content_containers:
            for container in content_containers:
                # Extract headers and subheaders
                headers = container.find_all(['h1', 'h2', 'h3', 'h4'])
                for header in headers:
                    article_content.append({
                        'type': 'header',
                        'text': header.get_text(strip=True)
                    })
                
                # Extract paragraphs with more context
                paragraphs = container.find_all(['p', 'li'])
                for p in paragraphs:
                    # Get the previous header for context
                    prev_header = p.find_previous(['h1', 'h2', 'h3', 'h4'])
                    prev_header_text = prev_header.get_text(strip=True) if prev_header else None
                    
                    article_content.append({
                        'type': 'paragraph' if p.name == 'p' else 'list-item',
                        'text': p.get_text(strip=True),
                        'section_header': prev_header_text
                    })

        if article_content:
            logging.info(f"Found {len(article_content)} content blocks")
            # Append the new item to the JSON file
            self.append_to_json({
                'link': response.meta['link'],
                'title': response.meta['title'],
                'description': response.meta['description'],
                'article_content': article_content
            })
        else:
            logging.warning(f"No matching content found for {response.url}")

    def append_to_json(self, item):
        existing_data = []
        
        # Load existing data if the file exists
        if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Error reading {self.output_file}. Creating a new file.")
        
        # Log article details being added
        title = item.get('title', 'Unknown Title')
        link = item.get('link', 'Unknown URL')
        content_count = len(item.get('article_content', []))
        
        print(f"📄 ADDING ARTICLE: \"{title}\"")
        print(f"   URL: {link}")
        print(f"   Content blocks: {content_count}")
        print(f"   ------------------------------------------------")
        
        # Append the new item
        existing_data.append(item)
        
        # Write back to the file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Appended article \"{title}\" to {self.output_file}. Total articles: {len(existing_data)}")
        print(f"✅ WROTE ARTICLE: \"{item['title']}\"")

    def get_tag_type(self, tag):
        tag_type_map = {
            'p': 'paragraph',
            'h2': 'header',
            'h3': 'subheader',
            'li': 'list-item'
        }
        return tag_type_map.get(tag, 'text')

    def closed(self, reason):
        """Log when the spider is closed"""
        logging.info(f"Spider closed: {reason}. Total URLs visited: {len(self.visited_urls)}")