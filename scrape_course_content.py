# scrape_course_content.py
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os
import json
import time

def scrape_html(url):
    """Scrapes text content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Prioritize relevant content areas, exclude navigation, headers, footers
        # You'll need to inspect the actual HTML of the IITM and s-anand.net pages
        # to refine these selectors.
        content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td', 'blockquote']
        texts = []
        for tag in content_tags:
            for element in soup.find_all(tag):
                text = element.get_text(separator=' ', strip=True)
                if text:
                    texts.append(text)

        full_text = ' '.join(texts)
        # Remove excessive whitespace
        full_text = ' '.join(full_text.split())
        return full_text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a local PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def download_pdf(url, filename):
    """Downloads a PDF from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {url}: {e}")
        return False

def get_course_content():
    """Gathers text content from predefined course URLs and PDFs."""
    course_data = []

    # --- Web pages to scrape ---
    course_urls = [
        "https://study.iitm.ac.in/ds/course_pages/BSSE2002.html",
        "https://tds.s-anand.net/",
        # Add more specific lecture/module URLs if available and publicly accessible
        # Example: if individual module pages exist, list them here.
        # "https://tds.s-anand.net/module1.html",
    ]

    for url in course_urls:
        print(f"Scraping: {url}")
        content = scrape_html(url)
        if content:
            course_data.append({
                'text': content,
                'source_url': url,
                'type': 'web_page',
                'title': url.split('/')[-1].replace('.html', '').replace('-', ' ').title()
            })
        time.sleep(1) # Be polite, avoid hammering the server

    # --- PDFs to download and extract ---
    # This is highly dependent on finding public PDF links.
    # You'll need to manually identify these by Browse the course websites.
    pdf_urls = {
        # "example_syllabus.pdf": "https://study.iitm.ac.in/ds/assets/syllabus.pdf",
        # "example_notes.pdf": "https://tds.s-anand.net/lecture_notes_week1.pdf"
    }

    pdf_dir = os.path.join('data', 'pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    for filename, url in pdf_urls.items():
        pdf_path = os.path.join(pdf_dir, filename)
        if download_pdf(url, pdf_path):
            print(f"Extracting text from: {filename}")
            pdf_text = extract_text_from_pdf(pdf_path)
            if pdf_text:
                course_data.append({
                    'text': pdf_text,
                    'source_url': url,
                    'type': 'pdf',
                    'title': filename.replace('.pdf', '').replace('_', ' ').title()
                })
        time.sleep(1) # Be polite

    return course_data

if __name__ == "__main__":
    print("Starting course content scraping...")
    all_course_content = get_course_content()
    output_path = os.path.join('data', 'course_content_raw.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_course_content, f, indent=2, ensure_ascii=False)
    print(f"Scraped {len(all_course_content)} course content items.")
    print(f"Raw course content saved to {output_path}")