import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import os
import json

# Note: need to run make setup before using this script.

save_dir ="Data/jsons"
filename = "article.json"

file_path = os.path.join(save_dir, filename)
os.makedirs(save_dir, exist_ok=True)

# Scrapes a wikipedia article into a JSON file containing the article text and See Also links. 
# Input is a string containing the url of the wikipedia article.
# Returns a dictionary representation of the JSON file (or None if the scraper fails)
def scrape_wikipedia_article(url):
    # Send HTTP request to wikipedia
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch the wikipedia article: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Process and clean article
    article_text = []
    
    for element in soup.select(".mw-parser-output h2, .mw-parser-output h3, .mw-parser-output p"):
        if element.name in ["h2", "h3"]: # Heading detected
            heading = element.get_text(strip=True).replace("[edit]", "") # Remove [edit] links
            article_text.append({"heading": heading, "paragraphs": []})
        elif element.name == "p" and article_text: 
            text = element.get_text(" ", strip=True) # Keep spaces while extracting text
            text = re.sub(r"\[\d+\]", "", text) #Remove citation numbers like [1]
            text = re.sub(r"\s+", " ", text) # Remove extra spaces
            if text:
                article_text[-1]["paragraphs"].append(text) # Paragraphs belong to the last heading
        
    # Remove sections with no data
    article_text = [section for section in article_text if section["paragraphs"]]

    # Extract see_also links
    see_also_links = []
    see_also_section = soup.find(id='See_also')

    if see_also_section:
        ul = see_also_section.find_next('ul')
        if ul:
            for li in ul.find_all('li'):
                a = li.find('a', href=True)
                if a:
                    full_link = urljoin(url, a['href'])        
                    see_also_links.append((a.get_text(strip=True), full_link))
            
    article_data = {
        "article_text": article_text,
        "see_also_link": see_also_links
    }

    # Stores the scraped data in a created JSON
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(article_data, json_file, indent=4, ensure_ascii=False)
    
    return article_data

# Scrapes a wikipedia article. Prints the text and see also links.
def print_wikipedia_article(url):
    # Send HTTP request to wikipedia
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch the wikipedia article: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Process and clean article
    
    for element in soup.select(".mw-parser-output h2, .mw-parser-output h3, .mw-parser-output p"):
        if element.name in ["h2", "h3"]: # Heading detected
            heading = element.get_text(strip=True).replace("[edit]", "") # Remove [edit] links
            print(heading)
        elif element.name == "p": 
            text = element.get_text(" ", strip=True) # Keep spaces while extracting text
            text = re.sub(r"\[\d+\]", "", text) #Remove citation numbers like [1]
            text = re.sub(r"\s+", " ", text) # Remove extra spaces
            if text:
                print(text)

    # Extract see_also links
    see_also_links = []
    see_also_section = soup.find(id='See_also')

    if see_also_section:
        ul = see_also_section.find_next('ul')
        if ul:
            for li in ul.find_all('li'):
                a = li.find('a', href=True)
                if a:
                    full_link = urljoin(url, a['href'])     
                    print((a.get_text(strip=True), full_link))   

scrape_wikipedia_article("https://en.wikipedia.org/wiki/Data_science")
print_wikipedia_article("https://en.wikipedia.org/wiki/Data_science")