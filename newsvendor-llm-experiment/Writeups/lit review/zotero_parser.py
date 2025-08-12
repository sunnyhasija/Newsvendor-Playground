#!/usr/bin/env python3
"""
Zotero Collection Report HTML Parser

Extracts bibliographic information from a Zotero collection report HTML file
and creates a JSON file with the specified structure.
"""

from bs4 import BeautifulSoup
import json
import re
from typing import List, Dict


def extract_last_name(full_name: str) -> str:
    """Extract last name from full name."""
    parts = full_name.strip().split()
    if parts:
        return parts[-1]  # Return last part as surname
    return full_name


def format_authors(author_list: List[str]) -> str:
    """Format authors according to specification: last names only, et al for >2."""
    if not author_list:
        return ""
    
    last_names = [extract_last_name(author) for author in author_list if author.strip()]
    
    if len(last_names) == 1:
        return last_names[0]
    elif len(last_names) == 2:
        return f"{last_names[0]} and {last_names[1]}"
    elif len(last_names) > 2:
        return f"{last_names[0]} et al"
    
    return ""


def extract_year_from_date(date_text: str) -> str:
    """Extract 4-digit year from various date formats."""
    if not date_text:
        return ""
    
    # Look for 4-digit year pattern
    year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
    if year_match:
        return year_match.group()
    
    return ""


def clean_text(text: str) -> str:
    """Clean text by removing line breaks and extra whitespace."""
    if not text:
        return ""
    
    # Remove line breaks and replace with spaces
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def parse_article(article_li) -> Dict[str, str]:
    """Parse individual article HTML element to extract required fields."""
    
    article_data = {
        'title': '',
        'author(s)': '',
        'year': '',
        'journal': '',
        'abstract': ''
    }
    
    # Extract title from h2 tag
    title_element = article_li.find('h2')
    if title_element:
        article_data['title'] = clean_text(title_element.get_text(strip=True))
    
    # Find the table with article details
    table = article_li.find('table')
    if not table:
        return article_data
    
    # Extract authors
    authors = []
    author_elements = table.find_all('th', class_='author')
    for author_th in author_elements:
        author_td = author_th.find_next_sibling('td')
        if author_td:
            author_name = clean_text(author_td.get_text(strip=True))
            if author_name:
                authors.append(author_name)
    
    article_data['author(s)'] = format_authors(authors)
    
    # Extract other fields by looking for th elements with specific text
    rows = table.find_all('tr')
    for row in rows:
        th = row.find('th')
        td = row.find('td')
        
        if not th or not td:
            continue
            
        field_name = th.get_text(strip=True)
        field_value = clean_text(td.get_text(strip=True))
        
        if field_name == 'Date':
            article_data['year'] = extract_year_from_date(field_value)
        
        elif field_name in ['Publication', 'Proceedings Title', 'Publisher']:
            if not article_data['journal']:  # Use first one found
                article_data['journal'] = field_value
        
        elif field_name == 'Abstract':
            article_data['abstract'] = field_value
    
    return article_data


def parse_zotero_html_report(html_path: str = "Zotero Report 1.html") -> List[Dict[str, str]]:
    """
    Main function to parse Zotero collection HTML report.
    
    Args:
        html_path: Path to the Zotero report HTML file
        
    Returns:
        List of parsed articles in the specified format
    """
    print(f"Processing HTML: {html_path}")
    
    # Read and parse HTML
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all article items
    article_items = soup.find_all('li', class_='item')
    print(f"Found {len(article_items)} articles")
    
    # Parse each article
    parsed_articles = []
    for i, article_li in enumerate(article_items):
        try:
            article_data = parse_article(article_li)
            # Only add if we have at least a title
            if article_data['title']:
                parsed_articles.append(article_data)
                print(f"Parsed: {article_data['title'][:60]}...")
            else:
                print(f"Skipped article {i+1}: No title found")
        except Exception as e:
            print(f"Error parsing article {i+1}: {e}")
    
    return parsed_articles


def main():
    """Main execution function."""
    html_path = "Zotero Report.html"
    output_path = "zotero_literature.json"
    
    try:
        # Parse the report
        articles = parse_zotero_html_report(html_path)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully parsed {len(articles)} articles")
        print(f"Results saved to: {output_path}")
        
        # Show brief summary
        if articles:
            print(f"\nParsed articles:")
            for i, article in enumerate(articles, 1):
                print(f"{i:2d}. {article['title'][:60]}..." if len(article['title']) > 60 else f"{i:2d}. {article['title']}")
                
    except FileNotFoundError:
        print(f"Error: Could not find '{html_path}'")
        print("Make sure you've saved your Zotero report as HTML in the same directory.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
