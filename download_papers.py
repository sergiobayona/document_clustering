import requests
import random
import os
import time
from bs4 import BeautifulSoup
import re
import argparse
import json
from urllib.parse import urljoin

class PMCDownloader:
    def __init__(self, output_dir="pmc_documents", delay=1, history_file="download_history.json"):
        """
        Initialize the PMC downloader.
        
        Args:
            output_dir (str): Directory to save downloaded documents
            delay (float): Delay between requests in seconds
            history_file (str): File to track downloaded documents
        """
        self.base_url = "https://pmc.ncbi.nlm.nih.gov"
        self.output_dir = output_dir
        self.delay = delay
        self.history_file = history_file
        self.downloaded_pmcids = self.load_history()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # User agent to mimic a browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
    
    def load_history(self):
        """
        Load download history from file and rebuild from existing files if needed.
        
        Returns:
            set: Set of PMC IDs that have been downloaded
        """
        pmcids = set()
        
        # First try to load from history file
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    pmcids = set(history.get("pmcids", []))
                print(f"Loaded {len(pmcids)} PMC IDs from history file")
            except Exception as e:
                print(f"Error loading history file: {e}")
        
        # Then scan the output directory to rebuild history from existing files
        if os.path.exists(self.output_dir):
            print("Scanning output directory to rebuild history...")
            file_count = 0
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.txt'):
                    # Extract PMC ID from filename (format: PMC123456_Title.txt)
                    pmc_match = re.match(r'(PMC\d+)_', filename)
                    if pmc_match:
                        pmcids.add(pmc_match.group(1))
                        file_count += 1
                    else:
                        # Try to extract PMC ID from file content if not in filename
                        try:
                            filepath = os.path.join(self.output_dir, filename)
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(1000)  # Read just the beginning
                                pmc_match = re.search(r'PMC ID: (PMC\d+)', content)
                                if pmc_match:
                                    pmcids.add(pmc_match.group(1))
                                    file_count += 1
                        except Exception as e:
                            print(f"Error reading file {filename}: {e}")
            
            print(f"Found {file_count} files in output directory")
            print(f"Total unique PMC IDs after rebuilding: {len(pmcids)}")
            
            # Save the rebuilt history
            try:
                with open(self.history_file, 'w') as f:
                    json.dump({"pmcids": list(pmcids)}, f)
                print(f"Saved rebuilt history to {self.history_file}")
            except Exception as e:
                print(f"Error saving rebuilt history: {e}")
        
        return pmcids
    
    def save_history(self):
        """
        Save download history to file.
        """
        try:
            with open(self.history_file, 'w') as f:
                json.dump({"pmcids": list(self.downloaded_pmcids)}, f)
        except Exception as e:
            print(f"Error saving history file: {e}")
    
    def get_random_pmcid(self):
        """
        Get a random PMC ID by accessing the advanced search page and picking a random article.
        """
        # Updated URL for PubMed search
        search_url = "https://pubmed.ncbi.nlm.nih.gov/advanced/"
        
        try:
            # Instead of using the advanced search page directly, use the search API
            # with a very general search term
            page_num = random.randint(1, 100)
            search_term = random.choice(["cell", "protein", "gene", "cancer", "disease", "treatment", "medicine", "biology"])
            search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={search_term}&page={page_num}"
            
            response = requests.get(search_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to access search results: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article links - updated selector for PubMed's current HTML structure
            article_links = soup.select("a.docsum-title")
            
            if not article_links:
                print("No article links found. Trying alternative approach...")
                # Try PMC-specific search as a fallback
                pmc_search_url = f"https://www.ncbi.nlm.nih.gov/pmc/?term={search_term}&page={page_num}"
                response = requests.get(pmc_search_url, headers=self.headers)
                if response.status_code != 200:
                    print(f"Failed to access PMC search results: {response.status_code}")
                    return None
                
                soup = BeautifulSoup(response.text, 'html.parser')
                article_links = soup.select("div.rslt a.title")
            
            if not article_links:
                print("No article links found in either search")
                return None
            
            # Try up to 10 random articles to find one not in history
            for _ in range(10):
                # Pick a random article link
                random_article = random.choice(article_links)
                article_url = random_article.get('href')
                
                # If the URL is relative, make it absolute
                if article_url and not article_url.startswith('http'):
                    if article_url.startswith('/'):
                        article_url = urljoin("https://www.ncbi.nlm.nih.gov", article_url)
                    else:
                        article_url = urljoin(search_url, article_url)
                
                # Extract PMC ID from the URL
                pmcid_match = re.search(r'PMC(\d+)', article_url)
                if pmcid_match:
                    pmcid = pmcid_match.group(0)
                    if pmcid not in self.downloaded_pmcids:
                        return pmcid
                    else:
                        print(f"Skipping already downloaded PMC ID: {pmcid}")
                        continue
                else:
                    # If we can't find a PMC ID in the URL, check if it's a PubMed ID and try to convert
                    pubmed_match = re.search(r'/(\d+)/?$', article_url)
                    if pubmed_match:
                        pubmed_id = pubmed_match.group(1)
                        # Try to find PMC ID in the article page
                        article_response = requests.get(f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}", headers=self.headers)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            pmc_link = article_soup.select_one("a[href*='PMC']")
                            if pmc_link:
                                pmc_match = re.search(r'PMC(\d+)', pmc_link.get('href', ''))
                                if pmc_match:
                                    pmcid = pmc_match.group(0)
                                    if pmcid not in self.downloaded_pmcids:
                                        return pmcid
                                    else:
                                        print(f"Skipping already downloaded PMC ID: {pmcid}")
                                        continue
            
            print("Could not find a new PMC ID that hasn't been downloaded before")
            return None
                
        except Exception as e:
            print(f"Error in get_random_pmcid: {e}")
            return None
    
    def download_document(self, pmcid):
        """
        Download a document by its PMC ID.
        
        Args:
            pmcid (str): PMC ID of the document to download
        
        Returns:
            bool: True if download was successful, False otherwise
        """
        article_url = f"{self.base_url}/articles/{pmcid}/"
        
        try:
            response = requests.get(article_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to download {pmcid}: {response.status_code}")
                return False
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title - try different possible selectors
            title_elem = soup.select_one("h1.content-title, h1.title")
            title = title_elem.text.strip() if title_elem else "Unknown Title"
            
            # Extract abstract - try different possible selectors
            abstract_elem = soup.select_one("div.abstract, .abstract, section.abstract")
            abstract = abstract_elem.text.strip() if abstract_elem else "No abstract available"
            
            # Extract main content - try different possible selectors
            main_content = soup.select_one("div.jig-ncbiinpagenav, article.main-content, main")
            
            if not main_content:
                # Try to find the main content area using a more general approach
                # Look for the largest text block after the abstract
                possible_content_divs = soup.select("div.sec, section.sec, div.body, section.body")
                if possible_content_divs:
                    content = "\n\n".join([div.text.strip() for div in possible_content_divs])
                else:
                    # Fallback: just get everything in the body except headers and footers
                    body = soup.select_one("body")
                    if body:
                        # Try to exclude navigation and other irrelevant sections
                        for elem in body.select("header, footer, nav, script, style"):
                            if elem:
                                elem.decompose()
                        content = body.text.strip()
                    else:
                        content = "No content available"
            else:
                content = main_content.text.strip()
            
            # Create a clean filename
            filename = f"{pmcid}_{re.sub(r'[^\w\s-]', '', title)[:50]}.txt"
            filename = os.path.join(self.output_dir, filename)
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\n")
                f.write(f"PMC ID: {pmcid}\n\n")
                f.write(f"URL: {article_url}\n\n")
                f.write(f"Abstract:\n{abstract}\n\n")
                f.write(f"Content:\n{content}\n")
            
            # Add to downloaded list and save history
            self.downloaded_pmcids.add(pmcid)
            self.save_history()
            
            print(f"Successfully downloaded: {pmcid} - {title}")
            return True
            
        except Exception as e:
            print(f"Error downloading {pmcid}: {e}")
            return False
    
    def download_random_documents(self, count=10):
        """
        Download a specified number of random documents.
        
        Args:
            count (int): Number of documents to download
        """
        success_count = 0
        attempt_count = 0
        max_attempts = count * 5  # Allow for more failures due to duplicates
        
        print(f"Starting download of {count} documents. Already downloaded: {len(self.downloaded_pmcids)} documents.")
        
        while success_count < count and attempt_count < max_attempts:
            pmcid = self.get_random_pmcid()
            if pmcid:
                print(f"Attempting to download {pmcid}...")
                if self.download_document(pmcid):
                    success_count += 1
                    print(f"Progress: {success_count}/{count}")
                
                # Respect robots.txt with a delay between requests
                time.sleep(self.delay)
            
            attempt_count += 1
        
        print(f"Downloaded {success_count} documents (requested {count})")
        print(f"Total unique documents in history: {len(self.downloaded_pmcids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download random documents from PubMed Central")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of documents to download")
    parser.add_argument("-o", "--output", type=str, default="pmc_documents", help="Output directory")
    parser.add_argument("-d", "--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("-f", "--history", type=str, default="download_history.json", help="History file to track downloads")
    args = parser.parse_args()
    
    downloader = PMCDownloader(output_dir=args.output, delay=args.delay, history_file=args.history)
    downloader.download_random_documents(count=args.num)