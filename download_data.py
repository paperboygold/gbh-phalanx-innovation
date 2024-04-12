import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import time
import os
import logging
import concurrent.futures
import re
import zipfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the reports
base_url = "https://nemweb.com.au"

# Ensure local data directory exists
data_directory = './data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Directories to download from

directories = [
    "PredispatchIS_Reports", 
    "DispatchIS_Reports", 
    "TradingIS_Reports"
    #"Regional_Summary_Report",
    #"Next_Day_PreDispatch", 
    #"Market_Notice",
    #"Next_Day_Offer_Energy", 
    #"Operational_Demand/ACTUAL_HH",
    #"Operational_Demand/ACTUAL_5MIN"
]



# Setup session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount('https://', adapter)
session.mount('http://', adapter)

def fetch_file_links(directory, base_url="https://nemweb.com.au", recursive=False):
    url = f"{base_url}/Reports/Current/{directory}/"
    response = session.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to access {url} with status code {response.status_code}")
        return [], 0
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    old_files_count = 0
    pre_tag = soup.find('pre')
    if pre_tag:
        for a in pre_tag.find_all('a', href=True):
            href = a['href'].strip('/')
            if href.endswith('/') and recursive:
                subdirectory = href[:-1]
                sub_links, sub_count = fetch_file_links(os.path.join(directory, subdirectory), base_url, True)
                links.extend(sub_links)
                old_files_count += sub_count
            elif href.endswith('.zip') or re.match(r'RSR_\d{8}\.R\d{3}', href) or re.match(r'.*\.R\d+$', href):
                full_url = f"{base_url}/{href}"
                links.append(full_url)
                # Regex to find date in various formats
                date_match = re.search(r'(\d{8})', href)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        comparison_date = datetime.now() - timedelta(days=1)
                        if file_date < comparison_date:
                            old_files_count += 1
                    except ValueError as e:
                        logging.error(f"Error parsing date from filename: {href}. Error: {e}")
                else:
                    logging.debug(f"No date found by regex in filename: {href}")
    else:
        logging.debug("No <pre> tag found, unable to parse directory or file links.")
    logging.info(f"Fetched {len(links)} links from {url}, {old_files_count} are older than the allowed threshold")
    return links, old_files_count

def is_old_link(link):
    date_match = re.search(r'(\d{8})', link)
    if date_match:
        date_str = date_match.group(1)
        file_date = datetime.strptime(date_str, '%Y%m%d')
        comparison_date = datetime.now() - timedelta(days=30)
        return file_date < comparison_date
    return False

def download_file(link, base_directory, attempt=1):
    max_attempts = 5
    backoff_factor = 2  # Exponential backoff factor
    logging.debug(f"Attempting to download file: {link}")
    try:
        response = session.get(link)
        logging.debug(f"HTTP response code: {response.status_code}")
        if response.status_code == 200:
            directory_name = link.split('/')[-2]  # Assuming the directory name is part of the link
            directory_path = os.path.join(base_directory, directory_name)
            os.makedirs(directory_path, exist_ok=True)  # Adjusted to handle existing directories
            file_path = os.path.join(directory_path, os.path.basename(link))
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.debug(f"Downloaded {file_path}")
            
            # Unzip the file if it is a zip file
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(directory_path)
                logging.debug(f"Unzipped {file_path} into {directory_path}")
                
                # Delete the original zip file
                os.remove(file_path)
                logging.debug(f"Deleted original zip file {file_path}")
            return (True, link)
                
        elif response.status_code == 403:
            logging.debug(f"403 Forbidden error for URL: {link}")  # Log only in debug mode
            if attempt < max_attempts:
                sleep_time = backoff_factor ** attempt
                time.sleep(sleep_time)
                return download_file(link, base_directory, attempt + 1)
            else:
                return (False, link)
        else:
            logging.error(f"Failed to download {link} with status code {response.status_code}")
            if attempt < max_attempts:
                sleep_time = backoff_factor ** attempt
                time.sleep(sleep_time)
                return download_file(link, base_directory, attempt + 1)
            else:
                return (False, link)
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        logging.warning(f"Attempt {attempt} failed for {link}. Error: {e}")
        if attempt < max_attempts:
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)
            return download_file(link, base_directory, attempt + 1)
        else:
            logging.error(f"Failed to download {link} after {max_attempts} attempts.")
            return (False, link)

def download_files(links, base_directory):
    failed_downloads = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Wrap the links with tqdm for a progress bar
        results = list(tqdm(executor.map(lambda link: download_file(link, base_directory), links), total=len(links), desc="Downloading files"))
        for success, link in results:
            if not success:
                failed_downloads.append(link)
    if failed_downloads:
        logging.error(f"Failed to download the following files after maximum attempts: {failed_downloads}")
    return failed_downloads

# Main execution loop
for directory in directories:
    links, old_files_count = fetch_file_links(directory, base_url, recursive=True)
    recent_links = [link for link in links if not is_old_link(link)]
    failed_downloads = download_files(recent_links, data_directory)
    print(f"Directory: {directory}, Files older than threshold: {old_files_count}, Failed Downloads: {failed_downloads}")