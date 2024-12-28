import json
import html2text
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
import requests
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import diskcache as dc
import tempfile

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
)

# Set up diskcache using the OS temporary directory
cache_dir = os.path.join(tempfile.gettempdir(), "structured_data_cache")
cache = dc.Cache(cache_dir)

def get_page_html(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to retrieve the page. Status code: {response.status_code}"
        )
    return response.text

def convert_html_to_markdown(html_content):
    return html2text.html2text(html_content)

class Investor(BaseModel):
    name: str
    company_holdings: int
    net_worth: str
    url: HttpUrl
    pid: Optional[str] = None

class StructuredData(BaseModel):
    individual_investors: List[Investor]
    institutional_investors: List[Investor]
    fii_investors: List[Investor]
    last_updated: str

def get_structured_data(markdown_content):
    prompt = f"""
    Extract the top investor list by category from the following Markdown content:
    {markdown_content}

    The categories are:
    - Individual Investors
    - Institutional Investors
    - FII Investors

    Extract the following details for each investor:
    - Name (e.g., Rakesh Jhunjhunwala and Associates)
    - Company Holdings
    - Net Worth (e.g., 50,072)
    - URL (e.g., /india-investors-portfolio/rakesh-jhunjhunwala-and-associates)

    Ensure the URL is prefixed with 'https://www.moneycontrol.com' to form the complete URL.

    Return the data as a JSON object with the following structure:
    {{
        "individual_investors": [
            {{
                "name": "Investor Name",
                "company_holdings": 12345,
                "net_worth": "50,072",
                "url": "https://www.moneycontrol.com/india-investors-portfolio/investor-name"
            }},
            ...
        ],
        "institutional_investors": [
            {{
                "name": "Investor Name",
                "company_holdings": 12345,
                "net_worth": "50,072",
                "url": "https://www.moneycontrol.com/india-investors-portfolio/investor-name"
            }},
            ...
        ],
        "fii_investors": [
            {{
                "name": "Investor Name",
                "company_holdings": 12345,
                "net_worth": "50,072",
                "url": "https://www.moneycontrol.com/india-investors-portfolio/investor-name"
            }},
            ...
        ]
    }}
    """

    response = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that helps in extracting structured data from Markdown content. Always return the data as a JSON object.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    structured_data = response.choices[0].message.content.strip()

    # Add the current date and time to the structured data in a human-readable format
    structured_data_dict = json.loads(structured_data)
    structured_data_dict["last_updated"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    # Parse the JSON response into the Pydantic model
    return StructuredData.model_validate(structured_data_dict)

def scrape_pid_from_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to retrieve the page. Status code: {response.status_code}"
        )
    soup = BeautifulSoup(response.text, "html.parser")
    pid_element = soup.find("input", {"id": "pid"})
    return pid_element["value"] if pid_element else "No PID"

def get_portfolio_id(structured_data: StructuredData):
    for category in [
        "individual_investors",
        "institutional_investors",
        "fii_investors",
    ]:
        investors = getattr(structured_data, category, [])
        for investor in investors:
            url = investor.url
            title = scrape_pid_from_page(url)
            investor.pid = title
    return structured_data

def load_cache():
    structured_data_json = cache.get("structured_data")
    if structured_data_json:
        # Convert the JSON string back to a dictionary and then to a Pydantic model
        structured_data_dict = json.loads(structured_data_json)
        return StructuredData.model_validate(structured_data_dict)
    return None

def save_cache(structured_data: StructuredData):
    # Convert the Pydantic model to a dictionary and then to a JSON string
    structured_data_json = structured_data.model_dump_json()
    cache.set("structured_data", structured_data_json, expire=30*60)  # Cache for 30 minutes

def write_structured_data_to_file(structured_data):
    # Convert the Pydantic model to a JSON string
    structured_data_json = structured_data.model_dump_json(indent=4)
    
    # Define the data folder path
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    
    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Define the file path
    file_path = os.path.join(data_folder, 'investors.json')
    
    # Write the JSON string to the file
    with open(file_path, 'w') as f:
        f.write(structured_data_json)

def main():
    url = "https://www.moneycontrol.com/india-investors-portfolio/"

    print("Checking cache for structured data...")
    structured_data = load_cache()
    if not structured_data:
        print("Fetching HTML content from the URL...")
        html_content = get_page_html(url)

        print("Converting HTML content to Markdown...")
        markdown_content = convert_html_to_markdown(html_content)

        print("Extracting structured data from Markdown content...")
        structured_data = get_structured_data(markdown_content)

        print("Saving structured data to cache...")
        save_cache(structured_data)
    else:
        print("Using cached structured data.")

    print("Retrieving portfolio IDs...")
    structured_data = get_portfolio_id(structured_data)

    print("Saving investors json to file...")
    # Write the structured data JSON to a file in the data folder
    write_structured_data_to_file(structured_data)

if __name__ == "__main__":
    main()