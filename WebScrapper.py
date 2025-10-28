import requests
from bs4 import BeautifulSoup

# URL of the blog page
url = "https://naijasteed.blogspot.com/2025/09/art14.html"

# Fetch the page content
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request fail

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all image tags
images = soup.find_all('img')

# Extract the 'src' attribute from each image tag
image_urls = [img.get('src') for img in images if img.get('src')]

# Print all image URLs
for img_url in image_urls:
    print(img_url)