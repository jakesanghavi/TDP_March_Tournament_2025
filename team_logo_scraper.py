from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Install the webdriver
driver = webdriver.Chrome(
            service=Service(executable_path=ChromeDriverManager().install()),
        )

# Open up the ESPN webpage of all D1 men's teams
driver.get("https://www.espn.com/mens-college-basketball/teams")

# Initialize a df to store our results
data = []

# We have to scroll down bc it's a dynamic webpage so start off with prev height of 0
previous_height = 0

# Dangerousto do while True but whatever
while True:
    # Find image elements w the class name for logos
    images = driver.find_elements(By.CLASS_NAME, "Image.Logo.Logo__lg")
    
    # Get the metadata
    for img in images:
        alt_text = img.get_attribute("alt")
        src = img.get_attribute("src")
        if alt_text and src:
            data.append((alt_text, src))
    
    # Scroll down to get more images
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    
    # Sleep to fight against website load issues
    time.sleep(2)
    
    # Check if scrolling has reached the bottom of the page and if so break
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == previous_height:
        break
        
    # Otherwise re-iterate
    previous_height = new_height

# Convert list to df
df = pd.DataFrame(data, columns=["Image Alt", "Image Src"])

# save output to CSV
df.to_csv("logos.csv", index=False)

# Close the driver
driver.quit()
