import pandas as pd
import string
import re

# Load the CSV file
data = pd.read_csv('news-dataset.csv')

# Define a set of publications
publications = {
    "Axios",
    "Business Insider",
    "Buzzfeed News",
    "CNBC",
    "CNN",
    "Economist",
    "Fox News",
    "Gizmodo",
    "Hyperallergic",
    "Mashable",
    "New Republic",
    "New Yorker",
    "People",
    "Politico",
    "Refinery 29",
    "Reuters",
    "TMZ",
    "TechCrunch",
    "The Hill",
    "The New York Times",
    "The Verge",
    "Vice",
    "Vice News",
    "Vox",
    "Washington Post",
    "Wired",
}

# Set the character count threshold
CHAR_THRESHOLD = 1000

# Clean text function
clean_text = lambda x: re.sub(r'[^\w\s]', '', str(x))
# Remove consecutive space function
remove_excess_spacing = lambda x: re.sub(' +', ' ', str(x))

print('pre-filter')

index = 0
# Group the articles by publication
for publication, articles in data.groupby("publication"):
    print(index, ': starting')
    print('before preliminary filtering')
    # Remove excess spaces
    articles["article"] = articles["article"].str.strip()
    # Remove capitalization
    articles["article"] = articles["article"].str.lower()
    # Replace the name of each publication with an empty string 
    articles["article"] = articles["article"].str.replace(publication.lower(), "")
    # Remove punctuation
    articles["article"] = articles["article"].apply(clean_text)
    # Remove excess spaces
    articles["article"] = articles["article"].apply(remove_excess_spacing)
    print('after preliminary filtering')
    # Filter the articles by character count
    filtered_articles = articles[(articles["article"].str.len() > CHAR_THRESHOLD) & 
                                 (articles["year"].astype(str).str.isnumeric())]
    # Save the filtered articles for this publication as a separate csv file
    print(index, ': filtered')
    filtered_articles.to_csv(f"{publication}.csv", index=False)
    
    index += 1

print('done with filtering')



