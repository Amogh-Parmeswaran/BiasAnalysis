import pandas as pd

# Load the CSV file
data = pd.read_csv('data.csv')

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

# Group the articles by publication and count the number of articles for each publication
for publication, articles in data.groupby("publication"):
    # Filter the articles by character count
    filtered_articles = articles[articles["article"].str.len() > CHAR_THRESHOLD]
    # Save the filtered articles for this publication as a separate csv file
    filtered_articles.to_csv(f"{publication}.csv", index=False)



