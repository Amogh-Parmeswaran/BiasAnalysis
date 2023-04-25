import pandas as pd

# Load the CSV file
data = pd.read_csv('data.csv')

# Filter the data based on the publication column
Axios_data = data[data['publication'] == 'Axios']
Business_Insider_data = data[data['publication'] == 'Business Insider']
Buzzfeed_News_data = data[data['publication'] == 'Buzzfeed News']
CNBC_data = data[data['publication'] == 'CNBC']
CNN_data = data[data['publication'] == 'CNN']
Economicst_data = data[data['publication'] == 'Economist']
Fox_News_data = data[data['publication'] == 'Fox News']
Gizmodo_data = data[data['publication'] == 'Gizmodo']
Hyperallergic_data = data[data['publication'] == 'Hyperallergic']
Mashable_data = data[data['publication'] == 'Mashable']
New_Republic_data = data[data['publication'] == 'New Republic']
New_Yorker_data = data[data['publication'] == 'New Yorker']
People_data = data[data['publication'] == 'People']
Politico_data = data[data['publication'] == 'Politico']
Refinery_29_data = data[data['publication'] == 'Refinery 29']
Reuters_data = data[data['publication'] == 'Reuters']
TMZ_data = data[data['publication'] == 'TMZ']
TechCrunch_data = data[data['publication'] == 'TechCrunch']
The_Hill_data = data[data['publication'] == 'The Hill']
The_New_York_Times_data = data[data['publication'] == 'The New York Times']
The_Verge_data = data[data['publication'] == 'The Verge']
Vice_data = data[data['publication'] == 'Vice']
Vice_News_data = data[data['publication'] == 'Vice News']
Vox_data = data[data['publication'] == 'Vox']
Washington_data = data[data['publication'] == 'Washington Post']
Wired_data = data[data['publication'] == 'Wired']

# Save the filtered data to a new CSV file
Axios_data.to_csv('Axios_data.csv', index=False)
Business_Insider_data.to_csv('Business_Insider_data.csv', index=False)
Buzzfeed_News_data.to_csv('Buzzfeed_News_data.csv', index=False)
CNBC_data.to_csv('CNBC_data.csv', index=False)
CNN_data.to_csv('CNN_data.csv', index=False)
Economicst_data.to_csv('Economicst_data.csv', index=False)
Fox_News_data.to_csv('Fox_News_data.csv', index=False)
Gizmodo_data.to_csv('Gizmodo_data.csv', index=False)
Hyperallergic_data.to_csv('Hyperallergic_data.csv', index=False)
Mashable_data.to_csv('Mashable_data.csv', index=False)
New_Republic_data.to_csv('New_Republic_data.csv', index=False)
New_Yorker_data.to_csv('New_Yorker_data.csv', index=False)
People_data.to_csv('People_data.csv', index=False)
Politico_data.to_csv('Politico_data.csv', index=False)
Refinery_29_data.to_csv('Refinery_29_data.csv', index=False)
Reuters_data.to_csv('Reuters_data.csv', index=False)
TMZ_data.to_csv('TMZ_data.csv', index=False)
TechCrunch_data.to_csv('TechCrunch_data.csv', index=False)
The_Hill_data.to_csv('The_Hill_data.csv', index=False)
The_New_York_Times_data.to_csv('The_New_York_Times_data.csv', index=False)
The_Verge_data.to_csv('The_Verge_data.csv', index=False)
Vice_data.to_csv('Vice_data.csv', index=False)
Vice_News_data.to_csv('Vice_News_data.csv', index=False)
Vox_data.to_csv('Vox_data.csv', index=False)
Washington_data.to_csv('Washington_data.csv', index=False)
Wired_data.to_csv('Wired_data.csv', index=False)


