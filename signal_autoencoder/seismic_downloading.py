"""Download data from the NCEDC."""


import os

from joblib import Parallel, delayed
from tqdm import tqdm

import dataloading


# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

# Event information to file
df_event = dataloading.fetch_events()
df_event.to_csv('data/events.csv', index=False)
files = os.listdir('data')

# Download "trace" data
Parallel(n_jobs=-1)(delayed(dataloading.process_one_event)(event_id)
                    for event_id in tqdm(df_event['eventID']))
