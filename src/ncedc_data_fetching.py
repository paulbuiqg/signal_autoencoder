"""Download data from the NCEDC."""


import os
import pickle

import dataloading


# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

# Event information to file
df_event = dataloading.fetch_events()
df_event.to_csv('data/events.csv', index=False)

# Download "trace" data
traces = []
files = os.listdir('data')
for event_id in df_event['eventID']:
    if any(str(event_id) in f for f in files):
        print(f'{event_id} ok')
        continue
    print(event_id)
    try:
        traces = dataloading.fetch_data_for_one_event(event_id)
        i, j = 0, 0
        while j + 2 < len(traces):
            tr = traces[j: j + 3]
            with open(f'../data/{event_id}_{i}.pkl', 'wb') as f:
                pickle.dump(tr, f)
            i += 1
            j += 3
    except BaseException as e:
        print(f'Event {event_id}:', e)
        continue
