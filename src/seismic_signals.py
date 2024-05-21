# %%

import pickle

import dataloading 

# %%

df_event = dataloading.fetch_events()
print(df_event)

# %%

traces = []

for event_id in df_event['eventID']:
    print(event_id)
    try:
        traces = dataloading.fetch_data_for_one_event(event_id)
        i, j = 0, 0
        while j + 2 < len(traces):
            tr = traces[j: j + 3]
            with open(f'data/{event_id}_{i}.pkl', 'wb') as f:
                pickle.dump(tr, f)
            i += 1
            j += 3
    except Exception as e:
        print(f'Event {event_id}:', e)
        continue