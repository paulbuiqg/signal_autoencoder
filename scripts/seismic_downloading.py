"""Download data from the Northern California Earthquake Data Center.

Source: https://ncedc.org/web-services-home.html
"""


import os
import pickle
from typing import List

import obspy
import pandas as pd
import requests
import xmltodict
from joblib import Parallel, delayed
from tqdm import tqdm


def fetch_events() -> pd.DataFrame:
    """Query NCDEC for events with magnitude between 5 and 9."""
    # NCEDC request
    response = requests.get(('https://service.ncedc.org/fdsnws/event/1/query?'
                             'minmag=5&maxmag=9'))
    d = xmltodict.parse(response.content)
    event_list = []
    # Event dataframe
    for evt in d['q:quakeml']['eventParameters']['event']:
        event_id, magnitude, magnitude_type = evt['@catalog:eventid'], \
            evt['magnitude']['mag']['value'], evt['magnitude']['type']
        if isinstance(evt['origin'], list):
            evt_origin = evt['origin'][0]
        else:
            evt_origin = evt['origin']
        latitude, longitude, time = evt_origin['latitude']['value'], \
            evt_origin['longitude']['value'], evt_origin['time']['value']
        event_type = evt['type']
        event_list.append([event_id, magnitude, magnitude_type, latitude,
                           longitude, event_type, time])
    return pd.DataFrame(columns=['eventID', 'magnitude', 'magnitude_type',
                                 'latitude', 'longitude', 'event_type',
                                 'time'],
                        data=event_list)


def fetch_data_for_one_event(event_id: int) -> List[obspy.Trace]:
    """Query NCDEC with ObsPy for event traces.

    Keep 3-channel seismigraphs with channels HNE, HNN, HNZ.
    (https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/)
    """
    # Query web service
    st = obspy.read(('https://service.ncedc.org/ncedcws/eventdata/1/query?'
                     f'eventid={event_id}'))
    #
    trace_stats = []
    traces = []
    i = 0
    for trace in st.traces:
        if trace.stats.channel not in ['HNE', 'HNN', 'HNZ']:
            continue
        dict_trace = dict(trace.stats)
        del dict_trace['mseed']
        trace_stats.append(dict_trace)
        traces.append(trace)
        i += 1
    trace_stats = pd.DataFrame(trace_stats)
    trace_stats['n_channel'] = (
        trace_stats[['network', 'station', 'location', 'channel']]
        .groupby(['network', 'station', 'location'])
        .transform('count')
    )
    #
    trace_stats = trace_stats.loc[(trace_stats['n_channel'] == 3) &
                                  (trace_stats['_format'] == 'MSEED')]
    trace_stats = (
        trace_stats
        .sort_values(['network', 'station', 'location', 'channel'])
        .drop(columns=['n_channel', '_format'])
    )
    traces = [traces[i] for i in trace_stats.index]
    return traces


def process_one_event(event_id: str, files: list):
    """Write signal files for one seismic event."""
    if any(event_id in f for f in files):
        return
    try:
        traces = fetch_data_for_one_event(event_id)
        i, j = 0, 0
        while j + 2 < len(traces):
            tr = traces[j: j + 3]
            with open(f'data/{event_id}_{i}.pkl', 'wb') as f:
                pickle.dump(tr, f)
            i += 1
            j += 3
    except BaseException as e:
        print(f'Event {event_id}:', e)


if __name__ == '__main__':

    # Create data directory
    if not os.path.exists('data'):
        os.makedirs('data')

    # Event information to file
    df_event = fetch_events()
    df_event.to_csv('data/events.csv', index=False)
    files = os.listdir('data')

    # Download "trace" data
    Parallel(n_jobs=-1)(delayed(process_one_event)(event_id, files)
                        for event_id in tqdm(df_event['eventID']))
