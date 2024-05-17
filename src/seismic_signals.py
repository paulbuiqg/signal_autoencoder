# %%

import obspy
import requests
import xml.etree.ElementTree as ET
import xmltodict

# Request NCEDC
response = requests.get('https://service.ncedc.org/fdsnws/event/1/query?minmag=5&maxmag=9')
tree = ET.fromstring(response.text)
d = xmltodict.parse(response.content)

# To get one eventid
print(d['q:quakeml']['eventParameters']['event'][0]['@catalog:eventid'])

st = obspy.read('https://service.ncedc.org/ncedcws/eventdata/1/query?eventid=73926401')
print(len(st))
st[0].plot()

# %%
