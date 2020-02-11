import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
import csv

def read_data():
    data = pd.read_excel('smaorter-2015_ver2.xlsx', skiprows=list(range(9)), usecols=list(range(4, 10)))
    data2 = pd.read_excel('tatorter-2015.xlsx', skiprows=list(range(10)), usecols=list(range(4, 20)))

    data.drop_duplicates(subset='Distriktsnamn', inplace=True)

    smaort = list(data.Distriktsnamn.unique())
    tatort = list(data2.Tätortsbeteckning.unique())
    #print(smaort)
    #print(tatort)

    sma_by = list(filter(lambda x: 'by' in x, smaort))
    sma_stad = list(filter(lambda x: 'stad' in x, smaort))

    print('Antal byar i grupp småort: {}'.format(len(sma_by)))
    print('Antal städer i grupp småort: {}'.format(len(sma_stad)))
    
    tat_by = list(filter(lambda x: 'by' in x, tatort))
    tat_stad = list(filter(lambda x: 'stad' in x, tatort))

    print('Antal byar i grupp tätort: {}'.format(len(tat_by)))
    print('Antal städer i grupp tätort: {}'.format(len(tat_stad)))

    return data, data2, smaort, tatort

_, _, smaort, tatort = read_data()
total = smaort+tatort

sammansattning = {name: [] for name in total}

def SALDO_sammansattning(ortsbeteckning):
    for ort in ortsbeteckning:
        r = requests.get('http://spraakbanken.gu.se/ws/saldo-ws/sms/json/{}'.format(ort))
        try:
            jsondecode = r.json()
            #print(jsondecode)
            if jsondecode != []:
                n = r.json()
                for item in n[0]:
                    print(item)
                    if item in sammansattning[ort]:
                        continue
                    #print(item['segment'])
                    else:
                        sammansattning[ort].append(item['segment'])
            if jsondecode == []:
            #else:
                sammansattning[ort].append(ort)
                #print(ort)
                #pass
        except:
            pass

SALDO_sammansattning(total)
print(sammansattning)


def scrapewiki(url):
    wiki_url = requests.get(url).text
    soup = BeautifulSoup(wiki_url,'lxml')
    tag = soup.find('div', {'class' : 'toc'})
    # find all span tags which contain the place name
    spans = tag.findAll('span')
    efterled = [span.contents[0] for span in spans]
    # the following line will still leave some redundant "floatlike" strings:
    efterled = [i for i in efterled[9:-11] if not i.isdigit()]
    efterled = [re.sub(r'[0-9|-]', '', s) for s in efterled]
    efterled = [re.sub(r'/', ' ', s) for s in efterled]
    empty = ['hult', 'borg', 'sta', 'vik', 'strand', 'berg', 'norra', 'södra', 'västra', 'östra', 'bro', 'bron', 'lund', 'sund']
    for suffix in efterled:
        n = suffix.split(',')
        n = n[0].split(' ')
        empty.extend(n)
    return empty

efterled = scrapewiki('https://sv.wikipedia.org/wiki/Svenska_ortnamnsefterled')

for key, value in sammansattning.items():
    #print(value)
    #print(key)
    #if len([key]) > 1:
    if len(value) > 1:
        continue
    else:    
        for suffix in efterled:
            if key.endswith(suffix):
                sammansattning[key] = [key.split(suffix)[0], suffix]
                continue

#print(sammansattning)


# for key, value in sammansatta.items():
#     #print(key)
#     if len([key]) > 1:
#        pass
#     else:    
#         for suffix in efterled:
#             if key.endswith(suffix):
#                 sammansatta[key] = [key.split(suffix)[0], suffix]
#                 continue

# print(sammansatta)

#sammansatta = SALDO_sammansattning(total)

# w = csv.writer(open("placenames.csv", "w"))
# for key, val in sammansattning.items():
#     w.writerow([key, val])


