import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def read_data():
    print('Importing Excel data')
    smaort_excel = pd.read_excel('excel_data/smaorter-2015_ver2.xlsx', skiprows=list(range(9)), usecols=list(range(4, 10)))
    tatort_excel = pd.read_excel('excel_data/tatorter-2015.xlsx', skiprows=list(range(10)), usecols=list(range(4, 20)))

    smaort_excel['Label'] = ['smaort' for item in smaort_excel.iterrows()]
    tatort_excel['Label'] = ['tatort' for item in tatort_excel.iterrows()]

    tatort_excel.rename(columns = {'Tätortsbeteckning': 'Distriktsnamn'}, inplace=True)

    all_names = pd.concat([smaort_excel, tatort_excel])

    all_names.drop_duplicates(subset='Distriktsnamn', inplace=True)
    all_names = all_names[['Distriktsnamn', 'Label']]

    train_df, test_df = train_test_split(all_names.sample(frac=1), test_size=0.2)

    smaort = list(smaort_excel.Distriktsnamn.unique())
    tatort = list(tatort_excel.Distriktsnamn.unique())

    smaort_train = [o for o in smaort if (train_df['Distriktsnamn']==o).any()]
    tatort_train = [t for t in tatort if (train_df['Distriktsnamn']==t).any()]

    smaort_test = [o for o in smaort if (test_df['Distriktsnamn']==o).any()]
    tatort_test = [t for t in tatort if (test_df['Distriktsnamn']==t).any()]

    sma_by = list(filter(lambda x: 'by' in x, smaort))
    sma_stad = list(filter(lambda x: 'stad' in x, smaort))

    print('Antal byar i grupp småort: {}'.format(len(sma_by)))
    print('Antal städer i grupp småort: {}'.format(len(sma_stad)))

    tat_by = list(filter(lambda x: 'by' in x, tatort))
    tat_stad = list(filter(lambda x: 'stad' in x, tatort))

    print('Antal byar i grupp tätort: {}'.format(len(tat_by)))
    print('Antal städer i grupp tätort: {}'.format(len(tat_stad)))

    with open('smaort_test.pkl', 'wb') as f:
            pickle.dump(smaort_test, f)

    with open('tatort_test.pkl', 'wb') as f:
            pickle.dump(tatort_test, f)

    return smaort_excel, tatort_excel, all_names, smaort_train, tatort_train
