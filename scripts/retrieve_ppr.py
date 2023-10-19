import io
import os
import zipfile

import numpy as np
import pandas as pd

from scripts.retrieve_wp import get_page_contents
from scripts.util import get_day_range_for_months, read_txt

PP_ARCHIVE = {'en': 'Wikipedia:Requests for page protection/Archive'}


def get_page_protection_requests(date_from='2012-10-01', date_to='2023-03-01', language='en',
                                 save_path='../datasets/wiki_pp/pp-requests/en'):
    print(f'Retrieve PPR from {date_from} to {date_to}')
    months = pd.date_range(date_from, date_to, freq='MS').strftime("%Y/%m").tolist()
    pc = get_page_contents([f'{PP_ARCHIVE[language]}/{month_str}' for month_str in months], language)
    print(months)
    res_dict = dict(zip(months, pc))
    if save_path:
        print(f'Saving in {save_path}.')
        for month, content in res_dict.items():
            np.savetxt(f'{save_path}/{month.replace("/", "_")}.txt', [content], fmt='%s',
                       delimiter='NODELIMPORFAVORMERCI', comments='NOCOMMENTSPORFAVORMERCI', newline='')
    return res_dict


def load_pp_req(save_path='../datasets/wiki_pp/pp-requests/en'):
    if save_path.endswith('.zip'):
        file_contents = {}
        with zipfile.ZipFile(save_path) as zf:
            all_files = zf.namelist()
            for file in all_files:
                with io.TextIOWrapper(zf.open(file), encoding="utf-8") as f:
                    file_contents[file.replace('.txt', '').replace('_', '/')] = f.read()
        return file_contents
    else:
        all_files = os.listdir(save_path)
        return {f.replace('.txt', '').replace('_', '/'):
                    read_txt(f'{save_path}/{f}') for f in all_files if f.endswith('.txt')}


def count_active_spells_per_day(df_spells, from_date=20080101, to_date=20230101):
    df_spells = df_spells.copy()
    df_spells.start.fillna(pd.to_datetime(str(from_date), format='%Y%m%d', utc=True), inplace=True)
    df_spells.end.fillna(pd.to_datetime(str(to_date), format='%Y%m%d', utc=True), inplace=True)

    df_spells['start_day'] = df_spells.start.dt.date
    df_spells['end_day'] = df_spells.end.dt.date
    print(np.sum(df_spells.duplicated(['title', 'start_day'])))
    days = get_day_range_for_months(from_date, to_date, ret_tuple=False)
    res_days = pd.DataFrame(days.date, columns=['date'])
    for level, df_level in df_spells.groupby('level'):
        res_days[level] = [np.sum((df_level.start <= day) & (day <= df_level.end)) for day in days]

    return res_days
