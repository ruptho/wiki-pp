import pickle
import re
from calendar import monthrange
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import List

import pandas as pd
from wikichatter.comment import Comment

EN_TS_FORMAT_1 = '%H:%M, %d %B %Y (%Z)'
EN_TS_FORMAT_2 = '%H:%M, %d %b %Y (%Z)'

TITLE_PATTERN_MOVES = r'".+?"(.+)"[^"]+"$'
CATEGORIES = lambda drop_missing: (["NoClass"] if not drop_missing else []) + ["Stub", "Start", "C", "B", "GA", "FA"]

COL_BLIND_PALETTE_3 = ['#1E88E5', '#FFC107', '#004D40']


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_replace_regex(t, pattern=TITLE_PATTERN_MOVES):
    if pd.isna(t):
        return t
    match = re.search(pattern, str(t))
    if match:
        return match.group(1)
    else:
        return t


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def flatten_arbitrary(xs):
    # https://stackoverflow.com/a/2158532
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten_arbitrary(x)
        else:
            yield x


def prune_string(text, remove_start_str='', remove_chars=' :*', remove_ws_symbol='&#32;', only_left=True):
    txt_new = text.lstrip(remove_chars) if only_left else text.strip(remove_chars)
    return txt_new.replace(remove_start_str, '').replace(remove_ws_symbol, '').strip()


def flatten_comment_tree(discussion: List[Comment]) -> List[Comment]:
    # https://stackoverflow.com/a/2158532
    crawled_comments = []
    stack_comments = discussion
    while len(stack_comments) > 0:
        curr_comment = stack_comments.pop(0)
        crawled_comments.append(curr_comment)
        stack_comments.extend(curr_comment.comments)
    return crawled_comments


def flatten_mixed_nested_list(l):
    total_list = []
    for item in l:
        if isinstance(item, list):
            total_list.extend(item)
        else:
            total_list.append(item)
    return total_list


def parse_timestamp(ts_string, utc=True):
    ts = pd.to_datetime(ts_string, format=EN_TS_FORMAT_1, utc=utc, errors='coerce')
    if pd.isna(ts):
        ts = pd.to_datetime(ts_string, format=EN_TS_FORMAT_2, utc=utc, errors='coerce')
        if pd.isna(ts):
            print(f'[ERROR] Parsing timestamp {ts_string}')
    return ts


def read_full_txt(path):
    with open(path, 'r') as content_file:
        content = content_file.read()
    return content


def extract_rfpp_from_wikitable(wikitable_file):
    full_txt = read_full_txt(wikitable_file)
    re_txt = re.compile('\|(.+?)\|\|.+?\|\|.+?\|')
    return [match.strip() for match in re_txt.findall(full_txt)]


def save_pickle(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def get_day_range_from_string(year_month_str):
    date_split = year_month_str.split('/')
    return get_day_range(int(date_split[0]), int(date_split[1]))


def get_day_range(year, month, ret_string=True):
    days = monthrange(year, month)[1]
    return [f'{year}{month:02d}{day:02d}' for day in range(1, days + 1)] if ret_string else [
        (year, month, day) for day in range(1, days + 1)]


def get_day_range_from_int(ymd_int, ret_tuple=False):
    year, month = int(str(ymd_int)[:4]), int(str(ymd_int)[4:6])
    days = monthrange(year, month)[1]
    return [f'{year}{month:02d}{day:02d}' for day in range(1, days + 1)] if not ret_tuple else [
        (year, month, day) for day in range(1, days + 1)]


def get_day_range_for_months(ymd_int_from, ymd_int_to, freq='D', ret_tuple=True):
    day_range = pd.date_range(pd.to_datetime(str(ymd_int_from), format='%Y%m%d', utc=True),
                              pd.to_datetime(str(ymd_int_to), format='%Y%m%d', utc=True), freq=freq)
    return day_range if not ret_tuple else [(f'{d.year}', f'{d.month:02d}', f'{d.day:02d}') for d in day_range]


def get_month_range_for_months(ymd_int_from, ymd_int_to, ret_tuple):
    return get_day_range_for_months(ymd_int_from, ymd_int_to, ret_tuple=ret_tuple)


def split_periods(start, end, date_format='%Y%m%d', start_on_end=False, cutoff_days=360):
    start_date = datetime.strptime(str(start), date_format)
    end_date = datetime.strptime(str(end), date_format)

    if end_date < start_date:
        print('End date is before start date')
        return None

    if (end_date - start_date).days <= cutoff_days:
        return [(start, end)]
    else:
        periods = []
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=cutoff_days)
            if current_end > end_date:
                current_end = end_date
            periods.append((int(current_start.strftime('%Y%m%d')), int(current_end.strftime('%Y%m%d'))))
            current_start = current_end + timedelta(days=1 if not start_on_end else 0)
        return periods


def get_wiki_timestamp(date):
    ts = pd.to_datetime(date)
    return ts.strftime('%Y-%m-%dT%H:%M:%SZ')


def format_significant_in_table(val, p_val=0.05):
    color = 'yellow' if val < p_val else ''
    bold = 'bold' if val < p_val else ''
    return f'font-weight: {bold};background-color:{color}'


def assign_quality_category(df_quality, drop_missing=True):
    categories = CATEGORIES(drop_missing)
    print(f'{len(df_quality)} Quality measurements, {sum(pd.isna(df_quality.prediction))} were NaN')
    df_q = (df_quality.dropna(subset=['prediction']) if drop_missing else df_quality).copy()

    df_q.prediction = pd.Categorical(df_q.prediction.fillna('NoClass'), categories=categories, ordered=True)
    df_q['prediction_int'] = pd.to_numeric(df_q.prediction.apply(lambda p: categories.index(p)))
    for c in categories:
        df_q[c] = pd.to_numeric(df_q[c])
    return df_q


def compute_quality_weighted_sum(df_quality, drop_missing=True):
    categories = CATEGORIES(drop_missing)

    # Quality metric proposed by Halfaker
    # "Interpolating Quality Dynamics in Wikipedia and Demonstrating the Keilana Effect"
    def weighted_quality_sum(r):
        return sum(r[c] * i for i, c in enumerate(categories))

    df_quality['Q_score'] = df_quality.apply(weighted_quality_sum, axis=1)
    return df_quality


def read_txt(path):
    with open(path) as fp:
        return fp.read()
