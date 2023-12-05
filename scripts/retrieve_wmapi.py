import numpy as np
import pandas as pd
import requests as rq
from joblib import Parallel, delayed
from mw.lib import title

from scripts.util import get_day_range_for_months, get_month_range_for_months, split_periods, get_wiki_timestamp, \
    chunk_list

WM_API = 'https://wikimedia.org/api/rest_v1'
headers = {"User-Agent": "custom_address@mail.com"} # change this accordingly

PV_ACCESS_ALL = 'all-access'
PV_ACCESS_DESKTOP = 'desktop'
PV_ACCESS_MOBAPP = 'mobile-app'
PV_ACCESS_MOBWEB = 'mobile-web'

PV_AGENT_ALL = 'all-agents'
PV_AGENT_USER = 'user'
PV_AGENT_SPIDER = 'spider'
PV_AGENT_AUTOMATED = 'automated'
USER_ALL_TYPES = [PV_AGENT_USER, PV_AGENT_SPIDER, PV_AGENT_AUTOMATED]

PV_GRANULARITY_HOUR = 'hourly'
PV_GRANULARITY_DAY = 'daily'
PV_GRANULARITY_MONTH = 'monthly'

EDITOR_TYPE_ANON = 'anonymous'
EDITOR_TYPE_USER = 'user'
EDITOR_TYPE_GBOT = 'group-bot'
EDITOR_TYPE_NBOT = 'name-bot'
EDITOR_TYPE_ALL_EDITOR_TYPES = 'all-editor-types'
EDITOR_ALL_TYPES = [EDITOR_TYPE_ANON, EDITOR_TYPE_USER, EDITOR_TYPE_GBOT, EDITOR_TYPE_NBOT]


def get_redirects(pagetitle, language):
    URL = "https://" + language + ".wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": pagetitle,  # q.utils.quote(pagetitle),
        "prop": "redirects",
        "rdlimit": "100"
    }
    red_req = rq.get(url=URL, params=PARAMS, headers=headers)
    print(red_req.url)
    try:
        data = red_req.json()
        page = data["query"]["pages"]

        redirects = []

        for key, value in page.items():
            for redirect in value["redirects"]:
                redirects.append(redirect["title"])
    except:
        redirects = [pd.NA]
    return redirects


def get_revisions_around_timestamp(pagetitle, ts_date, diff_before=pd.Timedelta('30D'), diff_after=pd.Timedelta('30D'),
                                   language='en',
                                   fields=('ids', 'timestamp', 'tags', 'userid', 'user', 'comment')):
    URL = "https://" + language + ".wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": pagetitle,
        "rvprop": '|'.join(fields),
        # "rvslots": '*',
        "formatversion": "2",
        "format": "json",
        "rvdir": "newer"
    }
    ts_date = pd.to_datetime(ts_date, utc=True)
    from_date = get_wiki_timestamp(ts_date - diff_before)
    to_date = get_wiki_timestamp(ts_date + diff_after)
    PARAMS['rvstart'] = get_wiki_timestamp(from_date)
    PARAMS['rvend'] = get_wiki_timestamp(to_date)

    if 'ids' in fields:
        fields = list(fields)
        fields.remove('ids')
        fields = ['revid', 'parentid'] + fields

    rev_data, errors = [], []

    lastContinue = {}
    data = {}
    while True:
        # Clone original request
        req = PARAMS.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = rq.get(URL, params=req)

        try:
            data = result.json()
            pages = data["query"]["pages"]

            for page in pages:  # should only be 1
                rev_data.extend(
                    [[pagetitle, ts_date] + [revision[field] if field in revision else np.NaN for field in fields] for
                     revision in page["revisions"]])

        except Exception as exp:
            print(f'Error for {pagetitle}', str(exp))
            errors.append([pagetitle, ts_date, result.request.url, str(exp)])
        finally:
            if 'continue' not in data:
                break
            lastContinue = data['continue']

    return pd.DataFrame(rev_data, columns=['pagetitle', 'request_timestamp'] + list(fields)), \
           pd.DataFrame(errors, columns=['pagetitle', 'request_timestamp', 'wmf_request', 'error'])


def get_revision_ids_around_date(pagetitle, date, request_timestamp, language='en', limit=3, direction='older',
                                 fields=('ids', 'timestamp', 'tags', 'userid', 'user', 'comment')):
    URL = "https://" + language + ".wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": pagetitle,
        "rvprop": '|'.join(fields),
        # "rvslots": '*',
        "formatversion": "2",
        "format": "json"
    }
    ts_date = pd.to_datetime(date, utc=True) - pd.Timedelta('1 Second')
    PARAMS['rvstart'] = get_wiki_timestamp(ts_date)
    PARAMS['rvdir'] = direction
    PARAMS['rvlimit'] = limit

    if 'ids' in fields:
        fields = list(fields)
        fields.remove('ids')
        fields = ['revid', 'parentid'] + fields

    rev_data, errors = [], []

    lastContinue = {}
    data = {}
    # while True:
    # Clone original request
    req = PARAMS.copy()
    # Modify it with the values returned in the 'continue' section of the last result.
    req.update(lastContinue)
    # Call API
    result = rq.get(URL, params=req)
    print(result.request.url)
    try:
        data = result.json()
        pages = data["query"]["pages"]

        for page in pages:  # should only be 1
            rev_data.extend(
                [[pagetitle, request_timestamp] + [revision[field] if field in revision else np.NaN for field in
                                                   fields] for
                 revision in page["revisions"]])

    except Exception as exp:
        print(f'Error for {pagetitle}', str(exp))
        errors.append([pagetitle, request_timestamp, result.request.url, str(exp)])
    finally:
        if 'continue' not in data:
            pass  # break
        else:
            lastContinue = data['continue']

    return pd.DataFrame(rev_data, columns=['pagetitle', 'request_timestamp'] + list(fields)), \
           pd.DataFrame(errors, columns=['pagetitle', 'request_timestamp', 'wmf_request', 'error'])


def get_revision_ids(pagetitle, date, language='en', limit=5, direction='newer'):
    URL = "https://" + language + ".wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": pagetitle,
        "rvprop": 'ids|timestamp',
        # "rvslots": '*',
        "formatversion": "2",
        "format": "json"
    }

    if date:
        PARAMS['rvstart'] = get_wiki_timestamp(date)
        PARAMS['rvdir'] = direction
        PARAMS['rvlimit'] = limit

    rev_data = []
    try:
        data = rq.get(url=URL, params=PARAMS, headers=headers).json()
        pages = data["query"]["pages"]

        for page in pages:  # should only be 1
            rev_data.extend(
                [[pagetitle, revision['revid'], revision['timestamp'], date] for revision in page["revisions"]])
    except Exception as exp:
        print(f'Error for {pagetitle}', str(exp))
        rev_data.extend([[pagetitle, np.NaN, str(exp), date]])
        print(rev_data)
    return pd.DataFrame(rev_data, columns=['pagetitle', 'revid', 'timestamp', 'request_timestamp'])


def get_revid_before_rfpp(df_rfpp: pd.DataFrame, lang='en', n_jobs=10, req_col='request_timestamp',
                          ts_col='min_timestamp', pagetitle_col='pagetitle'):
    # https://www.mediawiki.org/wiki/API:Revisions
    # get ids and possibly text around succesful protection requests and/or spells to derive topic and quality
    # before/after protection
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_revision_ids_around_date)(title, date, req_date, lang, 1) for title, req_date, date in
        df_rfpp[[pagetitle_col, req_col, ts_col]].values)
    dfs, errors = list(zip(*results))

    return pd.concat(dfs), pd.concat(errors)


def get_revid_for_rfpp(df_rfpp: pd.DataFrame, lang='en', n_jobs=10, ts_col='request_timestamp'):
    # https://www.mediawiki.org/wiki/API:Revisions
    # get ids and possibly text around succesful protection requests and/or spells to derive topic and quality
    # before/after protection

    df_revids = pd.concat(Parallel(n_jobs=n_jobs)(
        delayed(get_revision_ids)(title, date, lang) for title, date in df_rfpp[['norm_title', ts_col]].values))
    return df_revids


def get_revisions_for_rfpp_around_timestamp(df_rfpp: pd.DataFrame, diff_before=pd.Timedelta('30D'),
                                            diff_after=pd.Timedelta('30D'),
                                            lang='en', n_jobs=10, ts_col='request_timestamp'):
    # https://www.mediawiki.org/wiki/API:Revisions
    # get ids and possibly text around succesful protection requests and/or spells to derive topic and quality
    # before/after protection
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_revisions_around_timestamp)(str(title), date, diff_before, diff_after, lang) for title, date in
        df_rfpp[['norm_title', ts_col]].values)
    dfs, errors = list(zip(*results))

    return pd.concat(dfs), pd.concat(errors)


def get_redirects_for_pagetitles(pagetitles, language):
    redirect_dict = {title: [[title] + get_redirects(title, language)] for title in pagetitles}
    return pd.DataFrame.from_dict(redirect_dict, orient='index', columns=['redirect']).reset_index().rename(
        {'index': 'title'}, axis=1).explode('redirect')


def get_redirect_for_pagetitle_df(title, language):
    return pd.DataFrame.from_dict(
        {title: [[title] + get_redirects(title, language)]}, orient='index', columns=['redirect']).reset_index().rename(
        {'index': 'title'}, axis=1).explode('redirect')


def get_redirects_for_pagetitles_parallel(pagetitles, language, n_jobs=30):
    return pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(get_redirect_for_pagetitle_df)(title, language) for title in pagetitles))


def retrieve_edit_counts(lang, start=20180101, end=20201201, granularity='daily', editor_type='user',
                         page_type='content'):
    response = rq.get(
        f'{WM_API}/metrics/edits/aggregate/{lang}.wikipedia.org/{editor_type}/{page_type}/{granularity}/{start}/{end}')
    data = response.json()
    df_edits = pd.DataFrame(data['items'][0]['results'])
    df_edits['date'] = pd.to_datetime(df_edits.timestamp).dt.date
    df_edits['edits'] = df_edits.edits
    df_edits['user_kind'] = editor_type
    df_edits['lang'] = lang
    return df_edits[['edits', 'date', 'user_kind', 'lang']]


def retrieve_edit_counts_edit_types_lang(lang, start=20180101, end=20201201, edit_types=EDITOR_ALL_TYPES,
                                         granularity='daily', page_type='content'):
    return pd.concat([retrieve_edit_counts(
        lang, start, end, granularity, editor_type, page_type) for editor_type in edit_types], axis=0)


def retrieve_edits_per_article_and_editor_type(article, lang, start=20140101, end=20220101,
                                               editor_type=EDITOR_TYPE_USER,
                                               granularity=PV_GRANULARITY_DAY):
    lang_result = {'article': article, 'date': [], 'edits': [], 'type': editor_type, 'error': []}

    # This is another fun wrinkle in the Wikimedia API. The edits endpoint can only return timespans less than a year
    # This is DIFFERENT from the views endpoint, that can handle whatever
    for s, e in split_periods(start, end):
        url = f'{WM_API}/metrics/edits/per-page/{lang}.wikipedia.org/{rq.utils.quote(str(article), safe="")}/' \
              f'{editor_type}/{granularity}/{s}/{e if e > s else s + 1 if e == s else -1}'
        response = rq.get(url, headers=headers)
        try:
            string = response.json()
            if 'items' in string and 'results' in string['items'][0]:
                for res in string['items'][0]['results']:  # should only be one item
                    lang_result['date'].append(pd.to_datetime(res['timestamp'][:10], format='%Y-%m-%d'))
                    lang_result['edits'].append(res['edits'])
                    lang_result['error'].append(pd.NA)
            else:
                lang_result['edits'].append(pd.NA)
                lang_result['date'].append(s)
                lang_result['error'].append(string)
        except Exception as exc:
            lang_result['edits'].append(pd.NA)
            lang_result['date'].append(s)
            lang_result['error'].append(str(exc))
    return pd.DataFrame(lang_result)


def retrieve_edit_counts_edit_types_article(article, lang, start=20180101, end=20201201,
                                            edit_types=EDITOR_ALL_TYPES, granularity='daily'):
    return pd.concat(
        [retrieve_edits_per_article_and_editor_type(article, lang, start, end, editor_type, granularity) for editor_type
         in edit_types], axis=0)


def retrieve_edit_counts_edit_types_per_protected_article(article, lang, spell_start, spell_end, start=20180101,
                                                          end=20201201, edit_types=EDITOR_ALL_TYPES,
                                                          granularity='daily'):
    df_article = pd.concat(
        [retrieve_edits_per_article_and_editor_type(article, lang, start, end, editor_type, granularity) for editor_type
         in edit_types], axis=0)
    df_article['spell_start'] = spell_start
    df_article['spell_end'] = spell_end
    return df_article


def retrieve_edits_for_protected_articles_parallel(df_spells, lang='en', edit_types=EDITOR_ALL_TYPES,
                                                   granularity=PV_GRANULARITY_DAY, n_jobs=35,
                                                   date_lower='2008-01-01', date_upper='2023-03-01',
                                                   days_prev=7, days_post=8):
    df_spells = df_spells.copy()
    df_spells['start_day_ret'] = (df_spells.start.fillna(pd.to_datetime(date_lower, utc=True))
                                  - pd.Timedelta(f'{days_prev} days')).dt.strftime('%Y%m%d').astype(int)
    df_spells['end_day_ret'] = (df_spells.end.fillna(
        pd.to_datetime(date_upper, utc=True)) + pd.Timedelta(f'{days_post} days')).dt.strftime('%Y%m%d').astype(int)
    result_pairs = pd.concat(Parallel(n_jobs=n_jobs)(delayed(retrieve_edit_counts_edit_types_per_protected_article)(
        r.title, lang, r.start, r.end, r.start_day_ret, r.end_day_ret, edit_types, granularity)
                                                     for i, r in df_spells.iterrows()))
    return result_pairs


def retrieve_edit_counts_edit_types_articles(articles, lang, start=20180101, end=20201201, edit_types=EDITOR_ALL_TYPES,
                                             granularity='daily'):
    return pd.concat([retrieve_edit_counts_edit_types_article(
        article, lang, start, end, edit_types, granularity) for article in articles], axis=0)


def retrieve_pageviews_for_articles(articles, lang, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                    granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    df_retrieved = []
    for article in articles:
        df_art = retrieve_pageviews_per_article(article, lang, start, end, access, granularity, agent)
        df_art['article'] = article
        df_art['norm_article'] = title.normalize(article)
        df_retrieved.append(df_art)
    return pd.concat(df_retrieved)


def retrieve_pageviews_per_article(article, lang, start=20150101, end=20220101, access=PV_ACCESS_ALL,
                                   granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    article = str(article)
    # This endpoint is only available daily!
    # maybe we can still do "that were protected for a part of each day?" or day after
    url = f'{WM_API}/metrics/pageviews/per-article/{lang}.wikipedia.org/{access}/{agent}/' \
          f'{rq.utils.quote(str(article), safe="")}/{granularity}/{start}/{end}'

    # print(response.url)
    lang_result = {'date': [], 'views': [], 'article': [], 'error': []}

    response = rq.get(url, headers=headers)
    # print(url, response.text)
    try:
        string = response.json()
        if 'items' in string:
            for res in string['items']:
                lang_result['date'].append(pd.to_datetime(res['timestamp'][:-2], format='%Y%m%d'))
                lang_result['views'].append(res['views'])
                lang_result['article'].append(article)
                lang_result['error'].append(pd.NA)
        else:
            lang_result['article'].append(article)
            lang_result['views'].append(pd.NA)
            lang_result['date'].append(start)
            lang_result['error'].append(string)
    except Exception as e:
        lang_result['article'].append(article)
        lang_result['views'].append(pd.NA)
        lang_result['date'].append(start)
        lang_result['error'].append(str(e))

    return pd.DataFrame(lang_result)


def retrieve_pageviews_for_protected_articles(df_spells, lang='en', access=PV_ACCESS_ALL,
                                              granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    df_spells = df_spells.copy()
    df_spells['start_day_ret'] = df_spells.start.fillna(
        pd.to_datetime('2008-01-01', utc=True)).dt.strftime('%Y%m%d').astype(int)
    df_spells['end_day_ret'] = df_spells.end.fillna(
        pd.to_datetime('2023-01-01', utc=True)).dt.strftime('%Y%m%d').astype(int)
    return pd.concat(df_spells.apply(
        lambda r: retrieve_pageviews_per_article(r.title, lang, r.start_day_ret, r.end_day_ret, access, granularity,
                                                 agent), axis=1))


def retrieve_pageviews_per_protected_article(article, lang, spell_start, spell_end, start=20150101, end=20220101,
                                             access=PV_ACCESS_ALL, granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    df_article = retrieve_pageviews_per_article(article, lang, start=start, end=end, access=access,
                                                granularity=granularity, agent=agent)
    df_article['spell_start'] = spell_start
    df_article['spell_end'] = spell_end
    return df_article


def retrieve_pageviews_for_protected_articles_parallel(df_spells, lang='en', access=PV_ACCESS_ALL,
                                                       granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL,
                                                       n_jobs=50, date_lower='2008-01-01', date_upper='2023-01-01',
                                                       days_prev=7, days_post=8):
    df_spells = df_spells.copy()
    df_spells['start_day_ret'] = (df_spells.start.fillna(pd.to_datetime(date_lower, utc=True))
                                  - pd.Timedelta(f'{days_prev} days')).dt.strftime('%Y%m%d').astype(int)
    df_spells['end_day_ret'] = (df_spells.end.fillna(
        pd.to_datetime(date_upper, utc=True)) + pd.Timedelta(f'{days_post} days')).dt.strftime('%Y%m%d').astype(int)
    result_pairs = pd.concat(Parallel(n_jobs=n_jobs)(delayed(retrieve_pageviews_per_protected_article)(
        r.title, lang, r.start, r.end, r.start_day_ret, r.end_day_ret, access, granularity, agent) for i, r in
                                                     df_spells.iterrows()))
    return result_pairs


def retrieve_pageviews_aggregate(lang, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                 granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL, legacy=False):
    # this endpoint is available hourly and daily
    if not legacy:
        url = f'{WM_API}/metrics/pageviews/aggregate/{lang}.wikipedia.org/{access}/{agent}/{granularity}/{start}/{end}'
    else:
        url = f'{WM_API}/metrics/legacy/pagecounts/aggregate/{lang}.wikipedia.org/all-sites/{granularity}/{start}00/{end}00'

    response = rq.get(url, headers=headers)
    lang_result = {'date': [], 'views': [], 'type': []}
    # print(url, response.text)
    string = response.json()

    for res in string['items']:
        lang_result['date'].append(pd.to_datetime(res['timestamp'][:-2], format='%Y%m%d'))
        lang_result['views'].append(res['views'] if not legacy else res['count'])
        lang_result['type'].append(agent)
    return pd.DataFrame(lang_result)


def retrieve_pageviews_user_types_lang(lang, start=20180101, end=20201201, access=PV_ACCESS_ALL,
                                       view_types=USER_ALL_TYPES, granularity='daily', legacy=False):
    return pd.concat([retrieve_pageviews_aggregate(
        lang, start, end, access, granularity, type, legacy) for type in view_types], axis=0)


def retrieve_top_articles_by_pageviews(lang, start=20160101, end=20220101, access=PV_ACCESS_ALL, n_jobs=1):
    day_tuples = get_day_range_for_months(start, end, ret_tuple=True)
    month_tuples = [(y, m, 'all-days') for y, m, _ in get_month_range_for_months(start, end, ret_tuple=True)]

    return pd.concat(Parallel(n_jobs=n_jobs)(delayed(retrieve_top_articles_by_pageviews_day)(
        lang, year, month, day, access) for year, month, day in day_tuples + month_tuples))


def retrieve_top_articles_by_pageviews_day(lang, year, month, day, access=PV_ACCESS_ALL):
    url = f'{WM_API}/metrics/pageviews/top/{lang}.wikipedia.org/{access}/{year}/{month}/{day}'
    print(url)
    response = rq.get(url, headers=headers)
    lang_result = {'article': [], 'date': [], 'year': [], 'month': [], 'day': [], 'views': [], 'rank': []}
    # print(url, response.text)
    string = response.json()

    if 'items' in string:
        for res_art in string['items'][0]['articles']:
            lang_result['article'].append(res_art['article'])
            lang_result['views'].append(res_art['views'])
            lang_result['rank'].append(res_art['rank'])
            lang_result['year'].append(year)
            lang_result['month'].append(month)
            lang_result['day'].append(day)
            lang_result['date'].append(pd.to_datetime(f'{year}-{month}-{day}') if day != 'all-days' else pd.NA)
    return pd.DataFrame(lang_result)
