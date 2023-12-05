import os
import pickle

import numpy as np
import pandas as pd
import requests as rq
from joblib import Parallel, delayed

from scripts.logger import Logger
from scripts.util import chunk_list, flatten_list, compute_quality_weighted_sum

ORES_URL = lambda l: f'http://ores.wikimedia.org/v3/scores/{l}wiki'
headers = {"User-Agent": "custom_address@mail.com"} # change this accordingly
ORES_FEATURES_SEPT23 = ['feature.english.stemmed.revision.stems_length', 'feature.enwiki.infobox_images',
                        'feature.enwiki.main_article_templates', 'feature.enwiki.revision.category_links',
                        'feature.enwiki.revision.cite_templates', 'feature.enwiki.revision.cn_templates',
                        'feature.enwiki.revision.image_links', 'feature.enwiki.revision.image_template',
                        'feature.enwiki.revision.images_in_tags', 'feature.enwiki.revision.images_in_templates',
                        'feature.enwiki.revision.infobox_templates',
                        'feature.enwiki.revision.paragraphs_without_refs_total_length',
                        'feature.enwiki.revision.shortened_footnote_templates', 'feature.enwiki.revision.who_templates',
                        'feature.len(<datasource.english.idioms.revision.matches>)',
                        'feature.len(<datasource.english.words_to_watch.revision.matches>)',
                        'feature.len(<datasource.wikitext.revision.words>)', 'feature.wikitext.revision.chars',
                        'feature.wikitext.revision.content_chars', 'feature.wikitext.revision.external_links',
                        'feature.wikitext.revision.headings_by_level(2)',
                        'feature.wikitext.revision.headings_by_level(3)', 'feature.wikitext.revision.list_items',
                        'feature.wikitext.revision.ref_tags', 'feature.wikitext.revision.templates',
                        'feature.wikitext.revision.wikilinks']


def parse_articlequality(aq_score_json):
    rev_results = []
    for revid, results in aq_score_json.items():
        aq_res = results['articlequality']
        if 'score' not in aq_res:
            rev_results.append([int(revid), 'error', pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA])
        else:
            pred, prob = aq_res['score']['prediction'], aq_res['score']['probability']
            rev_results.append([int(revid), pred, prob['Stub'], prob['Start'], prob['C'], prob['B'],
                                prob['GA'], prob['FA']])

    return pd.DataFrame(rev_results, columns=['revid', 'prediction', 'Stub', 'Start', 'C', 'B', 'GA', 'FA'])


def parse_articlequality_features(aq_score_json):
    rev_results = []
    for revid, results in aq_score_json.items():
        if 'features' not in results['articlequality']:
            rev_results.append([int(revid), 'error'] + (len(ORES_FEATURES_SEPT23) * [np.nan]))
        else:
            rev_results.append([int(revid), 'ok'] +
                               [results['articlequality']['features'][f_name] for f_name in ORES_FEATURES_SEPT23])
    return pd.DataFrame(rev_results, columns=['revid', 'status'] + ORES_FEATURES_SEPT23)


def retrieve_articlequality_for_revisions(revisions, lang='en'):
    params = {
        'models': 'articlequality',
        'revids': '|'.join([str(rev) for rev in revisions])
    }
    rq_call = rq.get(url=ORES_URL(lang), params=params)
    rq_res = rq_call.json()
    return parse_articlequality(rq_res[f'{lang}wiki']['scores'])


def retrieve_articlequality_features_for_revisions(revisions, lang='en'):
    params = {
        'models': 'articlequality',
        'features': 'true'
    }
    quality_scores, quality_features, errors = [], [], []
    for revision in revisions:
        try:
            rq_call = rq.get(url=f'{ORES_URL(lang)}/{revision}', params=params)
            # Logger.instance(name='ores-retrieval-calls').info(rq_call.url)
            # print(rq_call.url)
            rq_res = rq_call.json()
            quality_scores.append(parse_articlequality(rq_res[f'{lang}wiki']['scores']))
            # print(json.dumps(json_object, indent=2))
            quality_features.append(parse_articlequality_features(rq_res[f'{lang}wiki']['scores']))
        except Exception as exp:
            errors.append(revision)
            Logger.instance(name='ores-retrieval-exp').info(f'Error while retrieving {revision}\n{repr(exp)}')
            quality_features, quality_scores = pd.DataFrame(), pd.DataFrame()
    return (pd.concat(quality_scores).merge(
        pd.concat(quality_features), on='revid', how='left')
            if len(quality_scores) > 0 and len(quality_features) > 0 else pd.DataFrame()), errors


def parse_articletopic(at_score_json):
    return parse_ores_dict_result(at_score_json, 'articletopic')


def parse_ores_dict_result(at_score_json, model_name):
    rev_results, pred_labels = [], []
    for revid, results in at_score_json.items():
        at_res, pred_results = results[model_name], [int(revid)]
        if 'score' not in at_res:
            pred_results.append('error')
        else:
            if 'probability' not in at_res['score']:
                pred_results.append('error')
            else:
                pred_results.append('ok')

                pred_results += list(at_res['score']['probability'].values())
                pred_labels = list(at_res['score']['probability'].keys())
        rev_results.append(pred_results)

    return pd.DataFrame(rev_results, columns=['revid', 'result'] + pred_labels)


def parse_damaging(at_score_json):
    return parse_ores_dict_result(at_score_json, 'damaging')


def parse_goodfaith(at_score_json):
    return parse_ores_dict_result(at_score_json, 'goodfaith')


def retrieve_articletopic_for_revisions(revisions, lang='en'):
    params = {
        'models': 'articletopic',
        'revids': '|'.join([str(rev) for rev in revisions])
    }
    rq_call = rq.get(url=ORES_URL(lang), params=params)
    rq_res = rq_call.json()
    return parse_articletopic(rq_res[f'{lang}wiki']['scores'])


def retrieve_articlequality(revisions, lang='en', n_jobs=1, rev_batch=20, retrieve_features=False):
    if retrieve_features:
        chunked_revisions = chunk_list(revisions, n_jobs) if n_jobs > 1 else [revisions]
        results = Parallel(n_jobs=n_jobs)(delayed(retrieve_articlequality_features_for_revisions)(
            rev_batch, lang) for rev_batch in chunked_revisions)
        return results
    else:
        chunked_revisions = chunk_list(revisions, rev_batch)
        results = Parallel(n_jobs=n_jobs)(delayed(retrieve_articlequality_for_revisions)(
            rev_batch, lang) for rev_batch in chunked_revisions)
        return pd.concat(results).reset_index(drop=True)


def retrieve_articletopic(revisions, lang='en', n_jobs=1, rev_batch=50):
    chunked_revisions = chunk_list(revisions, rev_batch)
    results = Parallel(n_jobs=n_jobs)(delayed(retrieve_articletopic_for_revisions)(rev_batch, lang)
                                      for rev_batch in chunked_revisions)
    return pd.concat(results).reset_index(drop=True)


def retrieve_multiple_models_for_revisions(revisions,
                                           models=['articletopic', 'articlequality', 'damaging', 'goodfaith'],
                                           lang='en'):
    MODEL_PARSE_FUNCS = {'articletopic': parse_articletopic,
                         'articlequality': parse_articlequality,
                         'damaging': parse_damaging,
                         'goodfaith': parse_goodfaith}
    params = {
        'models': '|'.join(models),
        'revids': '|'.join([str(rev) for rev in revisions])
    }
    try:
        rq_call = rq.get(url=ORES_URL(lang), params=params)
        rq_res = rq_call.json()
        results = {model: MODEL_PARSE_FUNCS[model](rq_res[f'{lang}wiki']['scores']) for model in models}
        errors = []
        exception = None
    except Exception as exp:
        Logger.instance(name='ores-retrieval-exp').info(f'Error while retrieving ({revisions[0]} - {revisions[-1]}): '
                                                        f'{repr(exp)}')
        exception = repr(exp)
        errors = revisions
        results = {model: pd.DataFrame() for model in models}
    return results, errors, exception


def retrieve_multiple_models(revisions, models=['articletopic', 'articlequality', 'damaging', 'goodfaith'], lang='en',
                             n_jobs=4, rev_batch=5):
    chunked_revisions = list(chunk_list(revisions, rev_batch))
    results = Parallel(n_jobs=n_jobs)(delayed(retrieve_multiple_models_for_revisions)(rev_batch, models, lang)
                                      for rev_batch in chunked_revisions)
    results = list(zip(*results))
    return results
    # {model: pd.concat([res[model] if model in res else pd.DataFrame() for res in results[0]]).reset_index(
    # drop=True) for model in models}, flatten_list(results[1])


def retrieve_ores_metrics_per_day(df_revisions: pd.DataFrame, models=['articletopic', 'articlequality'], lang='en',
                                  n_jobs=4, rev_batch=50):
    df_first_last = df_revisions.groupby(['page_title_historical', 'start', 'event_timestamp_day']).revision_id.agg(
        ['first', 'last']).rename({'first': 'day_first', 'last': 'day_last'}, axis=1)
    df_first_last['day_first'] = df_first_last['day_first'].astype(int)
    df_first_last['day_last'] = df_first_last['day_last'].astype(int)
    ret_revids = list(set(df_first_last.day_first.to_list() + df_first_last.day_first.to_list()))
    return df_first_last, retrieve_multiple_models(ret_revids, models, lang, n_jobs, rev_batch)


def build_ores_df_from_pickle(pickle_file, pickle_path):
    with open(f'{pickle_path}/{pickle_file}', 'rb') as handle:
        chunk_res = pickle.load(handle)
    dict_res = {model: pd.concat([res[model] if model in res else pd.DataFrame() for res in chunk_res[0]]).reset_index(
        drop=True) for model in chunk_res[0][0].keys()}
    errors = flatten_list(chunk_res[1])

    # ===== articlemetrics
    # quality
    df_2019_quality = dict_res['articlequality']
    compute_quality_weighted_sum(df_2019_quality)
    df_2019_quality_merge = df_2019_quality[['revid', 'prediction', 'Q_score']].rename(
        {'revid': 'revision_id', 'prediction': 'Q_pred'}, axis=1)

    # topic
    df_2019_topics = dict_res['articletopic']
    probability_columns = df_2019_topics.drop(['revid', 'result'], axis=1).columns
    result_df = df_2019_topics[['revid', 'result']].copy()
    result_df['high_prob_topics'] = df_2019_topics[probability_columns].gt(0.5).apply(
        lambda row: row.index[row].tolist(), axis=1)
    result_df['is_STEM'] = result_df['high_prob_topics'].apply(
        lambda hc: False if len(hc) < 1 else np.any(list(l.startswith('STEM.') for l in hc)))
    result_df['is_Culture'] = result_df['high_prob_topics'].apply(
        lambda hc: False if len(hc) < 1 else np.any(list(l.startswith('Culture.') for l in hc)))
    result_df['is_HandS'] = result_df['high_prob_topics'].apply(
        lambda hc: False if len(hc) < 1 else np.any(list(l.startswith('History and Society.') for l in hc)))
    result_df['is_Geo'] = result_df['high_prob_topics'].apply(
        lambda hc: False if len(hc) < 1 else np.any(list(l.startswith('Geography.') for l in hc)))
    result_df['highest_prob_topic'] = df_2019_topics[probability_columns].idxmax(axis=1)
    result_df['highest_prob_topic_prob'] = df_2019_topics[probability_columns].max(axis=1)
    df_2019_topics_merge = result_df[['revid', 'is_STEM', 'is_Culture', 'is_HandS', 'is_Geo', 'highest_prob_topic',
                                      'highest_prob_topic_prob']].rename({'revid': 'revision_id'}, axis=1)

    # ===== editmetrics
    # goodfaith/damaging
    df_2019_gf, df_2019_damaging = dict_res['goodfaith'], dict_res['damaging']
    df_2019_eq = df_2019_gf.merge(df_2019_damaging, on='revid', how='outer', suffixes=('_goodf', '_dmg'))
    df_2019_eq['prob_gf'] = df_2019_eq.true_goodf
    df_2019_eq['prob_dmg'] = df_2019_eq.true_dmg
    df_2019_eq['is_goodf'] = df_2019_eq.true_goodf > 0.5
    df_2019_eq['is_dmg'] = df_2019_eq.true_dmg > 0.5
    df_2019_eq_merge = df_2019_eq[['revid', 'prob_gf', 'is_goodf', 'prob_dmg', 'is_dmg']].rename(
        {'revid': 'revision_id'}, axis=1)

    df_2019_ores = df_2019_topics_merge.merge(df_2019_quality_merge, how='outer', on='revision_id').merge(
        df_2019_eq_merge, how='outer', on='revision_id')

    return df_2019_ores, errors


def build_ores_df_from_pickle_files(prefix, pickle_path='data/ores/experiment/'):
    files = [filename for filename in os.listdir(pickle_path) if
             filename.startswith(prefix) and filename.endswith('.pkl')]
    errors, df_complete = [], pd.DataFrame()
    for file in files:
        Logger.instance('ores-compute').info(f'Start processing {file}')
        df_ores, errors = build_ores_df_from_pickle(file, pickle_path)
        errors.extend(errors)
        df_complete = pd.concat([df_complete, df_ores])
    return df_complete, errors

def load_ores_feature_files(prefix='quality-features', pickle_path='data/ores/experiment/'):
    files = [filename for filename in os.listdir(pickle_path) if
             filename.startswith(prefix) and filename.endswith('.pkl')]
    dfs, errors = [], []
    for file in files:
       Logger.instance('ores-compute').info(f'Start processing {file}')
       with open(f'{pickle_path}/{file}', 'rb') as handle:
            chunk_res = pickle.load(handle)
            dfs.append(chunk_res[0])
            errors.extend(chunk_res[1])
    return pd.concat(dfs), errors

def retrieve_revids_as_chunks(revids_retrieve, file_prefix, file_path='../data/ores/experiment', skip_first_n=0,
                              chunk_size=2.5 * 10 ** 6, n_jobs=4, batch_size=5,
                              models=['articletopic', 'articlequality', 'damaging', 'goodfaith']):
    Logger.instance(name='ores-retrieval-exp').info(f'Start retrieving {file_prefix}')
    results = []
    for i, revids in enumerate(chunk_list(revids_retrieve, int(chunk_size))):
        if i < skip_first_n:
            # if something fails, this could be used to retrieve everything starting from a certain chunk
            continue
        result = retrieve_multiple_models(list(revids), models=models, batch_size=batch_size, n_jobs=n_jobs)
        # result_list.append(result)
        Logger.instance(name='ores-retrieval-exp').info(f'Loaded {i}, {revids[0]}-{revids[-1]}')

        with open(f'{file_path}/{file_prefix}_chunk_{i:03d}_{revids[0]}-{revids[-1]}.pkl', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        Logger.instance(name='ores-retrieval-exp').info(f'Stored {i}, {revids[0]}-{revids[-1]}')
        results.append(result)
    return results


def retrieve_features_as_chunks(revids_retrieve, file_prefix, file_path='data/ores/experiment', skip_first_n=0,
                                chunk_size=5 * 10 ** 4, n_jobs=4):
    Logger.instance(name='ores-retrieval-exp').info(f'Start retrieving {file_prefix}')
    results, errors = [], []
    for i, revids in enumerate(chunk_list(revids_retrieve, int(chunk_size))):
        if i < skip_first_n:
            # if something fails, this could be used to retrieve everything starting from a certain chunk
            continue
        result = retrieve_articlequality(list(revids), n_jobs=n_jobs, retrieve_features=True)
        all_res, all_errors = [], []
        for r, e in result:
            all_errors.append(e)
            all_res.append(r)
        result_data = pd.concat(all_res)
        # result_list.append(result)
        Logger.instance(name='ores-retrieval-features').info(f'Loaded {i}, {revids[0]}-{revids[-1]}')
        print(len(result_data))
        with open(f'{file_path}/{file_prefix}_chunk_{i:03d}_{revids[0]}-{revids[-1]}.pkl', 'wb') as handle:
            pickle.dump((result_data, all_errors), handle, protocol=pickle.HIGHEST_PROTOCOL)
        Logger.instance(name='ores-retrieval-exp').info(f'Stored {i}, {revids[0]}-{revids[-1]}')
        results.append(result_data)
        errors.append(flatten_list(all_errors))

    return pd.concat(results), flatten_list(errors)
