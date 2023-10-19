import bz2
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mw.lib import title as mw_t

from scripts.config import DUMPS_PATH
from scripts.logger import Logger
from scripts.preprocessing import combine_ranges


def filter_file(filename, search_strings):
    Logger.instance().info(f'Start filtering {filename}')
    i, results = 0, []
    with bz2.open(filename, 'rt') as file:
        for line in file:
            if any(search_string in line for search_string in search_strings):
                results.append(line)
            i += 1
            if i % 10 ** 8 == 0:
                Logger.instance().info(f'Processing {filename}: {i} lines')
    Logger.instance().info(f'Finished filtering {filename}')

    return results


def merge_temp_files(output_file, temp_folder):
    with bz2.open(output_file, 'wt') as output:
        for filename in os.listdir(temp_folder):
            temp_file = os.path.join(temp_folder, filename)
            with bz2.open(temp_file, 'rt') as temp:
                # Append the rest of the lines
                for line in temp:
                    output.write(line)


def filter_titles_files(search_strings, folder_path, output_folder, num_threads=None, sep='\t'):
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.tsv.bz2')]
    Logger.instance().info(f'Loading {len(file_list)} files: |{file_list}|')

    filtered_results = Parallel(n_jobs=num_threads if num_threads else len(file_list))(
        delayed(filter_file)(
            os.path.join(folder_path, filename),
            [f'{sep}{ss}{sep}' for ss in search_strings]
        ) for filename in file_list
    )

    Logger.instance().info(f'Finished filtering, merge results')
    filtered_lines = [line for result in filtered_results for line in result]

    # Merge filtered lines into the final output file
    output_file = os.path.join(output_folder, 'filtered.tsv.bz2')
    with bz2.open(output_file, 'wt') as output:
        for line in filtered_lines:
            output.write(line)

    Logger.instance().info(f'Finished merging')


def process_spell(spell_row, df_mw, days_before):
    article, start = spell_row['title'], spell_row['start']

    # Filter df_mw based on the criteria (within 24 hours before start)
    time_mask = (df_mw['date_utc'] >= start - pd.DateOffset(days=1)) & (df_mw['date_utc'] <= start)

    mw_count = df_mw[time_mask].groupby('page_title_historical').size()
    spell_count = mw_count.get(article, 0)
    others_count = mw_count[mw_count.index != article]

    # Check if the mw_count is within +/- 10% of the spell_count
    others_count = others_count[others_count.between(0.9 * spell_count, 1.1 * spell_count)]
    df_others = others_count.reset_index(name='count')
    df_others['page_title_historical'] = article
    df_others['start'] = start
    df_others['orig_count'] = spell_count
    return df_others


def process_lang_history(lang, column_names, dtypes, path=DUMPS_PATH, ending='tsv.bz2', years=list(range(2012, 2013)),
                         usecols=None, filter_revisions=True):
    df_lang = pd.DataFrame()
    # quick fix for small wikis
    try:
        start = time.time()
        df_all = pd.read_csv(f'{path}/{lang}.{ending}', sep='\t', names=list(column_names), dtype=dtypes,
                             on_bad_lines='warn', quoting=3, usecols=usecols)
        Logger.instance().info(f'Loaded {lang} in {time.time() - start}')
        return df_all
    except:
        traceback.print_exc()
        Logger.instance().info(f'PROBABLY EXPECTED ERROR: No "all-times" file for {lang}')

    for year in years:
        start = time.time()
        if lang == 'en':
            start = time.time()
            for month in range(1, 13):  # just throw exception when out of bounds here
                try:
                    df_month = pd.read_csv(
                        f'{path}/{lang}-{year}-{month:02d}.{ending}', sep='\t', names=list(column_names), dtype=dtypes,
                        on_bad_lines='warn', quoting=3, usecols=usecols).query('event_entity == "revision"')
                    df_lang = pd.concat([df_lang, df_month])
                    Logger.instance().info(f'Loaded {lang}-{year}-{month:02d} in {time.time() - start}')
                except:
                    traceback.print_exc()
                    Logger.instance().info(f'Error when processing {lang}-{year}-{month}')
        else:
            try:
                df_lang = pd.concat([df_lang, pd.read_csv(
                    f'{path}/{lang}-{year}.{ending}', sep='\t', names=list(column_names),
                    dtype=dtypes, on_bad_lines='warn', quoting=3, usecols=usecols)])
                Logger.instance().info(f'Loaded {lang}-{year} in {time.time() - start}')
            except:
                traceback.print_exc()
                Logger.instance().info(f'Error when processing {lang}-{year}')
    return df_lang


def find_similar_for_spells(df_mw, df_spells, days_before=1, n_partitions=100, n_jobs=10):
    import pandas as pd
    import dask.dataframe as dd
    from dask.distributed import Client

    # Start a Dask client
    client = Client(n_workers=n_jobs)

    # Convert the dataframes to Dask dataframes
    df_rel_dask = dd.from_pandas(df_spells, npartitions=n_partitions // 10)
    df_mw_dask = dd.from_pandas(df_mw, npartitions=n_partitions)

    # Define the processing function
    def process_spell(spell_row):
        article, start = spell_row['title'], spell_row['start']

        # Filter df_mw based on the criteria (within 24 hours before start)
        time_mask = (df_mw_dask['date_utc'] >= start - pd.DateOffset(days=days_before)) & (
                df_mw_dask['date_utc'] <= start)
        title_mask = (df_mw_dask['page_title_historical'] == article)

        mw_count = df_mw_dask[time_mask].groupby('page_title_historical').size().compute()
        spell_count = mw_count.get(article, 0)
        others_count = mw_count[mw_count.index != article]

        # Check if the mw_count is within +/- 10% of the spell_count
        others_count = others_count[others_count.between(0.9 * spell_count, 1.1 * spell_count)]
        df_others = others_count.reset_index(name='count')
        df_others['page_title_historical'] = article
        df_others['start'] = start
        df_others['orig_count'] = spell_count
        return df_others

    # Apply the processing function to each row in df_rel using Dask's map_partitions
    final_df_dask = df_rel_dask.map_partitions(lambda df: df.apply(process_spell, axis=1)).compute()
    final_df = pd.concat(final_df_dask, ignore_index=True)

    # Shut down the Dask client
    client.close()
    return final_df


def load_yearly_dump(lang, year, column_names, dtypes, path, ending='tsv.bz2', usecols=None):
    return pd.read_csv(f'{path}/{lang}-{year}.{ending}', sep='\t', dtype=dtypes,
                       names=usecols if usecols else column_names,
                       usecols=[column_names.tolist().index(col) for col in usecols] if usecols else None)


def filter_yearly_dumps(lang, years, included_titles, dtypes, path, ending='tsv.bz2'):
    df_lang_filtered = pd.DataFrame()
    for year in years:
        Logger.instance().info(f'Start loading {lang}-{year}')
        start = time.time()
        df_year = load_yearly_dump(lang, year, dtypes=dtypes, path=path, ending=ending)
        df_year = df_year[df_year.event_entity == 'revision']  # [restrict_columns]
        df_year['page_title_norm'] = df_year.page_title.swifter.apply(
            lambda title: mw_t.normalize(str(title)) if not pd.isna(title) else pd.NA)
        df_year['page_title_historical_norm'] = df_year.page_title_historical.swifter.apply(
            lambda title: mw_t.normalize(str(title)) if not pd.isna(title) else pd.NA)
        df_lang_filtered = pd.concat([df_lang_filtered,
                                      df_year[df_year.page_title_norm.isin(
                                          included_titles) | df_year.page_title_historical_norm.isin(included_titles)]])
        Logger.instance().info(f'Loaded {lang}-{year} in {time.time() - start}')
    df_lang_filtered.to_csv(f'{path}/{lang}-filtered-new.tsv.bz2', sep='\t', index=False)
    return df_lang_filtered


def select_columns_yearly_dumps(lang, years, included_columns, dtypes, path, ending='tsv.bz2'):
    df_lang_filtered = pd.DataFrame()
    for year in years:
        Logger.instance().info(f'Start loading {lang}-{year}')
        start = time.time()
        df_year = load_yearly_dump(lang, year, dtypes=dtypes, path=path, ending=ending, usecols=included_columns)
        df_lang_filtered = pd.concat([df_lang_filtered, df_year[df_year.event_entity == 'revision']])
        Logger.instance().info(f'Loaded {lang}-{year} in {time.time() - start}')
        if year == 2015:
            # store halfway through
            df_lang_filtered.to_csv(f'{path}/{lang}-filtered-columns-15.tsv.bz2', sep='\t', index=False)
    df_lang_filtered.to_csv(f'{path}/{lang}-filtered-columns.tsv.bz2', sep='\t', index=False)
    return df_lang_filtered


def filter_data_around_timestamp(df_filtered, df_treated, n_before=91, n_after=91,
                                 page_title_column='page_title_historical_norm', ts_column='event_timestamp',
                                 path='.', filename='mw_treated_90.tsv.bz2'):
    spell_mask = df_treated.type.isin(['spell-not-requested', 'rfpp-protected'])
    df_treated['treated_id'] = df_treated.index
    df_spells = df_treated[spell_mask]
    df_others = df_treated[~spell_mask].copy()
    df_others.request_timestamp = pd.to_datetime(df_others.request_timestamp, utc=True)

    # ======== NOW MERGE
    # A “forward” search selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    # selects the first row in the protection dataframe whose start date is greater than or equal to the event timestamp
    df_merged_before_start = pd.merge_asof(df_filtered, df_spells.sort_values('start'),
                                           left_by=page_title_column, right_by='norm_title',
                                           left_on=ts_column, right_on='start', allow_exact_matches=False,
                                           tolerance=pd.Timedelta(days=n_before), direction='forward').dropna(
        subset=['norm_title'])
    df_merged_before_start['is_over'] = False
    df_merged_before_start['is_active'] = False
    # A “backward” search selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key.
    # selects the first row in the protection dataframe whose end date is less than or equal to the event timestamp
    df_merged_after_end = pd.merge_asof(df_filtered, df_spells.sort_values('end'),
                                        left_by=page_title_column, right_by='norm_title',
                                        left_on=ts_column, right_on='end', allow_exact_matches=False,
                                        tolerance=pd.Timedelta(days=n_after), direction='backward').dropna(
        subset=['norm_title'])
    df_merged_before_start['is_active'] = False
    df_merged_before_start['is_over'] = True

    # A “backward” search selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key.
    # selects the first row in the protection dataframe whose start date is less than or equal to the event timestamp
    # ... then afterwards filter for event_timestamp <= end to actually get only rows within protection
    df_merged_between = pd.merge_asof(df_filtered, df_spells.sort_values('start'),
                                      left_by=page_title_column, right_by='norm_title',
                                      left_on=ts_column, right_on='start', allow_exact_matches=True,
                                      direction='backward').dropna(subset=['norm_title']).query(f'{ts_column} <= end')
    df_merged_before_start['is_active'] = True
    df_merged_before_start['is_over'] = False

    df_merged_spells = pd.concat([df_merged_before_start, df_merged_between, df_merged_after_end])

    # now merge others
    df_before_nonspell = pd.merge_asof(df_filtered, df_others.sort_values('request_timestamp'),
                                       left_by=page_title_column,
                                       right_by='norm_title', left_on='event_timestamp', right_on='request_timestamp',
                                       allow_exact_matches=False, tolerance=pd.Timedelta(days=n_before),
                                       direction='forward').dropna(subset=['norm_title'])
    df_before_nonspell['is_active'] = False
    df_before_nonspell['is_over'] = False

    df_after_nonspell = pd.merge_asof(df_filtered, df_others.sort_values('request_timestamp'),
                                      left_by=page_title_column, right_by='norm_title',
                                      left_on=ts_column, right_on='request_timestamp', allow_exact_matches=False,
                                      tolerance=pd.Timedelta(days=n_after * 2), direction='backward').dropna(
        subset=['norm_title'])
    df_before_nonspell['is_active'] = True
    df_before_nonspell['is_over'] = True

    df_nonspell_merged = pd.concat([df_before_nonspell, df_after_nonspell])
    df_all_merged = pd.concat([df_merged_spells, df_nonspell_merged])

    # f'../{PRE_PATH}/log_merged_90_new.tsv.bz2
    if filename:
        df_all_merged.to_csv(f'{path}/{filename}.tsv.bz2', sep='\t', index=False)
    return df_all_merged


def compute_treated_fields(df_treated):
    df_treated.start, df_treated.end = pd.to_datetime(df_treated.start, utc=True), \
        pd.to_datetime(df_treated.end, utc=True)
    df_treated['duration'] = df_treated.end - df_treated.start
    df_treated['duration_days'] = df_treated['duration'].round('1D').dt.days
    df_treated['grouped_days'] = df_treated.duration_days.apply(combine_ranges)
    return df_treated


def compute_mw_fields(df_mw):
    df_mw.page_namespace_historical = pd.to_numeric(df_mw.page_namespace_historical, errors='coerce')
    df_mw.event_timestamp = pd.to_datetime(df_mw.event_timestamp, utc=True)
    df_mw.request_timestamp = pd.to_datetime(df_mw.request_timestamp, utc=True)
    df_mw.start = pd.to_datetime(df_mw.start, utc=True)
    df_mw.end = pd.to_datetime(df_mw.end, utc=True)
    df_mw['page_namespace_historical'] = pd.to_numeric(df_mw['page_namespace_historical'], errors='coerce')
    df_mw['page_namespace_historical_content'] = (df_mw['page_namespace_historical'] % 2) == 0

    df_mw.revision_is_identity_revert = df_mw.revision_is_identity_revert.str.title() == 'True' \
        if df_mw.revision_is_identity_revert.dtype != bool else df_mw.revision_is_identity_revert
    df_mw.revision_text_bytes_diff = pd.to_numeric(df_mw.revision_text_bytes_diff, errors='coerce')

    df_mw['is_pp'] = (df_mw.event_timestamp >= df_mw.start) & (
            df_mw.event_timestamp <= df_mw.end)
    df_mw['was_pp'] = (df_mw.event_timestamp > df_mw.end)

    df_mw['start_diff'] = (df_mw.event_timestamp - df_mw.start)
    df_mw['end_diff'] = (df_mw.event_timestamp - df_mw.end)

    df_mw['start_diff_day'] = df_mw['start_diff'].dt.days
    df_mw['end_diff_day'] = df_mw['end_diff'].dt.days

    # for requests
    spell_mask = df_mw.type.isin(['spell-not-requested', 'rfpp-protected'])
    df_mw['is_pp_req'] = (df_mw.event_timestamp >= df_mw.request_timestamp)

    df_mw['request_diff'] = (df_mw.event_timestamp - df_mw.request_timestamp)
    df_mw['request_diff_day'] = df_mw['request_diff'].dt.days

    df_mw['decision_diff'] = (df_mw.event_timestamp - df_mw.request_timestamp)
    df_mw['decision_diff_day'] = df_mw['request_diff'].dt.days
    df_mw = compute_user_group(df_mw)
    return df_mw


def compute_basic_fields(df_mw, df_treated):
    df_treated = compute_treated_fields(df_treated)
    df_mw = compute_mw_fields(df_mw)

    return df_mw, df_treated


def compute_metrics(df_grp):
    # df_users = df_grp.groupby('event_user_text_historical').count()
    return pd.Series({'unique_editors': df_grp['event_user_text_historical'].nunique(),
                      'revisions': len(df_grp),
                      'revisions_content': df_grp['page_namespace_historical_content'].sum(),
                      # 'revisions_editor_median': df_users.median(),
                      # 'revisions_editor_mean': df_users.mean(),
                      'identity_reverts': df_grp['revision_is_identity_revert'].sum(),
                      'identity_reverts_content': (df_grp['revision_is_identity_revert'] &
                                                   df_grp['page_namespace_historical_content']).sum(),
                      'revision_text_bytes_diff_sum': df_grp['revision_text_bytes_diff'].sum(),
                      'revision_text_bytes_diff_median': df_grp['revision_text_bytes_diff'].median(),
                      'revision_text_bytes_diff_mean': df_grp['revision_text_bytes_diff'].mean()})

    # why not do it like this?
    '''
    .agg(
                revisions=('is_identity_revert', 'size'),
                identity_reverts=('is_identity_revert', 'sum'),
                identity_reverted=('is_identity_reverted', 'sum'),
                revision_text_bytes_diff_abs=('revision_text_bytes_abs', 'sum'),
                revision_text_bytes_diff_sum=('revision_text_bytes_diff', 'sum'),
                articlequality=('q_sum', 'mean'),
                damaging=('damaging', 'mean'),
                goodfaith=('goodfaith', 'mean'),
                unique_editors=('event_user_text_historical', 'nunique'),
                anon_revisions=('anonymous', 'sum'),
                admin_revisions=('admin', 'sum'),
                conf_revisions=('confirmed', 'sum')
            )
    '''


def compute_articlequality(df_grp):
    return pd.Series({'articlequality_last': df_grp.loc[df_grp['event_timestamp'].idxmax(), 'Q_score'],
                      'articlequality_first': df_grp.loc[df_grp['event_timestamp'].idxmax(), 'Q_score'],
                      'articlequality_median': df_grp['Q_score'].median(),
                      'articlequality_mean': df_grp['Q_score'].mean(),
                      'articlequality_max': df_grp['Q_score'].max()})


def compute_daily_metrics_with_revision_ids(df_grp):
    # print(df_grp['revision_text_bytes'], df_grp['Q_score'])
    # print(df_grp['Q_score'].fillna(-1).idxmax())
    # print(df_grp[['revision_id', 'Q_score', 'revision_text_bytes']])
    # print(df_grp.loc[df_grp['revision_text_bytes'].fillna(-1).idxmax()])
    return pd.Series({'revisions': len(df_grp),
                      'revision_id_max_quality': np.nan if len(df_grp) == 0 else df_grp.loc[
                          df_grp['Q_score'].fillna(-1).idxmax()].revision_id,
                      'revision_id_max_size': np.nan if len(df_grp) == 0 else df_grp.loc[
                          df_grp['revision_text_bytes'].fillna(-1).idxmax()].revision_id,
                      'revision_diff_bytes_sum': df_grp['revision_text_bytes_diff'].sum(),
                      'goodfaith': df_grp['prob_gf'].mean(),
                      'damaging': df_grp['prob_dmg'].mean(),
                      'goodfaith_count': df_grp['is_goodf'].sum().astype(int),
                      'damaging_count': df_grp['is_dmg'].sum().astype(int),
                      'articlequality_max': df_grp['Q_score'].max(),
                      'identity_reverts': df_grp['revision_is_identity_revert'].sum().astype(int),
                      'identity_reverted': df_grp['revision_is_identity_reverted'].sum().astype(int),
                      'page_size_max': df_grp['revision_text_bytes'].max(),
                      'page_age_max': df_grp['page_revision_count'].max()})


def compute_daily_metrics(df_grp):
    # print(df_grp['revision_text_bytes'], df_grp['Q_score'])
    if pd.isna(df_grp['Q_score'].max()) & \
            (~pd.isna(df_grp['Q_score'].iloc[-1])):
        print(df_grp[['treated_id', 'revision_id', 'Q_score',
                      'revision_text_bytes']])
    return pd.Series({'revisions': len(df_grp),
                      'goodfaith': df_grp['prob_gf'].mean(),
                      'damaging': df_grp['prob_dmg'].mean(),
                      'revision_diff_bytes_sum': df_grp['revision_text_bytes_diff'].sum(),
                      'revision_diff_bytes_abs_sum': df_grp['revision_text_bytes_diff'].abs().sum(),
                      'revision_diff_bytes_mean': df_grp['revision_text_bytes_diff'].mean(),
                      'goodfaith_count': df_grp['is_goodf'].sum().astype(int),
                      'damaging_count': df_grp['is_dmg'].sum().astype(int),
                      'articlequality_max': df_grp['Q_score'].max(),
                      'articlequality_last': df_grp['Q_score'].dropna().iloc[-1] if
                      len(df_grp['Q_score'].dropna()) > 0 else np.nan,
                      'revision_id_last': df_grp.dropna(subset=['Q_score']).revision_id.iloc[-1] if
                      len(df_grp['Q_score'].dropna()) > 0 else np.nan,
                      'identity_reverts': df_grp['revision_is_identity_revert'].sum().astype(int),
                      'identity_reverted': df_grp['revision_is_identity_reverted'].sum().astype(int),
                      'page_size_max': df_grp['revision_text_bytes'].max(),
                      'page_size_last': df_grp['revision_text_bytes'].dropna().iloc[-1] if len(
                          df_grp['revision_text_bytes'].dropna()) > 0 else np.nan,
                      'unique_users': df_grp['event_user_text'].nunique(),
                      })


def compute_ores_metrics(df_grp):
    return pd.Series({'revisions': len(df_grp),
                      'goodfaith': df_grp['prob_gf'].mean(),
                      'damaging': df_grp['prob_dmg'].mean(),
                      'goodfaith_count': df_grp['is_goodf'].sum(),
                      'damaging_count': df_grp['is_dmg'].sum(),
                      # 'articletopic_last': df_grp['artic'].last(),
                      # 'articletopic_first': df_grp['is_dmg'].first(),
                      'articlequality_last': df_grp.loc[df_grp['event_timestamp'].idxmax(), 'Q_score'],
                      'articlequality_first': df_grp.loc[df_grp['event_timestamp'].idxmax(), 'Q_score'],
                      'articlequality_median': df_grp['Q_score'].median(),
                      'articlequality_mean': df_grp['Q_score'].mean(),
                      'articlequality_max': df_grp['Q_score'].max()})
    # why not do it like this?
    '''
    .agg(
                revisions=('is_identity_revert', 'size'),
                identity_reverts=('is_identity_revert', 'sum'),
                identity_reverted=('is_identity_reverted', 'sum'),
                revision_text_bytes_diff_abs=('revision_text_bytes_abs', 'sum'),
                revision_text_bytes_diff_sum=('revision_text_bytes_diff', 'sum'),
                articlequality=('q_sum', 'mean'),
                damaging=('damaging', 'mean'),
                goodfaith=('goodfaith', 'mean'),
                unique_editors=('event_user_text_historical', 'nunique'),
                anon_revisions=('anonymous', 'sum'),
                admin_revisions=('admin', 'sum'),
                conf_revisions=('confirmed', 'sum')
            )
    '''


def agg_spell_metrics(df_edits, id_col='treated_id', start_diff_day_limit=-np.inf, end_diff_day_limit=np.inf):
    spell_mask = df_edits.type.isin(['spell-not-requested', 'rfpp-protected'])
    df_rel_spells = df_edits[spell_mask].copy()
    if id_col == 'treated_id':
        df_rel_spells.treated_id = df_rel_spells.treated_id.astype(int)
    print('rows', len(df_rel_spells))
    df_rel_spells = df_rel_spells[(df_rel_spells.start_diff_day >= start_diff_day_limit) &
                                  (df_rel_spells.end_diff_day <= end_diff_day_limit)]
    print('filtered', len(df_rel_spells))
    df_rel_spells.start_diff_day = df_rel_spells.start_diff_day.astype(int)
    df_rel_spells.end_diff_day = df_rel_spells.end_diff_day.astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        df_spell_group = df_rel_spells.query('(start_diff_day <= duration_days) & ~was_pp').groupby(
            [id_col, 'was_pp', 'is_pp', 'user_type', 'start_diff_day']).apply(compute_metrics)
        df_spell_group_after = df_rel_spells.query('was_pp & (end_diff_day >= 0)').groupby(
            [id_col, 'was_pp', 'is_pp', 'user_type', 'end_diff_day']).apply(compute_metrics)

    return df_spell_group.reset_index(), df_spell_group_after.reset_index()


def agg_spell_metrics_ores(df_edits, id_col='treated_id'):
    spell_mask = df_edits.type.isin(['spell-not-requested', 'rfpp-protected'])
    df_rel_spells = df_edits[spell_mask].copy()
    if id_col == 'treated_id':
        df_rel_spells.treated_id = df_rel_spells.treated_id.astype(int)

    df_rel_spells.start_diff_day = df_rel_spells.start_diff_day.astype(int)
    df_rel_spells.end_diff_day = df_rel_spells.end_diff_day.astype(int)

    df_rel_spells.is_goodf = df_rel_spells.is_goodf.astype(int)
    df_rel_spells.is_dmg = df_rel_spells.is_dmg.astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        df_spell_group = df_rel_spells.query('(start_diff_day <= duration_days) & ~was_pp').groupby(
            [id_col, 'was_pp', 'is_pp', 'user_type', 'start_diff_day']).apply(compute_ores_metrics)
        df_spell_group_after = df_rel_spells.query('was_pp & (end_diff_day >= 0)').groupby(
            [id_col, 'was_pp', 'is_pp', 'user_type', 'end_diff_day']).apply(compute_ores_metrics)

    return df_spell_group.reset_index(), df_spell_group_after.reset_index()


def agg_spell_metrics_by_request_ores(df_edits, id_col='treated_id', diff_column='request_diff_day',
                                      include_user_types=True, agg_func=compute_ores_metrics):
    # this is the same function than for page protections, but this time we wanna aggregate the requests (or user blocks)
    spell_mask = df_edits.type.isin(['spell-not-requested']) \
        if diff_column.startswith('request_diff_') else df_edits.type.isin(['rfpp-declined', 'rfpp-userint'])
    df_rel_spells = df_edits[~spell_mask].copy()
    df_rel_spells[diff_column] = df_rel_spells[diff_column].astype(int)

    columns = [id_col, 'is_pp_req', 'user_type', diff_column]
    if not include_user_types:
        columns.remove('user_type')
    # unfortunately the nanmeans warning is super annoying otherwise.... and unnecessary
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        df_spell_group_request = df_rel_spells.groupby(columns).apply(agg_func)

    return df_spell_group_request


def agg_spell_metrics_by_request(df_edits, id_col='treated_id', limit_days=(-np.inf, np.inf)):
    # this is the same function than for page protections, but this time we wanna aggregate the requests (or user blocks)
    spell_mask = df_edits.type.isin(['spell-not-requested'])
    df_rel_spells = df_edits[~spell_mask].copy()
    df_rel_spells = df_rel_spells[df_rel_spells.request_diff_day.between(limit_days[0], limit_days[-1])]
    print('filtered', len(df_rel_spells))

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        df_spell_group_request = df_rel_spells.groupby(
            [id_col, 'is_pp_req', 'user_type', 'request_diff_day']).apply(compute_metrics)

    return df_spell_group_request


def explode_grouped_requests(df_spell_group_filled, result_path, df_treated, id_col='treated_id'):
    df_spell_group_filled = df_spell_group_filled.reset_index().drop_duplicates(
        subset=['request_diff_day', id_col, 'is_pp_req', 'user_type'])

    index = pd.MultiIndex.from_product([df_spell_group_filled['request_diff_day'].unique(),
                                        df_spell_group_filled[id_col].unique(),
                                        [True, False], df_spell_group_filled['user_type'].unique()],
                                       names=['request_diff_day', id_col, 'is_pp_req', 'user_type'])
    # start_diff_day => difference from start date
    df_spell_group_filled_reindex = df_spell_group_filled.set_index(
        ['request_diff_day', id_col, 'is_pp_req', 'user_type']).reindex(index)[
        ['unique_editors', 'revisions', 'identity_reverts', 'revision_text_bytes_diff_sum',
         'revision_text_bytes_diff_median', 'revision_text_bytes_diff_mean', 'identity_reverts_content',
         'revisions_content']].fillna(0)

    df_spell_group_filled_reindex = df_spell_group_filled_reindex.reset_index().merge(
        df_treated.drop_duplicates(), on=id_col, how='left')

    df_spell_group_filled_reindex = df_spell_group_filled_reindex[
        (df_spell_group_filled_reindex.is_pp_req & (df_spell_group_filled_reindex.request_diff_day >= 0)) |
        (~df_spell_group_filled_reindex.is_pp_req & (df_spell_group_filled_reindex.request_diff_day < 0))]

    if result_path:
        df_spell_group_filled_reindex.to_csv(result_path, index=False)
    return df_spell_group_filled_reindex


def concat_beforeduring_and_after(df_group_before, df_group_after):
    df_complete = pd.concat([df_group_before, df_group_after])
    # vectorized way to compute start and end day into a single column
    df_complete['diff_day'] = (df_complete.start_diff_day.fillna(
        1) * df_complete.end_diff_day.fillna(1)).astype(int)
    df_complete['diff_date'] = df_complete.start + pd.to_timedelta(df_complete.diff_day, unit='D')
    return df_complete


def explode_grouped_spells(df_spell_group_filled, df_spell_group_end_filled, result_path, df_treated,
                           id_col='treated_id'):
    df_spell_group_filled = df_spell_group_filled.drop_duplicates(
        subset=['start_diff_day', id_col, 'was_pp', 'is_pp', 'user_type'])
    df_spell_group_end_filled = df_spell_group_end_filled.drop_duplicates(
        subset=['end_diff_day', id_col, 'was_pp', 'is_pp', 'user_type'])

    index = pd.MultiIndex.from_product([df_spell_group_filled['start_diff_day'].unique(),
                                        df_spell_group_filled[id_col].unique(), [False], [True, False],
                                        df_spell_group_filled['user_type'].unique()],
                                       names=['start_diff_day', id_col, 'was_pp', 'is_pp',
                                              'user_type'])
    # start_diff_day => difference from start date
    df_spell_group_filled_reindex = df_spell_group_filled.set_index(
        ['start_diff_day', id_col, 'was_pp', 'is_pp', 'user_type']).reindex(index)[
        ['unique_editors', 'revisions', 'identity_reverts', 'revision_text_bytes_diff_sum',
         'revision_text_bytes_diff_median', 'revision_text_bytes_diff_mean', 'identity_reverts_content',
         'revisions_content']].fillna(0)
    df_spell_group_filled_reindex = df_spell_group_filled_reindex.reset_index().merge(
        df_treated.drop_duplicates(),
        on=id_col, how='left')

    index_end = pd.MultiIndex.from_product([df_spell_group_end_filled['end_diff_day'].unique(),
                                            df_spell_group_end_filled[id_col].unique(), [True], [False],
                                            df_spell_group_end_filled['user_type'].unique()],
                                           names=['end_diff_day', id_col, 'was_pp', 'is_pp', 'user_type'])

    # end_diff_day => difference from start_date
    df_spell_group_filled_end_reindex = df_spell_group_end_filled.set_index(
        ['end_diff_day', id_col, 'was_pp', 'is_pp', 'user_type']).reindex(index_end)[
        ['unique_editors', 'revisions', 'identity_reverts', 'revision_text_bytes_diff_sum',
         'revision_text_bytes_diff_median', 'revision_text_bytes_diff_mean', 'identity_reverts_content',
         'revisions_content']].fillna(0)
    df_spell_group_filled_end_reindex = df_spell_group_filled_end_reindex.reset_index().merge(
        df_treated.drop_duplicates(), on=id_col, how='left')

    df_spell_group_filled_reindex_corr = df_spell_group_filled_reindex.copy()
    df_spell_group_filled_reindex_corr = df_spell_group_filled_reindex_corr[~(
            ~(df_spell_group_filled_reindex_corr.is_pp | df_spell_group_filled_reindex_corr.was_pp) & (
            df_spell_group_filled_reindex_corr.start_diff_day >= 0))]
    df_spell_group_filled_reindex_corr = df_spell_group_filled_reindex_corr[
        ~(df_spell_group_filled_reindex_corr.is_pp & (df_spell_group_filled_reindex_corr.start_diff_day < 0))]
    df_spell_group_filled_reindex_corr = df_spell_group_filled_reindex_corr[
        ~((df_spell_group_filled_reindex_corr.start_diff_day > df_spell_group_filled_reindex_corr.duration_days) & (
            ~df_spell_group_filled_reindex_corr.was_pp))]
    df_spell_group_filled_complete = concat_beforeduring_and_after(
        df_spell_group_filled_reindex_corr, df_spell_group_filled_end_reindex)

    if result_path:
        df_spell_group_filled_complete.to_csv(result_path, index=False)
    return df_spell_group_filled_complete


def classify_user(row):
    if pd.isna(row.event_user_id):
        return 'anonymous'
    elif not pd.isna(row.event_user_is_bot_by_historical):
        return 'bot'
    elif ~pd.isna(row.event_user_groups_historical) and (
            'sysop' in str(row.event_user_groups_historical) or 'bureaucrat' in str(row.event_user_groups_historical)):
        return 'admin'
    elif ~pd.isna(row.event_user_groups_historical) and ('extendedconfirmed' in str(row.event_user_groups_historical)):
        return 'extendedconfirmed'
    elif row.autoconfirmed or (
            ~pd.isna(row.event_user_groups_historical) and ('confirmed' in str(row.event_user_groups_historical))):
        return 'confirmed'
    else:
        return 'registered'


def compute_user_group(df_mw):
    df_mw['date_utc'] = pd.to_datetime(df_mw.event_timestamp, errors='coerce', utc=True)
    df_mw.event_user_registration_timestamp = pd.to_datetime(df_mw.event_user_registration_timestamp, errors='coerce',
                                                             utc=True)
    df_mw.event_timestamp = pd.to_datetime(df_mw.event_timestamp, errors='coerce', utc=True)
    df_mw.event_user_revision_count = pd.to_numeric(df_mw.event_user_revision_count, errors='coerce')

    print('Parsed fields')
    df_users_10_rev = df_mw[~pd.isna(df_mw.event_user_id) & (df_mw.event_user_revision_count >= 10)].copy()
    df_users_10_rev.event_user_registration_timestamp = pd.to_datetime(
        df_users_10_rev.event_user_registration_timestamp.fillna(pd.to_datetime('2006-01-01', utc=True)), utc=True)
    df_users_10_rev['account_age'] = df_users_10_rev.event_timestamp - df_users_10_rev.event_user_registration_timestamp
    df_users_10_rev['autoconfirmed'] = df_users_10_rev['account_age'] >= pd.Timedelta('96H')
    df_mw['autoconfirmed'] = df_mw.index.isin(set(df_users_10_rev[df_users_10_rev.autoconfirmed].index))
    print('computed autoconfirmed')
    df_mw['user_type'] = df_mw.apply(classify_user, axis=1)
    return df_mw
