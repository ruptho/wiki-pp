import warnings

import numpy as np
import pandas as pd
from mw.lib import title as mw_t

from scripts.util import find_replace_regex

warnings.filterwarnings("ignore", 'This pattern has match groups')


def get_activity_around_spells(df_mw_history, df_spells, save_path, n_days=90):
    # A “forward” search selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    # selects the first row in the protection dataframe whose start date is greater than or equal to the event timestamp
    df_merged_before_start = pd.merge_asof(df_mw_history, df_spells.sort_values('start'), right_by='title',
                                           left_by='page_title_historical_norm', left_on='date_utc', right_on='start',
                                           allow_exact_matches=False, tolerance=pd.Timedelta(days=n_days),
                                           direction='forward').dropna(subset=['title'])
    df_merged_before_start['pre'] = True
    df_merged_before_start['post'] = False
    # A “backward” search selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key.
    # selects the first row in the protection dataframe whose end date is less than or equal to the event timestamp
    df_merged_after_end = pd.merge_asof(df_mw_history, df_spells.sort_values('end'),
                                        left_by='page_title_historical_norm', right_by='title', left_on='date_utc',
                                        right_on='end', allow_exact_matches=False, tolerance=pd.Timedelta(days=n_days),
                                        direction='backward').dropna(subset=['title'])
    df_merged_after_end['pre'] = False
    df_merged_after_end['post'] = True
    # A “backward” search selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key.
    # selects the first row in the protection dataframe whose start date is less than or equal to the event timestamp
    # ... then afterwards filter for event_timestamp <= end to actually get only rows within protection
    df_merged_between = pd.merge_asof(df_mw_history, df_spells.sort_values('start'),
                                      left_by='page_title_historical_norm', right_by='title',
                                      left_on='date_utc', right_on='start', allow_exact_matches=True,
                                      direction='backward').dropna(subset=['title']).query('date_utc <= end')
    df_merged_between['pre'] = False
    df_merged_between['post'] = False
    df_merged_spells = pd.concat([df_merged_before_start, df_merged_between, df_merged_after_end])

    if save_path:
        # f'../{PRE_PATH}/log_merged_90_new.tsv.bz2'
        df_merged_spells.to_csv(save_path, sep='\t', index=False)
    return df_merged_spells


def load_protections(spell_paths='data/protections/spells.csv.zip', exclude_moves_during_pp=True,
                     exclude_subsequent=True, exclude_indefinite=True, exclude_ns=True, exclude_ts_after=None,
                     moves_file='data/supplementary/output-moves.tsv.zip', only_edits=True, cutoff=None):
    df_spells = pd.read_csv(spell_paths, parse_dates=['start', 'end', 'start_day', 'end_day'])

    # Fix time-constrained spells that are labeled incorrectly in the dataset
    df_spells.loc[df_spells.end < pd.to_datetime('2000-01-01', utc=True), 'end'] = df_spells.loc[
        df_spells.end < pd.to_datetime('2000-01-01', utc=True), 'end_day'] = pd.NaT
    print(f'All spells: {len(df_spells)}')

    """ 
    #This is only for measurement. can't use that for creation of final dataset because it will make the
    df_spells = df_spells[~pd.isna(df_spells.start) | (df_spells.start == pd.to_datetime('2000-01-01'))]
    print(f'Excluded already started protections: {len(df_spells)}')

    if cutoff:
        df_spells = df_spells[(df_spells.start >= pd.to_datetime(cutoff, utc=True))]
        print(f'Exclude protections starting after {cutoff}: {len(df_spells)}')
"""

    df_spells = df_spells[df_spells.type == 'edit'] if only_edits else df_spells
    print(f'Only edit spells: {len(df_spells)}')
    df_spells_all = df_spells.copy()

    if exclude_ns:
        df_spells = df_spells[~df_spells.title.astype(str).str.contains(
            '^(User|Talk|Template|Wikipedia|File|Portal|Wikipedia|MediaWiki|Module|Category|Help|Draft)(_[Tt]alk)?_?:')]
        print(f'Only ns0: {len(df_spells)}')

    if exclude_subsequent:
        df_spells.start = df_spells.start.fillna(
            pd.to_datetime('2000-01-01', utc=True))  # for correctly dropping already protected pages
        df_spells = df_spells.sort_values(['start', 'title'], ascending=True).drop_duplicates(
            subset=['title'], keep='first')
        print(f'Only first protections: {len(df_spells)}')

    df_spells = df_spells[~pd.isna(df_spells.start) | (df_spells.start == pd.to_datetime('2000-01-01'))]
    print(f'Excluded already started protections: {len(df_spells)}')

    if cutoff:
        df_spells = df_spells[(df_spells.start >= pd.to_datetime(cutoff, utc=True))]
        print(f'Exclude protections starting after {cutoff}: {len(df_spells)}')

    if exclude_moves_during_pp:
        df_moves = pd.read_csv(moves_file, sep='\t', quoting=0, doublequote=False,
                               header=0, names=['id', 'date', 'title_from', 'title_to'], parse_dates=['date'])
        df_moves['title_from'] = df_moves.title_from.apply(find_replace_regex)
        df_moves['date'] = pd.to_datetime(df_moves.date, utc=True)
        df_moves_during_spells = find_moves_during_protection(df_spells, df_moves)
        df_spells = df_spells[~df_spells.index.isin(df_moves_during_spells.old_id)]
        print(f'No Spells that were moved during protection: {len(df_spells)}')

    if exclude_indefinite:
        df_spells = df_spells[~pd.isna(df_spells.end)]
        print(f'No start or end date: {len(df_spells)}')

    if exclude_ts_after:
        df_spells['old_id'] = df_spells.index
        protected_within_n = pd.merge_asof(df_spells.sort_values('end'),
                                           df_spells_all[~pd.isna(df_spells_all.start)].sort_values('start'),
                                           by='title', left_on='end', right_on='start',
                                           tolerance=pd.Timedelta(exclude_ts_after), direction='forward').dropna()

        df_spells = df_spells[~df_spells.index.isin(set(protected_within_n.old_id))]
        print(f'No Spells that were again protected withing {exclude_ts_after}: {len(df_spells)}')
    df_spells.title = df_spells.title.apply(lambda title: mw_t.normalize(str(title)) if not pd.isna(title) else pd.NA)

    return df_spells


def find_moves_during_protection(df_spells, df_moves):
    df_merge = df_spells.copy()
    df_merge.end.fillna(pd.to_datetime('2222-02-22', utc=True), inplace=True)
    df_merge['old_id'] = df_merge.index
    df_merge.sort_values('start', ascending=True, inplace=True)
    df_merge_start = pd.merge_asof(df_merge, df_moves, left_on='start', right_on='date', left_by='title',
                                   right_by='title_to', direction='backward').dropna(subset=['date'])
    print('Start prot from moved', len(df_merge_start))
    df_merge.sort_values('end', ascending=True, inplace=True)

    df_merge_end = pd.merge_asof(df_merge, df_moves, left_on='end', right_on='date', left_by='title',
                                 right_by='title_from', direction='forward').dropna(subset=['date'])
    print('End prot from moved', len(df_merge_end))
    df_moves_during_spells = pd.concat([df_merge_start, df_merge_end])
    df_moves_during_spells.id = df_moves_during_spells.id.astype(int)
    print('In both ', len(set(df_merge_end) | set(df_merge_start)))
    return df_moves_during_spells


def filter_valid_rfpps(df_rfpp, cutoff='2023-03-01'):
    df_rfpp_rel = df_rfpp[
        (df_rfpp.decision_ts <= pd.to_datetime(cutoff, utc=True)) & ~pd.isna(df_rfpp.request_timestamp)].copy()
    df_rfpp_rel['decision_day'] = df_rfpp_rel.decision_ts.dt.date
    df_rfpp_dedup = df_rfpp_rel.sort_values('decision_ts', ascending=False).drop_duplicates(
        subset=['norm_title', 'request_timestamp'], keep='first').drop_duplicates(
        subset=['norm_title', 'decision_day'], keep='first').sort_values('request_timestamp')  # deduplicate responses

    print(f'Remove requests (and decisions) for same article with duplicate request days AND decision days.',
          (len(df_rfpp_rel) - len(df_rfpp_dedup)))
    # filter salted pages/create/move protection
    create_regex = r'(((creat(e|ion)|mov(e|ing)))(( semi| full)?[ -]protect(ion|ed)))|(un)?protect(ion)?( from) ' \
                   r'creat(e|ion)|repeatedly (re)?created'
    salt_regex = r'(Salt|[Uu]nsalt)(ing)?|SALT|creation salt'
    move_mask = ((~df_rfpp_dedup.request_level.str.lower().str.contains(create_regex) &
                  ~df_rfpp_dedup.request_text.fillna('').str.lower().str.contains(create_regex) &
                  ~df_rfpp_dedup.request_level.str.contains(salt_regex) &
                  ~df_rfpp_dedup.request_text.fillna('').str.contains(salt_regex)) &
                 ~df_rfpp_dedup.decision_type.isin(['Decision.CREATE', 'Decision.MOVE']))

    df_rfpp_rel_move = df_rfpp_dedup[move_mask]
    print(f'Removed Move/Salt/Create requests ({np.sum(~move_mask)}), remaining:', np.sum(move_mask))

    # filter pending changes
    pend_change_mask = ((df_rfpp_rel_move.decision_type == 'Decision.PEND_CHANGE') |
                        ((df_rfpp_rel_move.decision_type == 'Decision.HANDLED')))
    df_rfpp_nopendchange = df_rfpp_rel_move[~pend_change_mask]
    print((len(df_rfpp_rel_move), len(df_rfpp_rel), len(df_rfpp_nopendchange)))
    # filter out pages with multiple requests on same day?
    df_rfpp_nopendchange.sort_values('request_timestamp', ascending=False, inplace=True)
    print(f'Removed Pending Changes requests ({np.sum(pend_change_mask)}), remaining:', np.sum(~pend_change_mask))

    # plot_rfpp(df_rfpp_nopendchange, filename=None)

    return df_rfpp_nopendchange, df_rfpp_rel_move


def load_rfpp(rfpp_path='../data/requests/page_results.csv.zip'):
    df_rfpp = pd.read_csv(rfpp_path, parse_dates=['date', 'request_timestamp', 'decision_ts'])
    df_rfpp.norm_title = df_rfpp.norm_title.apply(lambda t: mw_t.normalize(str(t)) if not pd.isna(t) else pd.NA)
    len_parsed = len(df_rfpp)
    print('Overall parsed', len(df_rfpp))

    # remove page unprotection
    reduction_texts = 'reduc(tion in|ing|e) protection|(the )?protection is no( longer| t) necessary|(request(ing)?|' \
                      'considering (the )?|not-full) unprotection'
    unprot_filter = (df_rfpp.decision_status == 'DecisionClass.NOT_UNPROTECTED') | (
            df_rfpp.decision_type == 'Decision.UNPROTECTED') | df_rfpp.request_level.astype(str).fillna(
        '').str.startswith('Unprotection') | df_rfpp.request_text.astype(str).str.lower().str.contains(reduction_texts,
                                                                                                       regex=True) | df_rfpp.request_level.astype(
        str).str.lower().str.contains(reduction_texts, regex=True)
    df_unprotected = df_rfpp[unprot_filter].copy()
    df_rfpp = df_rfpp[~unprot_filter]
    print(f'Removed Unprotection requests ({len(df_unprotected)}), remaining:', len(df_rfpp))

    invalid_article_names = ['Name_of_page_you_are_requesting_an_edit_to', 'Example_Article_Name']
    invalid_mask = ~df_rfpp.norm_title.isin(invalid_article_names)
    df_rfpp = df_rfpp[invalid_mask]
    print(f'Removed misused templates ({np.sum(~invalid_mask)}), remaining:', np.sum(invalid_mask))

    filter_failed = pd.isna(df_rfpp.decision_status)
    df_rfpp = df_rfpp[~filter_failed]
    print(f'Removed failed parsing ({sum(filter_failed)}), remaining:', len(df_rfpp))

    filter_c_n_q = df_rfpp.decision_status == 'DecisionClass.COMM_OR_QUESTION'
    df_cnq = df_rfpp[filter_c_n_q].copy()

    df_rfpp = df_rfpp[~filter_c_n_q]
    print(f'Removed Comments, questions, or notes ({len(df_cnq)}), remaining:', len(df_rfpp))

    filter_ns = df_rfpp.norm_title.astype(str).str.contains(
        '^(User|Talk|Template|Wikipedia|File|Portal|Wikipedia|MediaWiki|Module|Category|WP|Help|Draft)(_[Tt]alk)?_?:')
    df_rfpp = df_rfpp[~filter_ns]
    print(f'Removed non-articles ({sum(filter_ns)}), remaining:', len(df_rfpp))

    df_requests = df_rfpp.drop_duplicates(['norm_title', 'request_timestamp'])
    print(f'Dropped duplicates', len(df_rfpp) - len(df_requests))
    print(
        f'{len(df_requests)} Requests for Protection with {len(df_rfpp)} decisions (dropped {len_parsed - len(df_requests)} requests)')
    # df_rfpp.decision_status.value_counts(dropna=False)

    df_rfpp['Status'] = df_rfpp.decision_status.apply(
        lambda s: 'Protected' if (s == 'DecisionClass.PROTECTED') or (s == 'DecisionClass.HANDLED') else 'Declined' if (
                s == 'DecisionClass.DECLINED') else
        'UserIssue' if (
                s == 'Decision.USER_BLOCK' or s == 'Decision.USER_REBLOCK' or s == 'DecisionClass.EDIT_WAR') else 'Existing' if s == 'DecisionClass.EXISTING' else 'Other')
    return df_rfpp


def merge_protected_to_accepted(df_spells, df_rfpp, merge_column_rfpp='decision_ts'):
    df_rfpp_prot = df_rfpp.dropna(subset=[merge_column_rfpp]).query('Status == "Protected"').sort_values(
        merge_column_rfpp)
    df_match_spell_to_rfpp = pd.merge_asof(df_spells, df_rfpp_prot, right_on=merge_column_rfpp, left_on='start',
                                           left_by='title', right_by='norm_title',
                                           tolerance=pd.to_timedelta('1D'), direction='nearest')
    df_match_rfpp_to_spells = pd.merge_asof(df_rfpp_prot, df_spells, left_on=merge_column_rfpp, right_on='start',
                                            right_by='title', left_by='norm_title', direction='nearest',
                                            tolerance=pd.to_timedelta('1D'))
    df_dropped_spells = df_match_spell_to_rfpp[pd.isna(df_match_spell_to_rfpp.request_level)]
    df_dropped_rfpp = df_match_rfpp_to_spells[pd.isna(df_match_rfpp_to_spells.level)]

    df_match_spell_to_rfpp.dropna(subset=['request_level'], inplace=True)
    df_match_rfpp_to_spells.dropna(subset=['level'], inplace=True)

    # why are so many rfpp missing -> cause we only consider first protections!!
    print(
        f'Matched {(len(df_match_spell_to_rfpp) / len(df_spells)) * 100:.2f}% of spells to accepted requests'
        f'({len(df_match_spell_to_rfpp)}/{len(df_spells)})')
    print(
        f'Matched {(len(df_match_rfpp_to_spells) / len(df_rfpp_prot)) * 100:.2f}% of accepted requests to spells'
        f'({len(df_match_rfpp_to_spells)}/{len(df_rfpp_prot)})')
    return df_match_spell_to_rfpp, df_match_rfpp_to_spells, df_dropped_spells, df_dropped_rfpp


def merge_rfpp_to_spells(df_rfpp_nopendchange, df_spells_rel, df_all_spells):
    df_rfpp_rel = df_rfpp_nopendchange.copy()
    df_rfpp_rel['request_timestamp_date'] = df_rfpp_rel.request_timestamp.dt.date
    df_rfpp_rel = df_rfpp_rel[df_rfpp_rel.Status.isin(['Protected', 'Declined', 'UserIssue']) &
                              (df_rfpp_rel.decision_ts <= pd.to_datetime('2022-12-31', utc=True))]

    df_spells_rel = df_spells_rel.copy()
    df_spells_rel = df_spells_rel[(df_spells_rel.start >= df_rfpp_rel.decision_ts.min()) &
                                  (df_spells_rel.end <= pd.to_datetime('2022-12-31', utc=True))]
    # only consider requests that have not been protected before
    merge_column_rfpp = 'decision_ts'

    df_rfpp_merge = df_rfpp_rel.query("Status == 'Protected'").sort_values(merge_column_rfpp)
    print(f'Protection spells: {len(df_rfpp_merge)}')

    # check if we can find spells for all requests and vice versa?
    # note the "df_all_spells" here
    print('====== Check if original spell dataset makes sense')
    merge_protected_to_accepted(df_all_spells.sort_values('start'), df_rfpp_merge)
    print('#### the "spells to accepted requests" should be very high')

    df_spells_to_rfpp, df_rfpp_to_spells, df_dropped_spells, df_dropped_rfpp = merge_protected_to_accepted(
        df_spells_rel, df_rfpp_merge)

    df_spell_protected = df_dropped_spells[['title', 'level', 'start', 'end']].copy().rename({'title': 'norm_title'},
                                                                                             axis=1)
    df_rfpp_accepted = df_rfpp_to_spells[
        ['norm_title', 'request_timestamp', 'decision_ts', 'level', 'start', 'end', ]].copy()
    df_rfpp_declined = df_rfpp_rel.query("Status == 'Declined'")[
        ['norm_title', 'request_timestamp', 'decision_ts']].copy()
    df_rfpp_userint = df_rfpp_rel.query("Status == 'UserIssue'")[
        ['norm_title', 'request_timestamp', 'decision_ts']].copy()
    df_rfpp_declined['old_rfpp_index'] = df_rfpp_declined.index
    df_rfpp_prev_protected = df_rfpp_declined.merge(df_all_spells, left_on='norm_title', right_on='title').query(
        'start < request_timestamp').sort_values('start', ascending=False).drop_duplicates(subset='old_rfpp_index')
    df_rfpp_declined_np = df_rfpp_declined[
        ~df_rfpp_declined.index.isin(df_rfpp_prev_protected['old_rfpp_index'].values)].drop(['old_rfpp_index'], axis=1)

    df_spell_protected['type'] = 'spell-not-requested'
    df_rfpp_accepted['type'] = 'rfpp-protected'
    df_rfpp_declined_np['type'] = 'rfpp-declined'
    df_rfpp_userint['type'] = 'rfpp-userint'
    df_experiment = pd.concat([df_rfpp_accepted, df_rfpp_declined_np, df_spell_protected, df_rfpp_userint])
    df_experiment.to_csv('data/experiment/df_treatment_groups.csv', index=False)
    return df_experiment


def combine_ranges(day, value_ranges=((4, 6), (8, 13), (15, 26), (27, 32), (58, 62), (88, 94), (181, 185), (360, 370))):
    if pd.isna(day):
        return pd.NA

    for r in value_ranges:
        if r[0] <= day <= r[1]:
            return f'{r[0]}-{r[1]}'
    return str(int(day))


def aggregate_count_timespans(df_plot, day_column='duration_days', y_column='value', group_cols=[], drop_grouped=True):
    # create a new column with the combined values
    df_plot['grouped_days'] = pd.to_numeric(df_plot[day_column], errors='coerce', downcast='integer').apply(
        combine_ranges)
    df_plot = df_plot.groupby(group_cols + ['grouped_days'], as_index=False)[y_column].sum().rename(
        {'grouped_days': day_column if drop_grouped else 'grouped_days'}, axis=1)
    return df_plot


def build_treatment_and_control_groups(df_rfpp_rel, df_rfpp_to_spells, df_dropped_spells):
    df_spell_protected = df_dropped_spells[['title', 'level', 'start', 'end']].copy().rename(
        {'title': 'norm_title'}, axis=1)
    df_spell_protected['type'] = 'spell-not-requested'

    df_rfpp_accepted = df_rfpp_to_spells[
        ['norm_title', 'request_timestamp', 'decision_ts', 'level', 'start', 'end', ]].copy()
    df_rfpp_accepted['type'] = 'rfpp-protected'

    df_rfpp_declined = df_rfpp_rel.query("Status == 'Declined'")[
        ['norm_title', 'request_timestamp', 'decision_ts']].copy()
    df_rfpp_declined['type'] = 'rfpp-declined'

    df_rfpp_userint = df_rfpp_rel.query("Status == 'UserIssue'")[
        ['norm_title', 'request_timestamp', 'decision_ts']].copy()
    df_rfpp_userint['type'] = 'rfpp-userint'

    return df_spell_protected, df_rfpp_accepted, df_rfpp_declined, df_rfpp_userint


'''
    if exclude_ts_after:
        df_spells['old_id'] = df_spells.index
        protected_within_n = pd.merge_asof(df_spells.sort_values('end'),
                                           df_spells_all[~pd.isna(df_spells_all.start)].sort_values('start'),
                                           by='title', left_on='end', right_on='start',
                                           tolerance=pd.Timedelta(exclude_ts_after), direction='forward').dropna()

        df_spells = df_spells[~df_spells.index.isin(set(protected_within_n.old_id))]
        print(f'No Spells that were again protected withing {exclude_ts_after}: {len(df_spells)}')
'''


def filter_treatment_and_control_for_experiment(df_groups, df_all_spells, threshold='90D', min_duration='23H'):
    df_temp = df_groups.copy()

    # First, remove additional interventions within 90 days of the end of the intervention
    # due to similar confounder concerns as for multiple protections.
    # end of intervention = request_timestamp (for control) or end (for protections)
    # Sort the DataFrame by 'norm_title' and 'threshold_start_timestamp'
    df_temp['threshold_end_timestamp'] = df_temp.apply(lambda r: r.request_timestamp if
    r.type in ['rfpp-declined', 'rfpp-userint'] else r.end, axis=1)
    df_temp['threshold_start_timestamp'] = df_temp.apply(lambda r: r.request_timestamp if
    r.type in ['rfpp-declined', 'rfpp-userint'] else r.start, axis=1)

    df_temp.sort_values(by=['norm_title', 'threshold_start_timestamp'], inplace=True)

    # Create a mask to identify duplicate rows based on your rule
    # Filter the DataFrame to remove duplicate rows
    mask = ((df_temp['norm_title'] == df_temp['norm_title'].shift(-1)) &
            (df_temp['threshold_start_timestamp'].shift(-1) - df_temp['threshold_end_timestamp']).le(
                pd.to_timedelta(threshold)))

    df_temp = df_temp[~mask]
    print(f'No Spells that had another intervention within {threshold}: {len(df_temp)} (dropped {mask.sum()})')

    # now, remove requests for previously protected pages
    df_rfpp_controls = df_temp[df_temp.type.isin(['rfpp-declined', 'rfpp-userint'])].copy()

    #df_rfpp_prev_protected = df_rfpp_controls[['norm_title', 'request_timestamp', 'treated_id']].merge(
    #    df_all_spells[['title', 'start']], left_on='norm_title', right_on='title', how='left').dropna(
    #    subset=['start']).query('start < request_timestamp').sort_values('start', ascending=False).drop_duplicates(
    #    subset='treated_id')

    #df_rfpp_controls_not_protected_before = df_rfpp_prev_protected[
    #    ~df_rfpp_prev_protected.treated_id.isin(df_rfpp_prev_protected['treated_id'].values)]

    #print(f'Removed {len(df_rfpp_controls) - len(df_rfpp_controls_not_protected_before)} control interventions '
     #     f'(declined, user intervention) due to previous protection '
      #    f'({len(df_rfpp_controls)}/{len(df_rfpp_controls_not_protected_before)}).')

    # now, remove protections with < 1 day of duration (do 23 hours as a threshold because sometimes they have 23:50h)
    df_protected_valid = df_temp[df_temp.type.isin(['rfpp-protected', 'spell-not-requested'])].copy()
    df_protected_valid['duration'] = df_protected_valid.end - df_protected_valid.start
    duration_mask = (df_protected_valid.duration < pd.to_timedelta(min_duration))
    df_protected_valid = df_protected_valid[~duration_mask]
    print(f'Removed {duration_mask.sum()} first spells < 1 Day of duration.')

    # now, stitch it together and call it a day (yay)
    df_new = pd.concat(
        [df_protected_valid.reset_index(drop=True),
         df_rfpp_controls.reset_index(drop=True)])

    return df_new
