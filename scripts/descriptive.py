from itertools import combinations

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from scipy.stats import kruskal, mannwhitneyu

from scripts.preprocessing import aggregate_count_timespans
from scripts.util import format_significant_in_table, COL_BLIND_PALETTE_3


def plot_percentage_bar_pageviews(sum_views_all_protected, sum_views_all,
                                  sum_views_top_protected, sum_views_top):
    sum_views_all_unprotected = sum_views_all - sum_views_all_protected
    sum_views_top_unprotected = sum_views_top - sum_views_top_protected
    bars = ("All\nArticles", "Top-\nArticles")

    weight_counts = {
        "Protected": np.array([sum_views_all_protected / sum_views_all, sum_views_top_protected / sum_views_top]),
        "Unprotected": np.array([sum_views_all_unprotected / sum_views_all, sum_views_top_unprotected / sum_views_top]),
    }

    fig, ax = plt.subplots(figsize=(1.5, 4))
    bottom = np.zeros(2)
    width, hatches = .8, ['xx', None, ]
    for i, (boolean, weight_count) in enumerate(weight_counts.items()):
        print(weight_count)
        p = ax.bar(bars, weight_count * 100, width, label=boolean, bottom=bottom,
                   color=[COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[-1]],
                   hatch=hatches[i], edgecolor="white")
        bottom += weight_count * 100
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage of Views (2022)', size='large')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15))
    leg = ax.get_legend()
    leg.legendHandles[0].set_facecolor('gray')
    leg.legendHandles[1].set_facecolor('gray')

    fig.savefig('figures/views_bars.pdf', bbox_inches='tight')
    return ax


def plot_views_only_left(views_all, views_prot_all, top_views, top_views_prot):
    import matplotlib.dates as mdates
    df_all = pd.concat([views_all, views_prot_all, top_views, top_views_prot])
    colors = [COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[0],
              COL_BLIND_PALETTE_3[-1], COL_BLIND_PALETTE_3[-1]]

    df_all['protected'] = df_all.type.str.contains('protected')
    df_all['date'] = pd.to_datetime(df_all['date'])
    fig, ax = plt.subplots(ncols=1, figsize=(6, 4))

    sns.lineplot(data=df_all, x='date', y='value', hue='type', palette=colors,
                 style='protected', dashes=['', (1, 1)], ax=ax)

    ax.set_ylabel('Pageviews (English Wikipedia)', fontsize='large')
    ax.set_xlabel('Date (in 2022)', size='large')
    ax.set_xlim(df_all['date'].min(), df_all['date'].max())

    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    # Get the handles and labels of the existing legends
    handles, labels = ax.get_legend_handles_labels()
    # Create a new legend with the desired labels and styles
    new_handles, new_labels = [], []
    overwrite_labels = ['All Articles', 'Top-Articles (1000)', 'Protected Views']
    for handle, label in zip(handles, labels):
        if not 'protected' in label and label not in ['type', 'protected', 'False']:
            new_handles.append(handle)
            new_labels.append(overwrite_labels.pop(0))  # Extract the last part of the label

    # Add the new legend with the customized labels and styles
    ax.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(.5, -0.175), ncol=3)
    fig.savefig('figures/views_single.pdf', bbox_inches='tight')
    return df_all


def plot_views(views_all, views_prot_all, top_views, top_views_prot):
    import matplotlib.dates as mdates
    df_all = pd.concat([views_all, views_prot_all, top_views, top_views_prot])
    colors = sns.color_palette('crest', n_colors=4)
    colors = [colors[0], colors[0], colors[-1], colors[-1]]

    df_all['protected'] = df_all.type.str.contains('protected')
    df_all['date'] = pd.to_datetime(df_all['date'])
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    sns.lineplot(data=df_all, x='date', y='value', hue='type', palette=colors, style='protected', ax=ax[0])

    ax_line_top_start, ax_line_top_end = (.995, .19), (0, 1)
    ax_line_bot_start, ax_line_bot_end = (.995, 0.033), (0, 0)

    conTop = ConnectionPatch(xyA=ax_line_top_start, coordsA=ax[0].transAxes, xyB=ax_line_top_end,
                             coordsB=ax[1].transAxes, linestyle=':', color='black', linewidth=1.5)
    conBot = ConnectionPatch(xyA=ax_line_bot_start, coordsA=ax[0].transAxes, xyB=ax_line_bot_end,
                             coordsB=ax[1].transAxes, linestyle=':', color='black', linewidth=1.5)
    plt.subplots_adjust(wspace=.13)
    fig.add_artist(conTop)
    fig.add_artist(conBot)
    df_zoomed = pd.concat([views_prot_all, top_views, top_views_prot])
    df_zoomed['protected'] = df_zoomed.type.str.contains('protected')

    sns.lineplot(data=df_zoomed, x='date', y='value', hue='type', palette=colors[1:],
                 style='protected', ax=ax[1], legend=None)
    # ax[1].set_ylim(bottom=0)

    # customize Plots from here
    rect = patches.Rectangle((0.01, 0.033), .985, .167, linewidth=2, edgecolor='black',
                             linestyle=':', facecolor='none', transform=ax[0].transAxes)
    ax[0].add_artist(rect)
    ax[1].set_ylabel('', fontsize='large')
    ax[0].set_ylabel('Pageviews (English Wikipedia)', fontsize='large')
    ax[1].set_xlabel('')
    ax[0].set_xlabel('')
    ax[0].set_xlim(df_all['date'].min(), df_all['date'].max())
    ax[1].set_xlim(df_all['date'].min(), df_all['date'].max())

    # ax[0].set_ylim(df_all['value'].min(), df_all['value'].max())
    # Set the locator
    # Specify the format - %b gives us Jan, Feb...

    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    for i_ax in [0, 1]:
        ax[i_ax].xaxis.set_major_locator(locator)
        ax[i_ax].xaxis.set_major_formatter(fmt)

    # Get the handles and labels of the existing legends
    handles, labels = ax[0].get_legend_handles_labels()
    # Create a new legend with the desired labels and styles
    new_handles, new_labels = [], []
    overwrite_labels = ['All Article Views', 'Top-Article Views (1000)', 'Protected Views']
    for handle, label in zip(handles, labels):
        if not 'protected' in label and label not in ['type', 'protected', 'False']:
            new_handles.append(handle)
            new_labels.append(overwrite_labels.pop(0))  # Extract the last part of the label

    # Add the new legend with the customized labels and styles
    ax[0].legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(1, -0.175), ncol=3, fontsize='large')
    fig.text(0.515, 0, 'Date (in 2022)', ha='center', fontsize='large')
    fig.savefig('figures/view_comp.pdf', bbox_inches='tight')
    return df_all


def plot_inverse_cdf(df_counts, days_limit=np.inf, ax=None, include_pre0=False):
    filter_string = f'duration_days <= {days_limit}{"" if include_pre0 else f" and duration_days >= 0"}'
    ax = sns.lineplot(data=df_counts.query(filter_string), x='duration_days', y='survival', ax=ax)
    # Get y value at x=160

    # Add marker for y value at x=160
    # ax.vlines(days_limit, ymin=0, ymax=y_value, color="red")
    xlim_min = -1 if include_pre0 else 0
    ax.set_xlim([xlim_min, days_limit])
    ax.set_ylim([0, 1])
    day_0 = df_counts.loc[df_counts["duration_days"] == 0, "survival"].values[0]
    ax.annotate(f"{day_0 * 100:.2f}% of PP\nlast < 1 day", (0.05, 0.025),
                ha="left", va="bottom", color='red', xycoords='axes fraction')

    day_val = df_counts.loc[df_counts["duration_days"] == days_limit, "survival"].values[0]
    ax.annotate(f"{day_val * 100:.2f}% of PP\n> {days_limit} days", (0.95, 0.025),
                ha="right", va="bottom", color='red', xycoords='axes fraction')


def plot_marker_day(ax, df_counts, day_info, days_limit, ha_left=True, bottom=True):
    day, day_text = day_info
    day_val = df_counts.loc[df_counts["duration_days"] == day, "survival"].values[0]
    x_position = (1 / days_limit) * day + (-0.15 if day > (days_limit - 5) else .15) + (
        .06 if day <= 3 else 0.1 if day < 5 else .06 if day < 10 else .06)
    # 0.8 + (.0 if (day < 3) else .06 if day < 5 else -.40 if day > 14 else -.2 if day > 7 else 0)
    #     ax.annotate(f"{day_text}\n{day_val * 100:.0f}%", xy=(day, day_val),
    y_position = 0.025
    if not bottom:
        if day < 7:
            y_position = min([day_val, 0.9])
        elif day < 15:
            y_position = day_val + .1
        elif day < 100:
            y_position = day_val + .15
        else:
            y_position = day_val + .20
    # (.975 if (day < 3) else .85 if day < 5 else .3 if day > 90 else .4 if day > 60 else .6 if day > 13
    # else .6 if day > 6 else 0)
    ax.annotate(f"{day_text}\n{day_val * 100:.0f}%", xy=(day, day_val),
                xytext=(x_position, y_position), ha="center", va="center", weight='bold',
                color='gray', xycoords='data', textcoords='axes fraction', fontsize='medium',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=2),
                bbox=dict(pad=-3 if (day > 5) and (day < days_limit - 5) else 0, facecolor="none", edgecolor="none"))


def plot_inverse_cdf_combined(df_counts, marks={0: '<1 Day', 3: '3 Days', 7: '1 Week', 14: '2 Weeks', 32: '1 Month',
                                                63: '2 Months', 94: '3 Months', 185: '6 Months', 366: '1 Year'},
                              days_limit=np.inf, include_pre0=False, figsize=(12.5, 5), agg_timespans=True, title=None,
                              ax=None, palette=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    filter_string = f'duration_days <= {days_limit}{"" if include_pre0 else f" and duration_days >= 0"}'
    df_counts_filt = df_counts.query(filter_string)
    x, y = df_counts_filt.duration_days.values, df_counts_filt['survival'].values

    norm = plt.Normalize(x.min(), x.max())
    cmap = sns.color_palette(palette="crest", as_cmap=True)
    colors = cmap(norm(x))
    segments = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=len(x) * [4])
    lc.set_array(x)
    lc.set_snap(True)
    ax.add_collection(lc)
    ax.autoscale()
    # https://stackoverflow.com/questions/65329041/how-to-make-a-multi-color-seaborn-lineplot-in-python
    # ax = sns.lineplot(x=df_counts_filt['duration_days'], y=df_counts_filt['survival'], ax=ax,
    #                  legend=None)

    # ax.vlines(days_limit, ymin=0, ymax=y_value, color="red")
    xlim_min = -1 if include_pre0 else 0
    ax.set_xlim([xlim_min, days_limit])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Ongoing Protection after Day n\n(1 - CDF)', fontsize='large')
    # ax.set_xlabel('Protection Duration (Days)', fontsize='large')
    # ax.annotate(f'Page Protection Events: {df_counts.value.sum()}', xy=(0.005, 0.95), color='blue',
    #            xycoords='axes fraction', ha="left", va="bottom")

    mark_infos = list(marks.items())
    for i, mark_info in enumerate(mark_infos[:-1]):
        plot_marker_day(ax, df_counts, mark_info, days_limit, bottom=False,
                        ha_left=mark_info[0] != mark_infos[-1][0])

    plot_marker_day(ax, df_counts, mark_infos[-1], days_limit, bottom=False, ha_left=False)
    ax.set_title(f'{title} ({int(df_counts.value.sum()):,} Protections)', fontsize='large')


def compute_cdf(df_counts, exclude_infinity=True):
    counts_non_inf = df_counts.query('duration_days < inf').copy() if exclude_infinity else df_counts.copy()
    counts_non_inf['cdf'] = (counts_non_inf.value.cumsum() / counts_non_inf.value.sum())
    counts_non_inf['survival'] = 1 - counts_non_inf['cdf']
    return counts_non_inf


def compute_protections_per_day(df_duration, group_col='duration_days', exclude_infinity=True):
    counts_per_day = df_duration[group_col].value_counts(dropna=False).rename('value').sort_index()
    counts_ongoing = counts_per_day[np.nan].copy() if np.nan in counts_per_day.index else 0
    counts_per_day = counts_per_day[counts_per_day.index.notnull()]
    counts_per_day = counts_per_day.dropna().reindex(index=np.arange(-1, counts_per_day.index.max() + 1), fill_value=0)
    if not exclude_infinity:
        counts_per_day.loc[np.inf] = counts_ongoing
    counts_per_day = counts_per_day.reset_index().rename({'index': group_col}, axis=1)
    return counts_per_day


def compute_duration_dfs(df_spells, type_col='type'):
    df_edit_spells = df_spells[df_spells[type_col] == 'edit'].copy()
    df_edit_spells['duration'] = pd.to_timedelta(df_edit_spells.end - df_edit_spells.start)
    df_edit_spells['duration_days'] = df_edit_spells['duration'].round('1D').dt.days
    df_prot_survival = df_edit_spells.copy()
    df_prot_survival['ended'] = ~pd.isna(df_prot_survival.end)
    df_prot_survival.loc[~df_prot_survival.ended, 'duration'] = pd.to_timedelta(
        pd.to_datetime('2023-03-15', utc=True) - df_prot_survival.loc[~df_prot_survival.ended, 'start'])
    df_prot_survival['duration_days'] = df_prot_survival['duration'].round('1D').dt.days

    # returns one normal df and one dataframe ready for survival analysis (in theory)
    return df_edit_spells, df_prot_survival


def plot_duration_hist_across_years(df, bins_limit, agg_timespans=True, exclude_infinity=True):
    years = range(2008, 2023)
    fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(20, 10))
    axs = axs.flatten()
    df_filtered = (df[~pd.isna(df.end)] if exclude_infinity else df).copy()
    unique_days = aggregate_count_timespans(compute_cdf(
        compute_protections_per_day(df_filtered, exclude_infinity=exclude_infinity),
        exclude_infinity=exclude_infinity))["duration_days"].unique()

    palette = dict(zip(unique_days, sns.color_palette(n_colors=len(unique_days))))
    for i, year in enumerate(years):
        df_year = df_filtered[df_filtered.start.dt.year == year].copy()
        df_day_counts = compute_protections_per_day(df_year, exclude_infinity=exclude_infinity)
        df_cdf_counts = compute_cdf(df_day_counts, exclude_infinity=exclude_infinity)
        # print(len(df_year), len(df_cdf_counts))

        plot_count_histogram(df_cdf_counts, bins_limit, xlabel=f'{year} PPs', ax=axs[i], agg_timespans=agg_timespans,
                             colors=palette)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
        if i % 5 != 0:
            axs[i].set_ylabel('', rotation=90)
    fig.tight_layout()


def plot_count_histogram(df, bins_limit=np.inf, title='', xlabel='Protection Duration (Days)', ax=None,
                         agg_timespans=True, colors=None):
    df_plot_all = df.copy()
    df_plot_all.duration_days = pd.to_numeric(df_plot_all.duration_days, errors='coerce')

    if agg_timespans:
        df_plot_all = aggregate_count_timespans(df_plot_all)

    if bins_limit < np.inf:
        df_plot = df_plot_all.nlargest(bins_limit, 'value') if bins_limit < np.inf else df

    def sortby(x):
        try:
            return int(x.split('-')[0])
        except ValueError:
            return float('inf')

    if colors is None:
        hue_order = list(sorted(list(df_plot.duration_days.unique()), key=sortby))
        palette = dict(zip(hue_order, sns.color_palette(palette="crest", n_colors=len(hue_order))))

    # print(xlabel, df_plot.duration_days.values, df_plot.value.values)
    ax = sns.barplot(data=df_plot.sort_values('duration_days'),
                     x='duration_days', y='value', ax=ax, palette=colors if colors else palette,
                     order=df_plot.sort_values('value', ascending=False).duration_days.values if colors else hue_order)
    plt.xticks(rotation=90)
    ax.set_title(title, fontsize='large')
    ax.set_ylabel('Number of Page Protections', fontsize='large')
    ax.set_xlabel(xlabel, fontsize='large')
    return df_plot if colors else df_plot, palette


def compute_cdf_and_plot(df_spells, days_limit=186, bins_limit=14, title='', exclude_infinity=True, figsize=(10, 5),
                         save_path='figures/pp_', filename=None):
    df_filtered = (df_spells[~pd.isna(df_spells.end)] if exclude_infinity else df_spells).copy()
    df_day_counts = compute_protections_per_day(df_filtered, exclude_infinity=exclude_infinity)
    df_cdf_counts = compute_cdf(df_day_counts, exclude_infinity=exclude_infinity)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, gridspec_kw={'width_ratios': [2.8, 1.2]})
    df_hist, palette = plot_count_histogram(df_cdf_counts, bins_limit=bins_limit, xlabel=None,
                                            title=f'Most-Frequent Durations', ax=axs[1])
    plot_inverse_cdf_combined(df_cdf_counts, days_limit=days_limit,
                              title=f'Page Protections Lasting n Days',
                              palette=palette, ax=axs[0])

    plt.subplots_adjust(wspace=0.3)
    fig.suptitle(f'{title}', fontsize='x-large')
    fig.text(0.6, -0.05, 'Protection Duration (Days)', ha='center', fontsize='large')
    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')
    # plt.tight_layout()
    return df_cdf_counts, df_hist


def bootstrap_metric(df, metric='median', n_iter=1000, n_threads=-1, is_pivotal=True):
    # assume values of interest are in first column, indices describe separate group
    # Pivotal vs. Percentile:
    # https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/77906546c17ee79eb6e64194175e82ed_MIT18_05S14_Reading24.pdf
    func_bs = bs_stats.median if metric == 'median' else bs_stats.mean
    df_bs = df.copy().to_frame()
    df_bs[metric] = df_bs['ci_low'] = df_bs['ci_high'] = None
    for index, r in df_bs.iterrows():
        bs_res = bs.bootstrap(np.array(r.iloc[0]), stat_func=func_bs, num_iterations=n_iter, num_threads=n_threads,
                              is_pivotal=is_pivotal)
        r[metric], r['ci_low'], r['ci_high'] = bs_res.value, bs_res.lower_bound, bs_res.upper_bound
    return df_bs


def plot_bootstrap_result(df_bs, xlabel='Year', metric='median', ax=None):
    # Create the point plot with error bars
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(df_bs.index, df_bs[metric], yerr=(df_bs[metric] - df_bs['ci_low'], df_bs['ci_high'] - df_bs[metric]),
                fmt='o', capsize=4)

    # Add labels and title
    ax.set_xlabel(f'{xlabel}')
    plt.setp(ax.get_xticklabels(), rotation='90', ha='left')
    ax.set_ylabel(f'{metric} PP Duration')
    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_title(f'Bootstrapped {metric.title()} with CIs')


def plot_rfpp(df_rfpp, figsize=(8, 2), path='figures/', filename='rfpp_final', dodge=False):
    fig, axs = plt.subplots(ncols=1, figsize=figsize)
    hue_order = ['Protected', 'Declined', 'UserIssue', 'Existing', 'Other']
    hue_labels = ['Protected', 'Declined', 'User(s) Blocked', 'Existing', 'Other']
    df_rfpp['Year'] = df_rfpp.decision_ts.dt.year.astype(int)
    counts_year_status = df_rfpp[['Year', 'Status']].value_counts().rename('Number of Requests').reset_index()
    # sns.barplot(data=counts_year_status, x='Year', y='Number of Responses', hue='Status', ax=axs[0], palette='crest',
    #            hue_order=hue_order)
    sns.barplot(data=counts_year_status, x='Year', y='Number of Requests', hue='Status', dodge=dodge,
                palette='crest', hue_order=hue_order, ax=axs)

    # Get the handles and labels from the legend
    handles, labels = axs.get_legend_handles_labels()

    # Create a new legend with the modified labels
    new_labels = [hue_labels[hue_order.index(label)] if label in hue_order else label for label in labels]
    axs.legend(handles, new_labels, title='', ncol=len(new_labels),
               loc='upper center', bbox_to_anchor=(0.45, -0.275))
    fig.suptitle(f'Requests for Page Protection by Year and Admin Decision ({len(df_rfpp):,} Requests)')
    # fig.tight_layout()
    if filename:
        fig.savefig(f'{path}/{filename}.pdf', bbox_inches='tight')


def plot_hist_across_years(df, val_col='duration_days', split_col='start_year', sharey=False, log_values=False,
                           height=3, aspect=1.5):
    # Set the style of the plots
    sns.set(style='ticks')
    # Create a FacetGrid with a histogram for each year
    df_plot = df.reset_index()
    df_plot[val_col] = df_plot[val_col].apply(lambda t: np.log1p(np.array(t)) if log_values else np.array(t))
    g = sns.FacetGrid(df_plot, col=split_col, col_wrap=5, sharey=sharey, height=height, aspect=aspect)
    g.map(plt.hist, val_col, bins=50 if log_values else 40)
    for i, ax in enumerate(g.axes.flat):
        mean_val, median_val = np.mean(df_plot[val_col][i]), np.median(df_plot[val_col][i]),
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=1, label='Mean')
        ax.axvline(median_val, color='g', linestyle=':', linewidth=1, label='Median')
        ax.text(mean_val, 0.75 * ax.get_ylim()[1], f'Avg\n{mean_val:.1f}', ha='left', va='center', color='r')
        ax.text(median_val, 0.75 * ax.get_ylim()[1], f'Med\n{median_val:.1f}', ha='right', va='center', color='g')

    # Add labels and title
    g.set_axis_labels(f'{val_col}{" (log)" if log_values else ""}', 'Frequency')
    g.fig.suptitle(f'Histogram of PP durations in {val_col}{" (log)" if log_values else ""} by {split_col}')
    plt.tight_layout()


def perform_kruskal_wallis(df_grouped):
    stat, p = kruskal(*df_grouped.values.tolist())

    if p < 0.05:
        print(f"Kruskal-Wallis test indicates a significance ({p})")

        # perform pairwise Mann-Whitney U tests and store results in a DataFrame
        groups, group_labels = df_grouped.values, df_grouped.index
        comparisons = list(combinations(range(len(groups)), 2))
        num_comparisons = len(comparisons)
        results = [[None] * len(group_labels) for _ in range(len(group_labels))]
        for i, j in comparisons:
            stat, p = mannwhitneyu(groups[i], groups[j])
            results[j][i] = p
            # results[j][i] = p
        # apply FDR correction to p-values
        # p_values = [p for p in flatten_list(results.values) if p is not None]
        # reject, p_corr, alphacSidak, alphacBonf = multipletests(p_values)

        results_df = pd.DataFrame(results, index=group_labels, columns=group_labels)
        results_df_bonf = results_df * len(comparisons)
        display(results_df_bonf.style.applymap(format_significant_in_table))


def plot_edits_per_protection_day(df, x_col, y_col, hue='article', log=True, days=7, figsize=(10, 5)):
    df_plot_sample = df.query(f'duration_days == {days}')
    article_titles = df_plot_sample.article.unique()
    # df_plot_sample = df_plot_sample[df_plot_sample.article.isin(set(np.random.choice(
    #    article_titles, n_sample if len(article_titles) < n_sample else len(article_titles))))]
    fig, ax = plt.subplots(figsize=figsize)
    if log:
        df_plot_sample[f'{y_col}_log'] = np.log1p(df_plot_sample[y_col])
    sns.lineplot(data=df_plot_sample, x=x_col, y=f'{y_col}_log' if log else y_col, hue=hue, ci=None,
                 palette=['darkgray'] * len(df_plot_sample.article.unique()), alpha=.1, legend=None, ax=ax)
    mean_df, median_df = df_plot_sample.groupby(x_col)[y_col].mean(), df_plot_sample.groupby(x_col)[y_col].median()
    ax.plot(np.log1p(mean_df) if log else mean_df, color='black', linestyle='--')
    ax.plot(np.log1p(median_df) if log else median_df, color='black', linestyle=':')


def group_values_for_plot(df_retrieved, value_col='edits', exclude_infinite=True):
    df = df_retrieved.copy()
    df = df[~pd.isna(df.spell_end)] if exclude_infinite else df
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df_spells_na = df[~pd.isna(df[value_col])].copy()

    df.date = pd.to_datetime(df.date)
    df_spells_na.spell_start = pd.to_datetime(df_spells_na.spell_start)
    df_spells_na.spell_end = pd.to_datetime(df_spells_na.spell_end)
    df_spells_na['date'] = pd.to_datetime(df_spells_na.date)
    df_spells_grouped = df_spells_na.groupby(
        ['article', 'spell_start', 'spell_end', 'date'])[value_col].sum().reset_index()

    df_spells_grouped['start_day'] = pd.to_datetime(df_spells_grouped.spell_start.dt.date)
    df_spells_grouped['end_day'] = pd.to_datetime(df_spells_grouped.spell_end.dt.date)
    df_spells_grouped['day_diff_start'] = pd.to_timedelta(
        df_spells_grouped['date'] - df_spells_grouped['start_day']).dt.days
    df_spells_grouped['day_diff_end'] = pd.to_timedelta(
        df_spells_grouped['date'] - df_spells_grouped['end_day']).dt.days

    df_spells_grouped['spell_duration'] = df_spells_grouped.spell_end - df_spells_grouped.spell_start
    df_spells_grouped['spell_duration_days'] = df_spells_grouped['spell_duration'].dt.days

    df_spells_grouped["pre_spell"] = df_spells_grouped["day_diff_start"] <= -1
    df_spells_grouped["post_spell"] = df_spells_grouped["day_diff_end"] >= 1
    df_spells_grouped[f'log_{value_col}'] = np.log1p(df_spells_grouped[value_col])
    df_spells_grouped[f'log_{value_col}'] = np.log1p(df_spells_grouped[value_col])

    df_pre_sum = df_spells_grouped[df_spells_grouped.pre_spell].groupby(
        ['article', 'spell_start'])[value_col].sum().reset_index().rename(columns={value_col: f"pre_sum_{value_col}"})
    df_spells_grouped = df_spells_grouped.merge(df_pre_sum, on=['article', 'spell_start'], how='left')
    # print(df_spells_grouped.head())
    df_spells_grouped[f'{value_col}_ratio'] = df_spells_grouped[value_col] / (
            df_spells_grouped[f"pre_sum_{value_col}"] + 1)
    return df_spells_grouped


def filter_limit_and_by_user_type_and_hue(df_res_gpd, day_column, hue_column, col_val, limit_days=None,
                                          user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered',
                                                      'extendedconfirmed'),
                                          include_only_hues=None, pagetitle_column='page_title_historical'):
    user_types = set(user_types) if user_types else None
    df_plot = df_res_gpd[df_res_gpd[day_column].between(limit_days[0], limit_days[1])] \
        if limit_days else df_res_gpd
    df_plot = df_plot[df_plot[hue_column].isin(include_only_hues)] if include_only_hues else df_plot
    df_plot = df_plot[df_plot.user_type.isin(user_types)] if user_types else df_plot

    # first have to combine user types
    return df_plot.groupby([pagetitle_column, 'start', 'is_pp', 'was_pp', 'level', day_column, hue_column])[
        col_val].sum().reset_index()


def filter_limit_and_groupby_user_type_and_hue(df_res_gpd, day_column, hue_column, col_val, limit_days=None,
                                               user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered',
                                                           'extendedconfirmed'),
                                               include_only_hues=None, pagetitle_column='page_title_historical',
                                               is_pp_col='is_pp', agg_func='sum'):
    user_types = set(user_types) if user_types else None
    df_plot = df_res_gpd[df_res_gpd[day_column].between(limit_days[0], limit_days[1])] \
        if limit_days else df_res_gpd

    df_plot = df_plot[df_plot[hue_column].isin(include_only_hues)] if include_only_hues else df_plot
    df_plot = df_plot[df_plot.user_type.isin(user_types)] if user_types else df_plot
    print('filtered rows, group now')
    # first have to combine user types
    return df_plot.groupby([pagetitle_column, 'start', is_pp_col, 'was_pp', 'level', day_column, hue_column, 'type'],
                           dropna=False)[col_val].agg(agg_func).reset_index()


def plot_pre_post_prot(df_res_gpd, col_val, hue_order=None, limit_days=None,
                       day_column='diff_day', hue_column='grouped_days',
                       ylabel='Mean %FIELD',
                       user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered', 'extendedconfirmed',
                                   'custom-comp'),
                       include_only_hues=None, metric='mean', log=True, filename=None, save_path='../figures/prepost_',
                       page_title_column='page_title_historical'):
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = filter_limit_and_groupby_user_type_and_hue(
        df_res_gpd, day_column, hue_column, col_val, limit_days, user_types, include_only_hues, page_title_column)

    # then we can mean
    df_pre_days = df_plot[~df_plot.is_pp & ~df_plot.was_pp].groupby([day_column, hue_column])[col_val].agg(
        metric).reset_index()
    df_post_days = df_plot[df_plot.was_pp].groupby([day_column, hue_column])[col_val].agg(metric).reset_index()
    # df_pre_days[day_column] = df_pre_days[day_column] - 1
    df_post_days[day_column] = df_post_days[day_column] + 1
    if log:
        df_pre_days[col_val] = np.log1p(df_pre_days[col_val])
        df_post_days[col_val] = np.log1p(df_post_days[col_val])
    hue_order = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else hue_order
    ax = sns.lineplot(data=df_pre_days, x=day_column, y=col_val, hue=hue_column, ax=ax,
                      hue_order=hue_order, palette="crest")
    sns.lineplot(data=df_post_days, x=day_column, y=col_val, hue=hue_column, ax=ax, legend=None,
                 hue_order=hue_order, palette="crest")
    # Set the range for the gap
    gap_start = -1
    gap_end = 1
    plt.legend(fontsize='large', title='PP Duration (Days)', loc='center left', bbox_to_anchor=(1.01, 0.5))

    # Create a shaded region to represent the gap
    ax.axvspan(gap_start, gap_end, color='grey', alpha=0.33)
    ax.annotate('After PP', (0.725, 0.6), xytext=(0.55, 0.6), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('Before PP', (0.15, 0.6), xytext=(0.25, 0.6), textcoords='figure fraction', xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    # Calculate the midpoint coordinates of the data area
    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 7

    ax.annotate('Page Protection Phase (n Days)', (-0.6, y_mid), rotation=90, xycoords='data', color='gray',
                ha='left', va='bottom', fontsize='x-large')
    ax.set_xlabel('Difference (in Days) from PP Start (Before PP) and End (After PP)', fontsize='x-large')
    ax.set_ylabel(ylabel.replace('%FIELD', col_val) + ('' if not log else ' (log1p)'), fontsize='x-large')
    ax.set_xlim(limit_days)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')

    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_pre_days, df_post_days, fig


def plot_pre_during_protection(df_res_gpd, col_val, hue_order=None, limit_days=None,
                               day_column='diff_day', hue_column='grouped_days',
                               user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered', 'extendedconfirmed'),
                               include_only_hues=None, ylabel='%METRIC %FIELD per 24 Hours', metric='mean',
                               log=True, filename=None, save_path='figures/preduring_',
                               page_title_column='page_title_historical'):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = df_res_gpd[~df_res_gpd.was_pp]

    df_plot = filter_limit_and_groupby_user_type_and_hue(
        df_plot, day_column, hue_column, col_val, limit_days, user_types, include_only_hues, page_title_column)

    df_pre_days = df_plot[~df_plot.is_pp].groupby([day_column, hue_column])[col_val].agg(
        metric).reset_index()

    df_during_days = df_plot[df_plot.is_pp].groupby([day_column, hue_column])[col_val].agg(metric).reset_index()
    df_during_days['lowest_group_day'] = df_during_days[hue_column].str.split('-').str[0].astype(int)
    df_during_days = df_during_days[df_during_days[day_column] < df_during_days['lowest_group_day']]
    # df_pre_days[day_column] = df_pre_days[day_column] + 1
    # df_during_days[day_column] = df_during_days[day_column] + 1

    if log:
        df_during_days[col_val] = np.log1p(df_during_days[col_val])
        df_pre_days[col_val] = np.log1p(df_pre_days[col_val])

    hue_order = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else hue_order
    ax = sns.lineplot(data=pd.concat([df_pre_days, df_during_days]), x=day_column, y=col_val, hue=hue_column, ax=ax,
                      hue_order=hue_order, palette="crest")
    # sns.lineplot(data=df_during_days, x=day_column, y=col_val, hue=hue_column, ax=ax, legend=None,
    #             hue_order=hue_order, palette="crest")
    ax.annotate('During PP\n(Right-Censored)', (0.8, 0.5), xytext=(0.55, 0.5), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('Pre PP', (0.25, 0.5), xytext=(0.4, 0.5), textcoords='figure fraction', xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('PP Event', (0.531, 0.9), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large')

    ax.axvline(0, color='gray')
    ax.set_xlabel('Difference (in Days) from PP Start (Pre) and End (Post)', fontsize='x-large')
    ax.set_ylabel(
        ylabel.replace('%FIELD', col_val).replace('%METRIC', metric.title()) + ('' if not log else ' (log1p)'),
        fontsize='x-large')
    plt.legend().set_title('Protection Duration (Days)')

    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_pre_days, df_during_days


def plot_pre_during_protection_simple(df_res_gpd, col_val, hue_order=None, limit_days=None,
                                      day_column='start_diff_day', hue_column='grouped_days',
                                      include_only_hues=None, ylabel='Mean %FIELD', metric='mean',
                                      log=True, filename=None, save_path='figures/preduring_',
                                      exclude_after=False,
                                      page_title_column='page_title_historical'):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = df_res_gpd[~df_res_gpd.was_pp] if exclude_after and 'was_pp' in df_res_gpd.columns else df_res_gpd

    df_plot = filter_limit_and_groupby_user_type_and_hue(
        df_plot, day_column, hue_column, col_val, limit_days, None, include_only_hues, page_title_column)

    if log:
        df_plot[col_val] = np.log1p(df_plot[col_val])
    df_plot = df_plot.groupby([hue_column, day_column])[col_val].agg(metric).reset_index()
    hue_order = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else hue_order
    ax = sns.lineplot(data=df_plot, x=day_column, y=col_val, hue=hue_column, ax=ax,
                      hue_order=hue_order, palette="crest")
    if exclude_after:
        df_plot['lowest_group_day'] = df_plot[hue_column].str.split('-').str[0].astype(int)
        df_plot = df_plot[df_plot[day_column] < df_plot['lowest_group_day']]

    # sns.lineplot(data=df_during_days, x=day_column, y=col_val, hue=hue_column, ax=ax, legend=None,
    #             hue_order=hue_order, palette="crest")
    ax.annotate('During PP\n(Right-Censored)', (0.8, 0.75), xytext=(0.55, 0.75), xycoords='figure fraction',
                color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('Pre PP', (0.25, 0.75), xytext=(0.4, 0.75), textcoords='figure fraction', xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('PP Event', (0.531, 0.9), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large')

    ax.axvline(0, color='gray')
    ax.set_xlabel('Difference (in Days) from PP Start (Pre) and End (Post)', fontsize='x-large')
    ax.set_ylabel(
        ylabel.replace('%FIELD', col_val).replace('%METRIC', metric.title()) + ('' if not log else ' (log1p)'),
        fontsize='x-large')
    plt.legend().set_title('Protection Duration (Days)')

    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_plot


def plot_during_post_protection(df_res_gpd, col_val, hue_order=None, limit_days=None,
                                day_column='diff_day', hue_column='grouped_days',
                                user_types=(
                                        'admin', 'anonymous', 'bot', 'confirmed', 'registered', 'extendedconfirmed'),
                                include_only_hues=None, ylabel='Mean %FIELD per 24 Hours', metric='mean',
                                log=True, filename=None, save_path='figures/during_post'):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = df_res_gpd[df_res_gpd.was_pp | df_res_gpd.is_pp]
    df_plot = df_plot[df_plot[hue_column].isin(include_only_hues)] if include_only_hues else df_plot
    df_plot = df_plot[df_plot.user_type.isin(user_types)]

    df_plot = filter_limit_and_groupby_user_type_and_hue(
        df_plot, day_column, hue_column, col_val, None, user_types, include_only_hues)

    df_post_days = df_plot[df_plot.was_pp].groupby([day_column, hue_column])[col_val].agg(
        metric).reset_index()
    df_during_days = df_plot[df_plot.is_pp].groupby([day_column, hue_column])[col_val].agg(metric).reset_index()

    df_during_days['lowest_group_day'] = df_during_days[hue_column].str.split('-').str[0].astype(int)
    df_during_days = df_during_days[df_during_days[day_column] < df_during_days['lowest_group_day']]
    df_during_days[day_column] = df_during_days[day_column] - df_during_days['lowest_group_day']
    df_post_days[day_column] = df_post_days[day_column]
    # df_during_days[day_column] = df_during_days[day_column] + 1

    df_during_days = df_during_days[df_during_days[day_column] >= limit_days[0]] if limit_days else df_during_days
    df_post_days = df_post_days[df_post_days[day_column] <= limit_days[1]] if limit_days else df_during_days

    if log:
        df_during_days[col_val] = np.log1p(df_during_days[col_val])
        df_post_days[col_val] = np.log1p(df_post_days[col_val])

    hue_order = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else hue_order
    ax = sns.lineplot(data=pd.concat([df_during_days, df_post_days]), x=day_column, y=col_val, hue=hue_column, ax=ax,
                      hue_order=hue_order, palette="crest")
    # sns.lineplot(data=df_during_days, x=day_column, y=col_val, hue=hue_column, ax=ax, legend=None,
    #             hue_order=hue_order, palette="crest")
    ax.annotate('Post PP\n(Right-Censored)', (0.8, 0.7), xytext=(0.55, 0.7), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('During PP\n(Left-Censored)', (0.2, 0.7), xytext=(0.45, 0.7), textcoords='figure fraction',
                xycoords='figure fraction',
                color='gray', va='center', ha='right', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('End of PP', (0.575, 0.86), xycoords='figure fraction', color='gray', rotation=90,
                va='center', ha='right', fontsize='x-large')

    ax.axvline(0, color='gray')
    ax.set_xlabel('Difference (in Days) from PP Start (Pre) and End (Post)', fontsize='x-large')
    ax.set_ylabel(ylabel.replace('%FIELD', col_val) + ('' if not log else ' (log1p)'), fontsize='x-large')
    plt.legend().set_title('Protection Duration (Days)')

    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_during_days, df_post_days


def bootstrap_groups_daily(df, bs_col, day_col='diff_day', group_col='grouped_days', metric='mean', n_iter=100,
                           n_threads=-1, is_pivotal=True):
    # Pivotal vs. Percentile:
    # https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/77906546c17ee79eb6e64194175e82ed_MIT18_05S14_Reading24.pdf
    func_bs = bs_stats.median if metric == 'median' else bs_stats.mean
    bs_results = []

    if group_col:
        for (group, day_col, was_pp, is_pp), df_grp in df.groupby([group_col, day_col, 'was_pp', 'is_pp']):
            bs_res = bs.bootstrap(df_grp[bs_col].values, stat_func=func_bs, num_iterations=n_iter,
                                  num_threads=n_threads, is_pivotal=is_pivotal)
            bs_results.append([group, day_col, was_pp, is_pp, bs_res.value, bs_res.lower_bound, bs_res.upper_bound])
    else:
        for (day_col, was_pp, is_pp), df_grp in df.groupby([day_col, 'was_pp', 'is_pp']):
            print(f'Bootstrap {len(df_grp[bs_col].values)} for {day_col}')
            bs_res = bs.bootstrap(df_grp[bs_col].values, stat_func=func_bs, num_iterations=n_iter,
                                  num_threads=n_threads, is_pivotal=is_pivotal)
            bs_results.append(['all', day_col, was_pp, is_pp, bs_res.value, bs_res.lower_bound, bs_res.upper_bound])
            print('done')
            break
    return pd.DataFrame(bs_results,
                        columns=[group_col, day_col, 'was_pp', 'is_pp', f'{bs_col}_{metric}', 'CI_lower', 'CI_upper'])


def plot_pre_during_protection_points(df_res_ci, col_val, show_groups=None, hue_order=None, limit_days=None,
                                      day_column='start_diff_day_was_pp', hue_column='grouped_days',
                                      ylabel='Mean Edits per 24 hours (log1p)'):
    fig, ax = plt.subplots(figsize=(15, 10))
    df_plot = df_res_ci.copy()
    df_plot = df_plot[df_plot.grouped_days.isin(show_groups)] if show_groups else df_plot
    df_plot = df_plot[df_plot[day_column].between(limit_days[0], limit_days[1])] if limit_days else df_plot

    # Create the point plot with error bars
    # ax = sns.pointplot(x=day_column, y=col_val, hue=hue_column, data=df_plot, hue_order=hue_order, ci=None, ax=ax,
    #              palette='crest')
    colors = sns.color_palette('crest', len(hue_order))

    for i, cat in enumerate(hue_order):
        df_cat = df_plot[df_res_ci[hue_column] == cat]
        ax.errorbar(x=df_cat[day_column], y=df_cat[col_val], yerr=(df_cat.CI_lower, df_cat.CI_upper),
                    fmt='o', capsize=2, color=colors[i])  # , color=colors)

    # Set the legend labels
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels)

    ax.annotate('During PP\n(Right-Censored)', (0.8, 0.5), xytext=(0.6, 0.5), xycoords='figure fraction',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('Pre PP', (0.25, 0.5), xytext=(0.4, 0.5), textcoords='figure fraction', xycoords='figure fraction',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('PP Event', (0.531, 0.9), xycoords='figure fraction',
                va='center', ha='left', fontsize='x-large')

    ax.axvline(0, color='gray')
    ax.set_xlabel('Difference (in Days) from PP Start (Pre) and End (Post)')
    ax.set_ylabel(ylabel)
    plt.tight_layout()


def plot_views_with_percentage(views_all, views_prot_all, top_views, top_views_prot):
    import matplotlib.dates as mdates
    df_all = pd.concat([views_all, views_prot_all, top_views, top_views_prot])
    colors = sns.color_palette('crest', n_colors=4)
    colors = [colors[0], colors[0], colors[-1], colors[-1]]

    df_all['protected'] = df_all.type.str.contains('protected')
    df_all['date'] = pd.to_datetime(df_all['date'])
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    sns.lineplot(data=df_all, x='date', y='value', hue='type', palette=colors, style='protected', ax=ax[0])

    plt.subplots_adjust(wspace=.25)

    df_zoomed = pd.concat([views_prot_all, top_views, top_views_prot])
    df_zoomed = df_zoomed.merge(views_all, on='date', suffixes=('', '_all'), how='left')
    df_zoomed['protected'] = df_zoomed.type.str.contains('protected')
    df_zoomed.value = (df_zoomed.value / df_zoomed.views) * 100

    sns.lineplot(data=df_zoomed, x='date', y='value', hue='type', palette=colors[1:],
                 style='protected', ax=ax[1], legend=None)
    # ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel('Percentage of Total Article Views', fontsize='large')
    # customize Plots from here

    ax[0].set_ylabel('Pageviews (English Wikipedia)', fontsize='large')
    ax[1].set_xlabel('')
    ax[0].set_xlabel('')
    ax[0].set_xlim(df_all['date'].min(), df_all['date'].max())
    ax[1].set_xlim(df_all['date'].min(), df_all['date'].max())

    # ax[0].set_ylim(df_all['value'].min(), df_all['value'].max())
    # Set the locator
    # Specify the format - %b gives us Jan, Feb...

    locator = mdates.MonthLocator()  # every month
    fmt = mdates.DateFormatter('%b')

    for i_ax in [0, 1]:
        ax[i_ax].xaxis.set_major_locator(locator)
        ax[i_ax].xaxis.set_major_formatter(fmt)

    # Get the handles and labels of the existing legends
    handles, labels = ax[0].get_legend_handles_labels()
    # Create a new legend with the desired labels and styles
    new_handles, new_labels = [], []
    overwrite_labels = ['Total Article Views', 'Top-Article Views (1000)', 'Protected Views']
    for handle, label in zip(handles, labels):
        if not 'protected' in label and label not in ['type', 'protected', 'False']:
            new_handles.append(handle)
            new_labels.append(overwrite_labels.pop(0))  # Extract the last part of the label

    # Add the new legend with the customized labels and styles
    ax[0].legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(1, -0.175), ncol=3, fontsize='large')
    fig.text(0.515, 0, 'Date (in 2022)', ha='center', fontsize='large')
    fig.savefig('figures/view_comp_perc.pdf', bbox_inches='tight')
    return df_all


# ==== AGG PLOTS

def plot_pre_post_prot_by_type(df_res_gpd, col_val, hue_order=('spell-not-requested', 'rfpp-protected'),
                               limit_days=None,
                               day_column='diff_day', hue_column='grouped_days',
                               ylabel='Mean %FIELD per 24 Hours',
                               user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered', 'extendedconfirmed',
                                           'custom-comp'),  # that's one by me just for computation
                               include_only_hues=None, metric='mean', log=True, filename=None,
                               save_path='figures/prepost_',
                               page_title_column='page_title_historical'):
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = filter_limit_and_groupby_user_type_and_hue(
        df_res_gpd, day_column, hue_column, col_val, limit_days, user_types, include_only_hues, page_title_column)

    # then we can mean
    df_pre_days = df_plot[~df_plot.is_pp & ~df_plot.was_pp].groupby([day_column, 'type'])[col_val].agg(
        metric).reset_index()
    df_post_days = df_plot[df_plot.was_pp].groupby([day_column, 'type'])[col_val].agg(metric).reset_index()
    # df_pre_days[day_column] = df_pre_days[day_column] - 1
    df_post_days[day_column] = df_post_days[day_column] + 1
    if log:
        df_pre_days[col_val] = np.log1p(df_pre_days[col_val])
        df_post_days[col_val] = np.log1p(df_post_days[col_val])
    hue_order = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else hue_order
    ax = sns.lineplot(data=df_pre_days, x=day_column, y=col_val, hue='type', ax=ax,
                      hue_order=hue_order, palette="crest")
    sns.lineplot(data=df_post_days, x=day_column, y=col_val, hue='type', ax=ax, legend=None,
                 hue_order=hue_order, palette="crest")
    # Set the range for the gap
    gap_start = -1
    gap_end = 1
    plt.legend().set_title('Protection Duration (Days)')

    # Create a shaded region to represent the gap
    ax.axvspan(gap_start, gap_end, color='grey', alpha=0.33)
    ax.annotate('Post PP', (0.75, 0.5), xytext=(0.6, 0.5), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate('Pre PP', (0.25, 0.5), xytext=(0.4, 0.5), textcoords='figure fraction', xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    # Calculate the midpoint coordinates of the data area
    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 4

    ax.annotate('Page Protection Phase (n Days)', (0, y_mid), rotation=90, xycoords='data', color='gray',
                ha='center', fontsize='x-large')
    ax.set_xlabel('Difference (in Days) from PP Start (Pre) and End (Post)', fontsize='x-large')
    ax.set_ylabel(ylabel.replace('%FIELD', col_val) + ('' if not log else ' (log1p)'), fontsize='x-large')
    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_pre_days, df_post_days


def plot_pre_during_protection_for_spells(df_res_gpd, col_val,
                                          hue_order={'spell-not-requested': 'Without Request',
                                                     'rfpp-protected': 'With Request'},
                                          limit_days=(-30, 30), day_column='diff_day', hue_column='grouped_days',
                                          user_types=(
                                                  'admin', 'anonymous', 'bot', 'confirmed', 'registered',
                                                  'extendedconfirmed', 'custom-comp'),
                                          include_only_hues=None, ylabel='%METRIC %FIELD', metric='mean',
                                          log=True, filename=None, save_path='../figures/preduring_',
                                          page_title_column='page_title_historical', single_hue=None, agg_func='sum',
                                          figsize=(6, 6)):
    return plot_pre_during_protection_by_type(df_res_gpd, col_val, hue_order, limit_days, day_column, hue_column,
                                              user_types, include_only_hues, ylabel, metric, log, filename, save_path,
                                              page_title_column, 'is_pp', single_hue=single_hue, day_plus_one=True,
                                              legend_title='Protection Requested?', agg_func=agg_func, figsize=figsize)


def plot_pre_during_protection_for_requests(df_res_gpd, col_val,
                                            hue_order={'rfpp-declined': 'Declined', 'rfpp-userint': 'User(s) blocked',
                                                       'rfpp-protected': 'Protected'},
                                            limit_days=(-30, 30), day_column='request_diff_day',
                                            hue_column='grouped_days',
                                            user_types=(
                                                    'admin', 'anonymous', 'bot', 'confirmed', 'registered',
                                                    'extendedconfirmed', 'custom-comp'),
                                            include_only_hues=None, ylabel='%METRIC %FIELD', metric='mean',
                                            log=True, filename=None, save_path='figures/preduring_',
                                            page_title_column='page_title_historical', agg_func='sum',
                                            ignore_ended_protections=True, figsize=(6, 6),
                                            title=None):
    # df_res_gpd = df_res_gpd.copy()
    df_res_gpd['was_pp'] = (df_res_gpd[day_column] >= df_res_gpd.duration_days)  # poor man's was_pp
    print('computed was_pp')
    return plot_pre_during_protection_by_type(df_res_gpd, col_val, hue_order, limit_days, day_column, hue_column,
                                              user_types, include_only_hues, ylabel, metric, log, filename,
                                              save_path, page_title_column, 'is_pp_req', 'RfPP',
                                              'Pre', 'Post', True, 'Admin Decision', agg_func=agg_func,
                                              ignore_ended_protections=ignore_ended_protections, figsize=figsize,
                                              palette=[COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[2]],
                                              title=title)


def plot_pre_during_protection_by_type(df_res_gpd, col_val, hue_order,
                                       limit_days, day_column='diff_day', hue_column='grouped_days',
                                       user_types=('admin', 'anonymous', 'bot', 'confirmed', 'registered',
                                                   'extendedconfirmed', 'custom-comp'),
                                       include_only_hues=None, ylabel='%METRIC %FIELD per 24 Hours', metric='mean',
                                       log=True, filename=None, save_path='../figures/',
                                       page_title_column='page_title_historical', pp_column='is_pp',
                                       event_label='PP Start', before_label='Pre PP', during_label='During PP',
                                       day_plus_one=False, legend_title=None, single_hue=None, agg_func='sum',
                                       ignore_ended_protections=True, figsize=(6, 6), palette=COL_BLIND_PALETTE_3,
                                       title=None):
    fig, ax = plt.subplots(figsize=figsize)
    if ignore_ended_protections:
        df_plot = df_res_gpd[~df_res_gpd.was_pp]
        df_plot = filter_limit_and_groupby_user_type_and_hue(df_plot, day_column, hue_column, col_val, limit_days,
                                                             user_types, include_only_hues, page_title_column,
                                                             pp_column, agg_func=agg_func)
        print(len(df_plot))
        df_pre_days = df_plot[~df_plot[pp_column]].groupby([day_column, 'type'], dropna=False)[col_val].agg(
            metric).reset_index()

        df_during_days = df_plot
        df_during_days['lowest_group_day'] = df_during_days[hue_column].apply(
            lambda g: int(g.split('-')[0]) if not pd.isna(g) else np.inf)
        df_during_days = df_during_days[df_during_days[day_column] < df_during_days['lowest_group_day']]
        df_during_days = df_during_days[df_during_days[pp_column]].groupby([day_column, 'type'])[col_val].agg(
            metric).reset_index()
    else:
        # other approach, keep aggregating
        df_plot = filter_limit_and_groupby_user_type_and_hue(df_res_gpd, day_column, hue_column, col_val, limit_days,
                                                             user_types, include_only_hues, page_title_column,
                                                             pp_column,
                                                             agg_func=agg_func)
        print(len(df_plot))
        df_pre_days = df_plot[df_plot[day_column] < 0].groupby([day_column, 'type'], dropna=False)[col_val].agg(
            metric).reset_index()
        df_during_days = df_plot[df_plot[day_column] >= 0].groupby([day_column, 'type'], dropna=False)[col_val].agg(
            metric).reset_index()

    if day_plus_one:
        df_pre_days[day_column] = df_pre_days[day_column] + 1
        df_during_days[day_column] = df_during_days[day_column] + 1

    if log:
        df_during_days[col_val] = np.log1p(df_during_days[col_val])
        df_pre_days[col_val] = np.log1p(df_pre_days[col_val])

    hue_labels = [hue for hue in hue_order if hue in include_only_hues] if include_only_hues else list(hue_order)
    ax = sns.lineplot(data=pd.concat([df_pre_days, df_during_days]), x=day_column, y=col_val,
                      hue=None if single_hue else 'type', ax=ax,
                      hue_order=None if single_hue else hue_labels, style='type' if single_hue else None,
                      style_order=hue_labels if single_hue else None,
                      palette=palette, color=single_hue if single_hue else None)

    '''
    ax.annotate(f'{during_label}', (0.62, 0.4), xytext=(0.45, 0.4), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate(f'{before_label}', (0.18, 0.4), xytext=(0.26, 0.4), textcoords='figure fraction',
                xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))

    '''
    ax.annotate(f'{event_label}', (0.39, 0.3), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large')


    '''
    ax.annotate(f'{during_label}', (0.9, 0.6), xytext=(0.65, 0.6), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large', arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate(f'{before_label}', (0.2, 0.6), xytext=(0.35, 0.6), textcoords='figure fraction',
                xycoords='figure fraction',
                color='gray', va='center', ha='left', fontsize='x-large',
                arrowprops=dict(edgecolor='gray', arrowstyle='->', lw=1))
    ax.annotate(f'{event_label}', (0.565, 0.1425), xycoords='figure fraction', color='gray',
                va='center', ha='left', fontsize='x-large')
    '''
    ax.axvline(0, linestyle='--', color='gray')
    ax.set_xlabel(f'Difference (in Days) from {event_label}', fontsize='x-large')
    ax.set_ylabel(
        ylabel.replace('%FIELD', col_val).replace('%METRIC', metric.title()) + ('' if not log else ' (log1p)'),
        fontsize='x-large')
    legend = ax.legend(list(hue_order.values()), title=legend_title, loc='lower right', fontsize='large')
    #legend.set_visible(False)
    # loc = 'upper left',
    ax.set_xlim(limit_days)
    plt.xticks(fontsize='large')
    ax.set_title(title, fontsize='x-large')
    plt.yticks(fontsize='large')
    if filename:
        fig.savefig(f'{save_path}{filename}.pdf', bbox_inches='tight')

    return df_pre_days, df_during_days, ax


def dot_and_whisker(df, ylabel, figsize=(4, 1), label_dict=None, filename=None):
    # Create the dot-whisker plot with error bars only
    fig, ax = plt.subplots(figsize=figsize)
    df = df.copy()
    df['marker_style'] = (df['Pr(>|t|)'] < 0.05).replace({False: 'x', True: 'o'}).astype(str)
    df['line_color'] = (df['Pr(>|t|)'] < 0.05).replace({False: '#88be91', True: '#2c3171'}).astype(str)
    # Plot error bars as whiskers with customized markers
    for i, row in df.iterrows():
        label = [i] if not label_dict else [label_dict[i]]
        ax.errorbar([row['Estimate']], label, xerr=[row['Std. Error'] * 1.96], fmt=row['marker_style'],
                    color=row['line_color'], markersize=6, capsize=3, label='Estimate')

    # Add a vertical dotted line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ylimits = ax.get_ylim()
    print(ylimits)
    ax.set_ylim(ylimits[0] - .25, ylimits[1] + .25)
    ax.set_xlim(-0.05, 0.05)
    # Set labels for x and y axes44
    ax.set_xlabel('Aggregated Effect', fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    # Show the plot
    # plt.title('Dot-Whisker Plot of Regression Results')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    if filename:
        fig.savefig(filename, bbox_inches='tight')


def dot_and_whisker_qualitytopics(results_topic, period):
    sns.set(style="ticks")
    # Plot the dots for coefficients
    fig, ax = plt.subplots(figsize=(4, 4))
    data = results_topic.query(f'request_diff_week == {period}').sort_values(['Quality', 'Topic'])

    # Add a new column 'Marker_Style' based on the values in 'Marker_Column'
    data['Color'] = data['Quality'].map({'All': COL_BLIND_PALETTE_3[1], 'Low': COL_BLIND_PALETTE_3[2],
                                         'High': COL_BLIND_PALETTE_3[0]})
    data['Style'] = data['significant'].map({True: '-', False: '--'})

    ax = sns.pointplot(
        x='coef', y='Topic', hue='Quality', data=data, join=False,
        dodge=.5, errorbar=None, markers=['$A$', '$H$', '$L$'], palette=[
            COL_BLIND_PALETTE_3[1], COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[2]])

    ax.axvline(0, linestyle='--', color='gray')
    # Customize the plot
    plt.xlabel('Coefficient Values (Change in Quality)')
    plt.title(f'{period + 1} Week{"s" if period >= 1 else ""} After Request')

    legend1 = ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-', label='p < 0.05'),
                                 Line2D([0], [0], color='black', linestyle='--', label='p ≥ 0.05')],
                        loc='upper left', bbox_to_anchor=(1.05, .4))

    legend2 = ax.legend(title='Quality', loc='upper left', bbox_to_anchor=(1.097, .8))

    x_coords, y_coords = [], []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    data['x_coord'] = x_coords
    data['y_coord'] = y_coords
    # Adding error bars separately using Matplotlib's errorbar function
    data['errors'] = data['97.5 %'] - data['coef']

    for sig, df_sig in data.groupby('significant'):
        eb = ax.errorbar(df_sig.x_coord.values, df_sig.y_coord.values, xerr=df_sig.errors.values,
                         ecolor=df_sig.Color.values, fmt=' ', zorder=-1)
        if not sig:
            eb[-1][0].set_linestyle('--')

    # Add the custom legends to the plot
    plt.gca().add_artist(legend1)
    sns.set(style="ticks")
    ax.axhline(.5, color='black')
    ax.axhline(1.5, color='black')
    ax.axhline(2.5, color='black')
    ax.set_ylabel('')
    plt.yticks(rotation=45)
    fig.savefig(f'figures/quality_topics_week{period}.pdf', bbox_inches='tight')


def dot_and_whisker_qualitytopics_multi(results_topic, periods, file=None):
    sns.set(style="ticks")
    # Plot the dots for coefficients
    fig, axs = plt.subplots(ncols=len(periods), figsize=(6, 3))

    for i, period in enumerate(periods):
        data = results_topic.query(f'request_diff_week == {period}').sort_values(['Quality', 'Topic'])
        print(i, period, )
        # Add a new column 'Marker_Style' based on the values in 'Marker_Column'
        data['Color'] = data['Quality'].map({'All': COL_BLIND_PALETTE_3[1], 'Low': COL_BLIND_PALETTE_3[2],
                                             'High': COL_BLIND_PALETTE_3[0]})
        data['Style'] = data['significant'].map({True: '-', False: '--'})

        sns.pointplot(
            x='coef', y='Topic', hue='Quality', data=data, join=False,
            dodge=.5, errorbar=None, markers=['$A$', '$H$', '$L$'], palette=[
                COL_BLIND_PALETTE_3[1], COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[2]], ax=axs[i])
        if i < (len(periods) - 1):
            axs[i].legend().set_visible(False)

        axs[i].axvline(0, linestyle='--', color='gray')
        # Customize the plot
        axs[i].set_xlabel('')
        axs[i].set_title(f'Week {period}')

        x_coords, y_coords = [], []
        for point_pair in axs[i].collections:
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)

        data['x_coord'] = x_coords
        data['y_coord'] = y_coords
        # Adding error bars separately using Matplotlib's errorbar function
        data['errors'] = data['97.5 %'] - data['coef']

        for sig, df_sig in data.groupby('significant'):
            eb = axs[i].errorbar(df_sig.x_coord.values, df_sig.y_coord.values, xerr=df_sig.errors.values,
                                ecolor=df_sig.Color.values, fmt=' ', zorder=-1)
            if not sig:
                eb[-1][0].set_linestyle('--')

        sns.set(style="ticks")
        axs[i].axhline(.5, color='black')
        axs[i].axhline(1.5, color='black')
        axs[i].axhline(2.5, color='black')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        if i < len(periods) - 1:
            axs[i].set_yticklabels(axs[i].get_yticklabels(), rotation=45)
        else:
            axs[i].set_yticklabels([])

        if i == len(periods) - 1:
            legend1 = axs[i].legend(handles=[Line2D([0], [0], color='black', linestyle='-', label='p < 0.05'),
                                            Line2D([0], [0], color='black', linestyle='--', label='p ≥ 0.05')],
                                   loc='upper left', bbox_to_anchor=(1.05, .35), handlelength=1.25)
            legend2 = axs[i].legend(title='Quality', loc='upper left', bbox_to_anchor=(1.0875, .85))
            plt.gca().add_artist(legend1)
            axs[i].yaxis.set_ticklabels([])
        fig.text(0.5, -0.05, 'Coefficient Values (Change in Quality)', ha='center', fontsize='medium')
    plt.subplots_adjust(wspace=.1)

    if file:
        fig.savefig(file, bbox_inches='tight')


def dot_and_whisker_features_multi(results, file=None):
    sns.set(style="ticks")
    # Plot the dots for coefficients
    fig, ax = plt.subplots(figsize=(6, 20))
    custom_dict = {'Before': 0, '1 Week After': 1, '13 Weeks After': 2}
    data = results.sort_values(by=['Feature']).sort_values(by=['Week'], key=lambda x: x.map(custom_dict)).copy()

    # Add a new column 'Marker_Style' based on the values in 'Marker_Column'
    data['Color'] = data['Week'].map({'Before': COL_BLIND_PALETTE_3[1], '1 Week After': COL_BLIND_PALETTE_3[2],
                                         '13 Weeks After': COL_BLIND_PALETTE_3[0]})
    data['Style'] = data['significant'].map({True: '-', False: '--'})

    sns.pointplot(
        x='coef', y='Feature', hue='Week', data=data, join=False, #markers=['$A$', '$H$', '$L$'],
        dodge=.5, errorbar=None, palette=[
            COL_BLIND_PALETTE_3[1], COL_BLIND_PALETTE_3[2], COL_BLIND_PALETTE_3[0]], ax=ax)

    x_coords, y_coords = [], []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    ax.axvline(0, linestyle='--', color='gray')
    for f in range(len(data.Feature.unique())):
        axs[i].axhline(0.5 + 1 * f, color='black')

    data['x_coord'] = x_coords
    data['y_coord'] = y_coords
    # Adding error bars separately using Matplotlib's errorbar function
    data['errors'] = data['97.5 %'] - data['coef']

    for sig, df_sig in data.groupby('significant'):
        eb = ax.errorbar(df_sig.x_coord.values, df_sig.y_coord.values, xerr=df_sig['errors'].values,
                            ecolor=df_sig.Color.values, fmt=' ', zorder=-1)
        if not sig:
            eb[-1][0].set_linestyle('--')

    #sns.set(style="ticks")
    for i in range(len(data.Feature.unique())):
        ax.axhline(0.5 + 1*i, color='black')


   # ax.set_yticklabels(ax.get_yticklabels(), rotation=45)


    legend1 = ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-', label='p < 0.05'),
                                    Line2D([0], [0], color='black', linestyle='--', label='p ≥ 0.05')],
                           loc='upper left', bbox_to_anchor=(1.05, .35), handlelength=1.25)
    legend2 = ax.legend(title='Week', loc='upper left', bbox_to_anchor=(1.0875, .85))
    plt.gca().add_artist(legend1)
    plt.tight_layout()
    if file:
        fig.savefig(file, bbox_inches='tight')


def dot_and_whisker_quality_features_multi(feature_quality, periods=(0, 12), y_col='Feature', file=None):
    dict_labels = {
        'feature.wikitext.revision.chars_norm': 'Total Characters',
        'feature.len..datasource.wikitext.revision.words.._norm': 'Total Words',
        'feature.wikitext.revision.content_chars_norm': 'Content Characters',
        'feature.english.stemmed.revision.stems_length_norm': 'Stemmed Words',
        'feature.wikitext.revision.headings_by_level.2._norm': 'Level 2 Headings',
        'feature.wikitext.revision.headings_by_level.3._norm': 'Level 3 Headings',
        'feature.enwiki.revision.paragraphs_without_refs_total_length_norm': 'Paragraphs w/o Refs.',
        'feature.len..datasource.english.idioms.revision.matches.._norm': 'Idioms',
        'feature.len..datasource.english.words_to_watch.revision.matches.._norm': '"Words to Watch"',

        'feature.wikitext.revision.ref_tags_norm': 'References',
        'feature.enwiki.revision.cite_templates_norm': 'Citation Templates',
        'feature.enwiki.revision.cn_templates_norm': 'Citation Needed Templates',

        'feature.wikitext.revision.wikilinks_norm': 'Wikilinks',
        'feature.wikitext.revision.external_links_norm': 'External Links',
        'feature.enwiki.revision.category_links_norm': 'Category Links',
        'feature.enwiki.revision.image_links_norm': 'Image Links',

        'feature.enwiki.infobox_images_norm': 'Infobox Images',
        'feature.enwiki.revision.images_in_tags_norm': 'Images in Tags',
        'feature.enwiki.revision.images_in_templates_norm': 'Images in Templates',

        'feature.wikitext.revision.list_items_norm': 'List Items',
        'feature.enwiki.revision.infobox_templates_norm': 'Infobox Templates',
        'feature.enwiki.revision.image_template_norm': 'Image Templates',

        'feature.wikitext.revision.templates_norm': 'Revision Templates',
        'feature.enwiki.main_article_templates_norm': 'Article Templates',
        'feature.enwiki.revision.shortened_footnote_templates_norm': 'Footnote Templates',
        'feature.enwiki.revision.who_templates_norm': '"Who" Templates',
    }

    feature_label_order = {f: 100-i for i, f in enumerate(dict_labels.values())}
    feature_quality.Feature = feature_quality.Feature.replace(dict_labels)
    # Plot the dots for coefficients
    fig, axs = plt.subplots(ncols=len(periods), figsize=(8, 15))

    for i, period in enumerate(periods):
        data = feature_quality.query(f'Week == {period}').sort_values(
            by=['Feature'], key=lambda x: x.map(feature_label_order), kind='mergesort').sort_values(
            by=['Quality'], kind='mergesort')
        # Add a new column 'Marker_Style' based on the values in 'Marker_Column'
        data['Color'] = data['Quality'].map({'All': COL_BLIND_PALETTE_3[1], 'Low': COL_BLIND_PALETTE_3[2],
                                             'High': COL_BLIND_PALETTE_3[0]})
        data['Style'] = data['significant'].map({True: '-', False: '--'})

        sns.pointplot(x='coef', y=y_col, hue='Quality', data=data, join=False,
            dodge=.5, errorbar=None, markers=['$A$', '$H$', '$L$'], palette=[
                COL_BLIND_PALETTE_3[1], COL_BLIND_PALETTE_3[0], COL_BLIND_PALETTE_3[2]], ax=axs[i])

        if i < (len(periods) - 1):
            axs[i].legend().set_visible(False)

        axs[i].axvline(0, linestyle='--', color='gray')
        # Customize the plot
        axs[i].set_xlabel('')
        axs[i].set_title(f'Week {period}')

        x_coords, y_coords = [], []
        for point_pair in axs[i].collections:
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)

        data['x_coord'] = x_coords
        data['y_coord'] = y_coords
        # Adding error bars separately using Matplotlib's errorbar function
        data['errors'] = data['97.5 %'] - data['coef']

        for sig, df_sig in data.groupby('significant'):
            eb = axs[i].errorbar(df_sig.x_coord.values, df_sig.y_coord.values, xerr=df_sig.errors.values,
                             ecolor=df_sig.Color.values, fmt=' ', zorder=-1)
            if not sig:
                eb[-1][0].set_linestyle('--')

        for f in range(len(data.Feature.unique())):
            axs[i].axhline(0.5 + 1 * f, color='gray')

        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        if i < len(periods) - 1:
            axs[i].set_yticklabels(axs[i].get_yticklabels())#, rotation=90)
        else:
            axs[i].set_yticklabels([])

        if i == len(periods) - 1:
            legend1 = axs[i].legend(handles=[Line2D([0], [0], color='black', linestyle='-', label='p < 0.05'),
                                            Line2D([0], [0], color='black', linestyle='--', label='p ≥ 0.05')],
                                   loc='upper left', bbox_to_anchor=(0.075, -0.055), handlelength=1.25, ncol=2)
            legend2 = axs[i].legend(title='Quality', loc='upper left', bbox_to_anchor=(-1.11, -0.045), ncol=3)
            plt.gca().add_artist(legend1)
            axs[i].yaxis.set_ticklabels([])
        axs[i].set_ylim((-0.5, len(data.Feature.unique())-.5))
        axs[i].set_xlim((-0.018, 0.018))
    fig.text(0.5, 0.09, 'Coefficient Values (Change in Quality)', ha='center', fontsize='medium')
    plt.subplots_adjust(wspace=.1)

    if file:
        fig.savefig(file, bbox_inches='tight')


