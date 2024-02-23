from enum import Enum, auto as get_auto_enum_value
from dataclasses import dataclass
from itertools import combinations as get_combinations
import pandas as pd
from pathlib import Path
from random import sample as random_sample, seed as random_seed
import streamlit as st
from typing import Callable, Final, Iterator


win_value: Final = 1.0
tie_value: Final = 0.5
loss_value: Final = 0.0


class BaseMetric(Enum):
    points = get_auto_enum_value()
    yards = get_auto_enum_value()
    wins = get_auto_enum_value()
    epa = get_auto_enum_value()
    turnovers = get_auto_enum_value()
    success = get_auto_enum_value()


class FilterType(Enum):
    subject = get_auto_enum_value()
    scenario = get_auto_enum_value()
    base = get_auto_enum_value()
    adjusted_by = get_auto_enum_value()
    unit = get_auto_enum_value()
    per = get_auto_enum_value()


@dataclass()
class MetricGenerator:
    base: BaseMetric
    get_scores: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class WeekSplit:
    example_weeks: set[int]
    prediction_weeks: set[int]


@dataclass(frozen=True)
class Scenario:
    name: str
    is_neutral: bool
    get_plays: Callable[[], pd.DataFrame]


@dataclass(frozen=True)
class FilterInfo:
    prompt: str
    all_options: list[str]


@dataclass(frozen=True)
class Context:
    full_schedule: pd.DataFrame
    all_game_results: pd.DataFrame
    all_metric_generators: list[MetricGenerator]
    all_plays: pd.DataFrame
    final_plays: pd.DataFrame
    week_splits: list[WeekSplit]


def load_plays_from_file_for_year(year: int, *, win_prob: float) -> pd.DataFrame:
    csv_stem: Final = f'play_by_play_{year}.csv.gz'
    plays: Final = pd.read_csv(csv_stem, compression='gzip', low_memory=False)
    plays['unique_drive'] = plays['fixed_drive'] + ((plays['week'] - 1) * 1_000)
    plays['unique_series'] = plays['series'] + ((plays['week'] - 1) * 1_000)
    plays['neutral_scenario'] = (
        (plays.wp > win_prob) &
        (plays.wp < (1.0-win_prob)))
    return plays


def get_final_plays(plays: pd.DataFrame) -> pd.DataFrame:
    final_plays: Final = plays[plays.desc == 'END GAME'].copy()
    final_plays['home_wins'] = final_plays.apply(
        lambda row:
            win_value if row.home_score > row.away_score else
            tie_value if row.home_score == row.away_score else
            loss_value,
        axis=1)
    final_plays['away_wins'] = 1.0 - final_plays.home_wins
    return final_plays


def get_week_splits(count: int, *, randomize: bool) -> list[WeekSplit]:
    random_seed()

    upper_week_range: Final = maximum_week + 1
    all_weeks: Final = set(range(1, upper_week_range))
    week_combinations: list[WeekSplit] = []

    if randomize:
        while count > 0:
            count -= 1
            prediction_weeks: set[int] = set(random_sample(range(1, upper_week_range), prediction_week_count))
            week_combinations.append(WeekSplit(example_weeks=all_weeks - prediction_weeks, prediction_weeks=prediction_weeks))
    else:
        for combination in get_combinations(all_weeks, prediction_week_count):
            prediction_weeks: set[int] = set(combination)
            week_combinations.append(WeekSplit(example_weeks=all_weeks - prediction_weeks, prediction_weeks=prediction_weeks))
            if len(week_combinations) >= count:
                break

    return week_combinations


def get_schedule_from(final_plays: pd.DataFrame) -> pd.DataFrame:
    half_schedule: Final = final_plays[['home_team', 'week', 'away_team']]
    schedule: Final = pd.concat(
        [half_schedule, half_schedule.rename(columns={'home_team': 'away_team', 'away_team': 'home_team'})],
        axis=0)
    schedule.columns = ['team', 'week', 'opponent']
    schedule.sort_values(by=['team', 'week'], inplace=True)
    schedule.reset_index(drop=True, inplace=True)
    return schedule


def get_game_results_from(final_plays: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({
                'team': final_plays['home_team'],
                'week': final_plays['week'],
                'wins': final_plays['home_wins'], }),
            pd.DataFrame({
                'team': final_plays['away_team'],
                'week': final_plays['week'],
                'wins': final_plays['away_wins'], }), ],
        ignore_index=True)


def get_totals_from(final_plays: pd.DataFrame, useful_plays: pd.DataFrame, *, for_offense: bool) -> pd.DataFrame:
    plays_by_team: Final = useful_plays[
        [
            'posteam',
            'defteam',
            'week',
            'unique_drive',
            'unique_series',
            'pass_touchdown',
            'rush_touchdown',
            'field_goal_result',
            'two_point_conv_result',
            'extra_point_result',
            'yards_gained',
            'series_success',
            'epa',
            'fumble_lost',
            'interception', ]]\
        .groupby('posteam' if for_offense else 'defteam')

    totals: Final = plays_by_team.agg(
        games=('week', 'nunique'),
        drives=('unique_drive', 'nunique'),
        series=('unique_series', 'nunique'),
        plays=('week', 'count'),
        pass_touchdowns=('pass_touchdown', 'sum'),
        rush_touchdowns=('rush_touchdown', 'sum'),
        field_goals=('field_goal_result', lambda series: (series == 'made').sum()),
        two_point_conversions=('two_point_conv_result', lambda series: (series == 'success').sum()),
        extra_points=('extra_point_result', lambda series: (series == 'good').sum()),
        yards_gained=('yards_gained', 'sum'),
        epa=('epa', 'sum'),
        fumbles=('fumble_lost', 'sum'),
        interceptions=('interception', 'sum'), )
    totals.index.name = 'team'

    home_wins: Final = final_plays.groupby('home_team').home_wins.sum()
    away_wins: Final = final_plays.groupby('away_team').away_wins.sum()

    totals['wins'] = home_wins.add(away_wins, fill_value=0)

    return totals


def get_unadjusted_scores(raw_scores: pd.DataFrame, offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    unadjusted_scores: Final = pd.DataFrame(
        index=raw_scores.index,
        data={
            'offense_game': raw_scores.offense / offensive_totals.games,
            'offense_drive': raw_scores.offense / offensive_totals.drives,
            'offense_series': raw_scores.offense / offensive_totals.series,
            'offense_play': raw_scores.offense / offensive_totals.plays,

            'defense_game': raw_scores.defense / defensive_totals.games,
            'defense_drive': raw_scores.defense / defensive_totals.drives,
            'defense_series': raw_scores.defense / defensive_totals.series,
            'defense_play': raw_scores.defense / defensive_totals.plays, })

    return unadjusted_scores


def add_differential_columns_to(scores: pd.DataFrame) -> None:
    scores['differential_game'] = scores.offense_game + scores.defense_game
    scores['differential_drive'] = scores.offense_drive + scores.defense_drive
    scores['differential_play'] = scores.offense_play + scores.defense_play


def get_points_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': offensive_totals.apply(
                lambda row: row.pass_touchdowns * 6 + row.rush_touchdowns * 6 + row.field_goals * 3 + row.two_point_conversions * 2 + row.extra_points,
                axis=1),
            'defense': -defensive_totals.apply(
                lambda row: row.pass_touchdowns * 6 + row.rush_touchdowns * 6 + row.field_goals * 3 + row.two_point_conversions * 2 + row.extra_points,
                axis=1), })


def get_yards_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': offensive_totals.yards_gained,
            'defense': -defensive_totals.yards_gained, })


def get_wins_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': offensive_totals.wins,
            'defense': defensive_totals.wins, })


def get_epa_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': offensive_totals.epa,
            'defense': -defensive_totals.epa, })


def get_turnovers_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': -offensive_totals.apply(lambda row: row.fumbles + row.interceptions, axis=1),
            'defense': defensive_totals.apply(lambda row: row.fumbles + row.interceptions, axis=1), })


def get_success_from(offensive_totals: pd.DataFrame, defensive_totals: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        index=offensive_totals.index,
        data={
            'offense': offensive_totals.epa > 0,
            'defense': defensive_totals.epa < 0, })


def yield_progress_then_single_scenario_correlations(context: Context, useful_plays: pd.DataFrame) -> Iterator[float | pd.DataFrame]:
    accumulated_unadjusted_correlations: dict[BaseMetric, list[pd.Series]] = {bm: [] for bm in BaseMetric}
    accumulated_adjusted_correlations: dict[BaseMetric, list[pd.Series]] = {bm: [] for bm in BaseMetric}

    for i, week_split in enumerate(context.week_splits):
        yield i / len(context.week_splits)

        example_final_plays: pd.DataFrame = context.final_plays.loc[context.final_plays.week.isin(week_split.example_weeks)]
        example_useful_plays: pd.DataFrame = useful_plays.loc[useful_plays.week.isin(week_split.example_weeks)]

        prediction_all_game_results: pd.DataFrame = context.all_game_results.loc[
            context.all_game_results.week.isin(week_split.prediction_weeks)]
        prediction_game_results: pd.DataFrame = prediction_all_game_results.drop(columns='week').groupby('team').sum()

        example_offensive_totals = get_totals_from(example_final_plays, example_useful_plays, for_offense=True)
        example_defensive_totals = get_totals_from(example_final_plays, example_useful_plays, for_offense=False)
        example_schedule: pd.DataFrame = context.full_schedule.loc[
            context.full_schedule.week.isin(week_split.example_weeks)]

        for metric_generator in context.all_metric_generators:
            example_raw_scores: pd.DataFrame = metric_generator.get_scores(example_offensive_totals, example_defensive_totals)
            example_unadjusted_scores = get_unadjusted_scores(example_raw_scores, example_offensive_totals, example_defensive_totals)
            example_all_opponent_scores = example_schedule.merge(example_unadjusted_scores, left_on='opponent', right_index=True, how='left')
            example_avg_opponent_scores = example_all_opponent_scores.groupby('team').agg('mean', numeric_only=True)

            example_adjusted_scores = pd.DataFrame(
                index=example_unadjusted_scores.index,
                data={
                    'offense_game': example_unadjusted_scores.offense_game + example_avg_opponent_scores.defense_game,
                    'offense_drive': example_unadjusted_scores.offense_drive + example_avg_opponent_scores.defense_drive,
                    'offense_series': example_unadjusted_scores.offense_series + example_avg_opponent_scores.defense_series,
                    'offense_play': example_unadjusted_scores.offense_play + example_avg_opponent_scores.defense_play,

                    'defense_game': example_unadjusted_scores.defense_game + example_avg_opponent_scores.offense_game,
                    'defense_drive': example_unadjusted_scores.defense_drive + example_avg_opponent_scores.offense_drive,
                    'defense_series': example_unadjusted_scores.defense_series + example_avg_opponent_scores.offense_series,
                    'defense_play': example_unadjusted_scores.defense_play + example_avg_opponent_scores.offense_play, })

            add_differential_columns_to(example_unadjusted_scores)
            add_differential_columns_to(example_adjusted_scores)

            unadjusted_correlations: pd.Series = example_unadjusted_scores.corrwith(prediction_game_results['wins'])
            adjusted_correlations: pd.Series = example_adjusted_scores.corrwith(prediction_game_results['wins'])

            accumulated_unadjusted_correlations[metric_generator.base].append(unadjusted_correlations)
            accumulated_adjusted_correlations[metric_generator.base].append(adjusted_correlations)

    base_metric_correlations: list[pd.DataFrame] = []

    for base_metric, accumulated_unadjusted_correlation in accumulated_unadjusted_correlations.items():
        unadjusted_correlations_frame: pd.DataFrame = pd.concat(accumulated_unadjusted_correlation, axis=1)
        avg_unadjusted_correlations: pd.Series = unadjusted_correlations_frame.mean(axis=1)
        base_metric_correlations.append(
            pd.DataFrame({
                'base': base_metric.name,
                'adjusted_by': 'none',
                'unit': [i.split('_')[0] for i in avg_unadjusted_correlations.index],
                'per': [i.split('_')[1] for i in avg_unadjusted_correlations.index],
                'correlation': avg_unadjusted_correlations, }))

    for base_metric, accumulated_adjusted_correlation in accumulated_adjusted_correlations.items():
        adjusted_correlations_frame: pd.DataFrame = pd.concat(accumulated_adjusted_correlation, axis=1)
        avg_adjusted_correlations: pd.Series = adjusted_correlations_frame.mean(axis=1)
        base_metric_correlations.append(
            pd.DataFrame({
                'base': base_metric.name,
                'adjusted_by': 'opponent',
                'unit': [i.split('_')[0] for i in avg_adjusted_correlations.index],
                'per': [i.split('_')[1] for i in avg_adjusted_correlations.index],
                'correlation': avg_adjusted_correlations, }))

    yield pd.concat(base_metric_correlations)


def update_progress_then_return_frame(iterator: Iterator[float | pd.DataFrame], text_format: str) -> pd.DataFrame:
    progress_bar: Final = st.progress(0, text='')

    for progress_or_results in iterator:
        if isinstance(progress_or_results, pd.DataFrame):
            progress_bar.empty()
            return progress_or_results
        else:
            assert isinstance(progress_or_results, float)
            progress_bar.progress(progress_or_results, text=text_format.format(progress_or_results * 100))


def yield_progress_then_all_correlations(context: Context) -> Iterator[float | pd.DataFrame]:
    scenarios: Final = [
        Scenario('any', False, lambda: context.all_plays.loc[
            context.all_plays.play_type.isin(['run', 'pass', 'field_goal', 'extra_point'])]),
        Scenario('core', False, lambda: context.all_plays.loc[
            context.all_plays.play_type.isin(['run', 'pass'])]),
        Scenario('rushing', False, lambda: context.all_plays.loc[
            context.all_plays.play_type == 'run']),
        Scenario('passing', False, lambda: context.all_plays.loc[
            context.all_plays.play_type == 'pass']),

        Scenario('any', True, lambda: context.all_plays.loc[
            context.all_plays.play_type.isin(['run', 'pass', 'field_goal', 'extra_point']) & context.all_plays.neutral_scenario]),
        Scenario('core', True, lambda: context.all_plays.loc[
            context.all_plays.play_type.isin(['run', 'pass']) & context.all_plays.neutral_scenario]),
        Scenario('rushing', True, lambda: context.all_plays.loc[
            (context.all_plays.play_type == 'run') & context.all_plays.neutral_scenario]),
        Scenario('passing', True, lambda: context.all_plays.loc[
            (context.all_plays.play_type == 'pass') & context.all_plays.neutral_scenario]), ]

    scenarios_correlations: list[pd.DataFrame] = []
    for i, scenario in enumerate(scenarios):
        yield i / len(scenarios)

        is_neutral_str = 'neutral' if scenario.is_neutral else 'any'
        scenario_correlations: pd.DataFrame = update_progress_then_return_frame(
            yield_progress_then_single_scenario_correlations(context, scenario.get_plays()),
            f'Running all metrics in scenario {is_neutral_str}, subject {scenario.name}...' + ' {:.0f}%')
        scenario_correlations.insert(0, 'scenario', is_neutral_str)
        scenario_correlations.insert(0, 'subject', scenario.name)
        scenarios_correlations.append(scenario_correlations)

    yield pd.concat(scenarios_correlations, ignore_index=True)


def get_all_correlations(context: Context) -> pd.DataFrame:
    correlations: pd.DataFrame = update_progress_then_return_frame(
        yield_progress_then_all_correlations(context),
        'Running all subject/scenario pairs... {:.0f}%')

    correlations.sort_values(by='correlation', inplace=True, ascending=False)
    return correlations


def filter_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    lookup = None
    for ft, fi in all_filters.items():
        if lookup is None:
            lookup = correlations[ft.name].isin(st.session_state[ft.name])
        else:
            lookup = lookup & correlations[ft.name].isin(st.session_state[ft.name])
    return correlations.loc[lookup]


def get_final_frame() -> pd.DataFrame:
    final_frame_stem: Final = 'final_frame.csv'
    if Path(final_frame_stem).exists():
        return filter_correlations(pd.read_csv(final_frame_stem))

    all_plays: Final = load_plays_from_file_for_year(year_to_process, win_prob=neutral_win_probability)
    all_final_plays: Final = get_final_plays(all_plays)
    context: Final = Context(
        full_schedule=get_schedule_from(all_final_plays),
        all_game_results=get_game_results_from(all_final_plays),
        all_metric_generators=[
            MetricGenerator(BaseMetric.points, get_points_from),
            MetricGenerator(BaseMetric.yards, get_yards_from),
            MetricGenerator(BaseMetric.wins, get_wins_from),
            MetricGenerator(BaseMetric.epa, get_epa_from),
            MetricGenerator(BaseMetric.turnovers, get_turnovers_from),
            MetricGenerator(BaseMetric.success, get_success_from), ],
        all_plays=all_plays,
        final_plays=all_final_plays,
        week_splits=get_week_splits(max_week_split_count, randomize=randomize_weekly_splits), )

    all_correlations: Final = get_all_correlations(context)
    all_correlations.to_csv(final_frame_stem, index=False)
    return filter_correlations(all_correlations)


year_to_process: Final = 2023
neutral_win_probability: Final = 0.05
max_week_split_count: Final = 10000
prediction_week_count: Final = 9
randomize_weekly_splits: Final = True
maximum_week: Final = 18

all_filters: Final = {
    FilterType.subject: FilterInfo('Subjects:', ['any', 'core', 'rushing', 'passing']),
    FilterType.scenario: FilterInfo('Scenarios:', ['any', 'neutral']),
    FilterType.base: FilterInfo('Bases:', ['points', 'yards', 'wins', 'epa', 'turnovers', 'success']),
    FilterType.adjusted_by: FilterInfo('Adjustments:', ['none', 'opponent']),
    FilterType.unit: FilterInfo('Units:', ['offense', 'defense', 'differential']),
    FilterType.per: FilterInfo('Frequency:', ['game', 'drive', 'series', 'play']), }

for filter_type, filter_info in all_filters.items():
    if filter_type.name not in st.session_state:
        st.session_state[filter_type.name] = filter_info.all_options

st.title(f'Predictive accuracy of NFL metrics using {year_to_process} play-by-play data')
final_frame: Final = get_final_frame()
st.dataframe(final_frame, use_container_width=True)
final_correlations: Final = final_frame['correlation']
st.write(f'Correlation summary: min = {final_correlations.min():.4f}, max = {final_correlations.max():.4f}, mean = {final_correlations.mean():.4f}, median = {final_correlations.median():.4f}, rows = {len(final_correlations)}')
st.subheader('Notes')
st.write('Offensive and defensive performances are modified so that greater values are always better. For instance, offensive yards are positive (the more the better) but defensive yards are negative (the fewer the better). Turnovers are the odd metric out because defensive turnovers are positive and offensive turnovers are negative.')
st.write('The "core" subject only includes rushes and passes while the "any" subject includes rushes, passes, field goal attempts, and extra point attempts.')
st.write(f'The "neutral" scenario filters out plays with a win probability greater than {100 - neutral_win_probability * 100:.0f}% or less than {neutral_win_probability * 100:.0f}%.')
st.write('The "success" base is 1 if the play\'s EPA was greater than zero and 0 otherwise.')
st.write('The "differential" unit is the offense plus the defense (remember that defensive values are the opposite of offensive values so this will work).')
if randomize_weekly_splits:
    st.write(f'Raw play-by-play data was processed {max_week_split_count:,} times per subject/scenario pair, each time split into a random {maximum_week - prediction_week_count} week training data set and a leftover {prediction_week_count} week prediction data set.')
else:
    st.write(f'Raw play-by-play data was processed no more than {max_week_split_count:,} times per subject/scenario pair, once for each combination of {prediction_week_count} week prediction data set and a leftover {maximum_week - prediction_week_count} week prediction data set.')
st.write('Raw play-by-play data from [nflfastR](https://github.com/nflverse/nflverse-data/).')

with st.sidebar:
    st.subheader('Filters')
    st.button('Reset all', use_container_width=True, on_click=lambda: st.session_state.clear())
    st.write('If no metrics show in the table, try resetting all filters.')

    for filter_type, filter_info in all_filters.items():
        st.multiselect(
            filter_info.prompt,
            filter_info.all_options,
            key=filter_type.name, )

# streamlit run .\main.py
