# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# !jupyter nbconvert --to script AFL.ipynb


# %%
# !pip install pyomo
# !pip install --upgrade --user ortools
# !apt-get install -y glpk-utils
# !sudo apt-get update -y
# !sudo apt-get install -y coinor-cbc


# %%
# !pip install -U git+https://github.com/coin-or/pulp
# !sudo pulptest
# !pip install pandas


# %%
import numpy as np
import pandas as pd
from pulp import *


# %%
def test1(test_df):
    assert len(test_df[test_df['Name'].isna()])==0
    for r in test_df['Round'].unique():
        names = test_df[test_df['Round']==r]['Name'].unique()
        try:    
            assert len(test_df[~(test_df['Name'].isin(names))])==0
        except AssertionError:
            # print(r)
            # print(test_df[~(test_df['Name'].isin(names))])
            assert len(test_df[~(test_df['Name'].isin(names))])==0


# %%

def getPlayers(full_player_scores = pd.read_csv('full_players.csv', index_col=0)):
    full_player_scores.dropna(subset=['Name'])
    full_player_scores = full_player_scores.reset_index(drop=True)
    full_player_scores['Score'] = pd.to_numeric(full_player_scores['Score'], errors='coerce')

    # testfilter
    # full_player_scores = full_player_scores[full_player_scores['Round']<4]
    # players_to_keep = ["Adam Tomlinson", "Max Gawn"]
    # players_to_keep = full_player_scores.sort_values('Score',ascending=False)['Name'].unique()[:100]
    # full_player_scores = full_player_scores[full_player_scores['Name'].isin(players_to_keep)]


    full_player_scores_2021 = full_player_scores[full_player_scores['Year']==2021]
    full_player_scores = full_player_scores[full_player_scores['Year']==2020]

    # Only keep players who are playing this season
    full_player_scores = full_player_scores[full_player_scores['Name'].isin(full_player_scores_2021['Name'].unique())].dropna(how='all').dropna(how='all')

    df = full_player_scores.copy()

    df1 = df[['Name','Team','Position','Price', 'Team 1','Year']].drop_duplicates()
    df2 = df[['Round']].drop_duplicates()
    df1['key'] = 0
    df2['key'] = 0
    df = df1.merge(df2, how='outer', on=['key'])
    df = df.sort_values('Round')
    df = df.drop('key', axis=1)
    full_player_scores = full_player_scores.merge(df, on=['Name', 'Team', 'Position','Round','Price', 'Team 1','Year'],how='right').reset_index(drop=True)
    try:
        test1(full_player_scores)
    except AssertionError:
        print('WARNING: failed a test')


    player_num_games_played = full_player_scores[full_player_scores['Score']>0].groupby('Name')['Team 2'].transform('count')

    opponent_average_score = full_player_scores.groupby('Team 2')['Score'].transform('mean')
    player_average_score = full_player_scores.groupby('Name')['Score'].transform('mean')
    player_average_value = player_average_score/full_player_scores['Price']

    na_locs = full_player_scores['Score'].isna()

    full_player_scores.loc[na_locs,'Score'] = ((opponent_average_score+player_average_score)/2).loc[na_locs]

    na_locs = full_player_scores['Score'].isna()

    full_player_scores.loc[na_locs,'Score'] = (player_average_score).loc[na_locs]

    RU_Players = full_player_scores.copy()

    assert list(full_player_scores['Year'].unique())==[2020]

    RU_Players = RU_Players.sort_values('Score',ascending=False)

    RU_Players = full_player_scores.copy()

    assert list(full_player_scores['Year'].unique())==[2020]

    RU_Players = RU_Players.sort_values('Score',ascending=False)
    RU_Players['id'] = "R:"+RU_Players['Round'].astype(str)+"_P:"+RU_Players['Name']

    # Only take players who played more than 5 games
    # players_to_keep = player_num_games_played[player_num_games_played>5].index.values
    # RU_Players = RU_Players.reindex(index=players_to_keep).dropna(how='all')

    # Only take players in top quater of average values (score/price)
    # player_best_values = player_average_value.sort_values(ascending=False)[:int(len(player_average_value)/2)]
    # RU_Players = RU_Players.reindex(index=player_best_values.index.values).dropna(how='all') 

    try:
        test1(RU_Players)
    except AssertionError:
        print('WARNING: failed a test')

    RU_Players.groupby(['Round','Position']).count()
    return RU_Players
RU_Players = getPlayers()


# %%
def getProblem(RU_Players):
  player_contraints = {}
  prob = LpProblem("aflProblem", LpMaximize)
  overall_score = LpVariable('OverallScore',0)
  for p_id,score in zip(RU_Players['id'].iteritems(),RU_Players['Score'].iteritems()):
    p_id = p_id[1]
    player_contraints[p_id] = LpVariable(p_id, 0, 1, cat='Binary')

  player_contraints = LpVariable.dicts("PC", player_contraints, 0, 1, cat='Binary')
  prob += lpSum([player_contraints[p_id[1]]*score[1] for p_id,score in zip(RU_Players['id'].iteritems(),RU_Players['Score'].iteritems())]), "Total score is maximized"

  # START TRANFERS
  RU_Players_prev_round = RU_Players
  RU_Players_prev_round['Round_prev'] = RU_Players_prev_round['Round'] - 1

  RU_transfers = RU_Players[['id','Position','Round','Name']].merge(RU_Players_prev_round[['id','Position','Round_prev','Name']],left_on=['Position','Round'],right_on=['Position','Round_prev'], suffixes = ('_prev','_next'))

  RU_transfers['Transfer'] = RU_transfers['id_prev']+'-'+RU_transfers['id_next']
  RU_transfers=RU_transfers[['id_prev',	'Round',	'Name_prev',	'id_next',	'Round_prev',	'Name_next',	'Transfer']].drop_duplicates()

  transfer_contraints = {}

  # Define transfer itermediaries
  for i,t_id in RU_transfers['Transfer'].iteritems():
    transfer_contraints[t_id] = LpVariable(t_id, 0, 1, cat='Binary')

  transfer_contraints = LpVariable.dicts("TC", transfer_contraints, 0, 1, cat='Binary')

  # map rounds end to itermediary
  for prev_player,trans in RU_transfers.groupby(['id_prev'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[prev_player], f"Previous player equals transfer intermediatary for {prev_player} equals {trans}"

  # map intermediary to next round
  for next_player,trans in RU_transfers.groupby(['id_next'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[next_player], f"Next player equals transfer intermediatary for {next_player} equals {trans}"

  for round, transfers in RU_transfers[RU_transfers['Name_prev'] != RU_transfers['Name_next']][['Round','Transfer']].drop_duplicates().groupby(['Round'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in transfers]) <= 4, f"Round: {round}, has less than or equal to 4 transfers"
    # print(f'{[transfer_contraints[t_id] for t_id in transfers]}) <= 4')
  # END TRANFERS

  # START money contraint
  money = 13 * 10**6
  for round,player in RU_Players[['id','Price','Round']].drop_duplicates().groupby(['Round'])['id'].apply(list).iteritems():
    prob += lpSum([player_contraints[p_id]*RU_Players[RU_Players['id']==p_id]['Price'].values[0] for p_id in player]) <= money, f"Round: {round}, has less than ${money}"
  # END money contraint

  #START Contrain each player to a single position per round
  RU_Players_Position_Casting = RU_Players.copy()
  RU_Players_Position_Casting['pl_pos_id'] = (RU_Players['id']+'_'+RU_Players['Position'])

  Players_Position_Casting_Series = RU_Players_Position_Casting[['id','pl_pos_id','Round']].drop_duplicates().groupby(['id','Round'])['pl_pos_id'].apply(list).iteritems()

  Players_Position_Casting = {}
  for i,r in RU_Players_Position_Casting[['pl_pos_id']].drop_duplicates().iterrows():
    Players_Position_Casting[r['pl_pos_id']] = LpVariable(r['pl_pos_id'], 0, 1, cat='Binary')

  Players_Position_Casting = LpVariable.dicts("Players_Position_Casting", Players_Position_Casting, 0, 1, cat='Binary')

  for (p_id,round), pl_pos_ids in Players_Position_Casting_Series:
      player_possible_positions = [Players_Position_Casting[pl_pos_id] for pl_pos_id  in pl_pos_ids] 
      prob += lpSum(player_possible_positions) == player_contraints[p_id], f"Must have only have equal to {p_id} (0 or 1) of {player_possible_positions} positions in round {round}"
  #END Contrain each player to a single position per round

  # START max players from each position
  allowed_holds_per_position = {'DE': 8, "MI" : 10, 'RU' : 3, 'FO':9}
  # allowed_holds_per_position = {'DE': 1, "MI" : 1, 'RU' : 1, 'FO':1}

  print('START max players from each position')
  for (position,round), pl_pos_ids in RU_Players_Position_Casting[['pl_pos_id','Position','Round']].drop_duplicates().groupby(['Position', 'Round'])['pl_pos_id'].apply(list).iteritems():
    prob += lpSum([Players_Position_Casting[pl_pos_id] for pl_pos_id in pl_pos_ids]) == allowed_holds_per_position[position], f"Position: {position}, has less than {allowed_holds_per_position[position]} in round {round}"
    # print('\n')
  # END max players from each position

  print('done with pre-work')
  return prob, transfer_contraints, Players_Position_Casting, RU_Players_Position_Casting
prob, transfer_contraints, Players_Position_Casting, RU_Players_Position_Casting = getProblem(RU_Players)


# %%
try:
    solver = getSolver('COIN_CMD', timeLimit=2000, msg=True,gapRel = 0.1)
    prob.solve(solver)
except:
    print('USING INBUILT SOLVER!!  \n'*100)
    solver = getSolver('PULP_CBC_CMD',gapRel = 0.1)
    prob.solve(solver)


# %%
prob.status


# %%
for i,v in transfer_contraints.items():
    if v.value() != 0:
        # print(v.value())
        # print(i)
        last_name = i[i.find(' ')+1: i.find('-')]
        last_letters = i[-len(last_name):]
        # print(last_name)
        # print(last_letters)
        if not last_name == last_letters:
            print(i)


# %%
results = []
for name, player_position in Players_Position_Casting.items():
    if player_position.value() != 0:
        results.append([name,1])
Players_selected = pd.DataFrame(results, columns=['pl_pos_id', 'is_selected'])
RU_Players_Position_Casting = RU_Players_Position_Casting.merge(Players_selected, on = ['pl_pos_id'])
RU_Players_Position_Casting.to_csv('solution.csv')
print(RU_Players_Position_Casting.groupby(['Round','Position']).sum()['is_selected'])


# %%
RU_Players_Position_Casting.sort_values(['Round'])


# %%
for round in RU_Players_Position_Casting['Round'].unique():
    assert RU_Players_Position_Casting[RU_Players_Position_Casting['Round']==round]['is_selected'].sum()


# %%


