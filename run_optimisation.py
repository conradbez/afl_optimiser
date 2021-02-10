# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
# !pip install pyomo
# !pip install --upgrade --user ortools
# !apt-get install -y glpk-utils
# !sudo apt-get update -y
# !sudo apt-get install -y coinor-cbc


# %%
# get_ipython().system('pip install -U git+https://github.com/coin-or/pulp')
# get_ipython().system('sudo pulptest')


# %%
import pandas as pd
from pulp import *


# %%
# full_player_scores[full_player_scores['Score']<1 and ]


# %%
# full_player_scores[(full_player_scores['Name']=='Zac Bailey')&(full_player_scores['Year']==2020)].sort_values('Round')


# %%
full_player_scores = pd.read_csv('full_players.csv', index_col=0)
full_player_scores = full_player_scores.reset_index(drop=True)
full_player_scores['Score'] = pd.to_numeric(full_player_scores['Score'], errors='coerce')
# full_player_scores['Value'] = full_player_scores['Score']/full_player_scores['Price']
full_player_scores = full_player_scores[full_player_scores['Year']==2020]

player_num_games_played = full_player_scores[full_player_scores['Score']>0].groupby('Name')['Team 2'].transform('count')

opponent_average_score = full_player_scores.groupby('Team 2')['Score'].transform('mean')
player_average_score = full_player_scores.groupby('Name')['Score'].transform('mean')
player_average_value = player_average_score/full_player_scores['Price']

na_locs = full_player_scores['Score'].isna()

full_player_scores.loc[na_locs,'Score'] = ((opponent_average_score+player_average_score)/2).loc[na_locs]

# RU_Players = full_player_scores[(full_player_scores['Round']<=2)&(full_player_scores['Year']==2020)] # Only two rounds
# RU_Players = full_player_scores[(full_player_scores['Round']<=5)&(full_player_scores['Position']=='RU')&(full_player_scores['Year']==2020)] # Only rucks
# RU_Players = full_player_scores[(full_player_scores['Round']<=3)&(full_player_scores['Year']==2020)]
RU_Players = full_player_scores[(full_player_scores['Year']==2020)]
# RU_Players = full_player_scores[(full_player_scores['Position']=='RU')&(full_player_scores['Year']==2020)] # Only rucks

RU_Players = RU_Players.sort_values('Score',ascending=False)
RU_Players['id'] = "Round:"+RU_Players['Round'].astype(str)+"_Player:"+RU_Players['Name']
# RU_Players = RU_Players[RU_Players['Round']<3]

# Only take players in top quater of average values (score/price)
# player_best_values = player_average_value.sort_values(ascending=False)[:int(len(player_average_value)/4)]
# RU_Players = RU_Players.reindex(index=player_best_values.index.values).dropna(how='all') 

# Only take players who played more than 5 games
# players_to_keep = player_num_games_played[player_num_games_played>5].index.values
# RU_Players = RU_Players.reindex(index=players_to_keep).dropna(how='all') 


# %%
trades_allowed = None
player_contraints = {}
prob = LpProblem("aflProblem", LpMaximize)
overall_score = LpVariable('OverallScore',0)
for p_id,score in zip(RU_Players['id'].iteritems(),RU_Players['Score'].iteritems()):
  p_id = p_id[1]
  player_contraints[p_id] = LpVariable(p_id, 0, 1, cat='Binary')

player_contraints = LpVariable.dicts("player_contraints", player_contraints, 0, 1, cat='Binary')
prob += lpSum([player_contraints[p_id[1]]*score[1] for p_id,score in zip(RU_Players['id'].iteritems(),RU_Players['Score'].iteritems())]), "Total score is maximized"

# START TRANFERS
RU_Players_prev_round = RU_Players
RU_Players_prev_round['Round_prev'] = RU_Players_prev_round['Round'] - 1

RU_transfers = RU_Players[['id','Position','Round']].merge(RU_Players_prev_round[['id','Position','Round_prev']],left_on=['Position','Round'],right_on=['Position','Round_prev'], suffixes = ('_prev','_next'))
RU_transfers['Transfer'] = RU_transfers['id_prev']+'->'+RU_transfers['id_next']

transfer_contraints = {}

# Define transfer itermediaries
for i,t_id in RU_transfers['Transfer'].iteritems():
  transfer_contraints[t_id] = LpVariable(t_id, 0, 1, cat='Binary')

transfer_contraints = LpVariable.dicts("player_contraints", transfer_contraints, 0, 1, cat='Binary')

# map rounds end to itermediary
for prev_player,trans in RU_transfers.groupby(['id_prev'])['Transfer'].apply(list).iteritems():
  prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[prev_player], "Previous player equals transfer intermediatary for "+prev_player

# map intermediary to next round
for next_player,trans in RU_transfers.groupby(['id_next'])['Transfer'].apply(list).iteritems():
  prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[next_player], "Next player equals transfer intermediatary for "+next_player
# END TRANFERS

# START max players from each position
allowed_holds_per_position = {'DE': 8, "MI" : 10, 'RU' : 3, 'FO':9}

for (position,round), player in RU_Players[['id','Position','Round']].drop_duplicates().groupby(['Position', 'Round'])['id'].apply(list).iteritems():
  prob += lpSum([player_contraints[p_id] for p_id in player]) <= allowed_holds_per_position[position], f"Position: {position}, has less than {allowed_holds_per_position[position]} in round {round}"
# END max players from each position

# START money contraint
money = 13 * 10**6
for round,player in RU_Players[['id','Price','Round']].drop_duplicates().groupby(['Round'])['id'].apply(list).iteritems():
  prob += lpSum([player_contraints[p_id]*RU_Players[RU_Players['id']==p_id]['Price'].values[0] for p_id in player]) <= money, f"Round: {round}, has less than ${money}"
# END money contraint

print('solving')


# %%

# solver = getSolver('COIN_CMD',  msg=True,)
solver = getSolver('COIN_CMD', msg=True, cuts=True, maxSeconds=700)

# prob.solve(pulp.PULP_CBC_CMD(msg=True, maxSeconds=10))
# prob.solve(PULP_CBC_CMD(gapRel = 0.05))

# solver = getSolver('GLPK_CMD')
prob.solve(solver)

# pulp.COIN(maxSeconds=your_time_limit))
# prob.solve(solver)
# list_solvers(onlyAvailable=True)



# %%
for v in prob.variables():
    if v.varValue != 0:
        print(v.name)
        print(v.value())

#   if v.varValue != 0 and '10_' in v.name and not '11_' in v.name and not '9_' in v.name:


# %%



