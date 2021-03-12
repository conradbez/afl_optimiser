# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from pulp import *


# %%
full_players = pd.read_csv('full_players.csv', index_col=0).reset_index(drop=True)
# full_players = full_players[::50]

full_players['Score'] = pd.to_numeric(full_players['Score'].replace('-',0))
schedule = pd.read_csv('2021_schedule', index_col=0) 

players_2021 = full_players[full_players['Year']==2021][['Name','Price','Team 1', 'Position']]
player_prev = full_players[full_players['Name'].isin(players_2021['Name'].unique())]
player_prev = player_prev.groupby(['Name','Team 2']).mean()['Score']

players_schedule_2021 = players_2021.merge(schedule, on=['Team 1'])
players_schedule_score_2021 = players_schedule_2021.merge(player_prev, on = ['Name','Team 2'])

# add bye's
all_players_list = players_schedule_score_2021[['Name','Price','Position', 'Team 1']].drop_duplicates()
all_players_list['key'] = 1
all_rounds_list =  players_schedule_score_2021[['Round']].drop_duplicates()
all_rounds_list['key'] = 1
all_players_list = all_players_list.merge(all_rounds_list, on=['key']).drop('key',1)
current_players_list = players_schedule_score_2021.copy()
missing_players = all_players_list.merge(current_players_list, how='outer',indicator=True)
missing_players = missing_players[missing_players['_merge'] == 'left_only'].drop('_merge',1)
missing_players['Score'] = 0
missing_players['Team 2'] = '_BYE'

players_schedule_score_2021 = players_schedule_score_2021.append(missing_players).reset_index(drop=True)
# end add bye's

P = players_schedule_score_2021
# P = P[(P['Round']<3)&(P['Round']<14)]
# P = P[(P['Round']>10)&(P['Round']<14)]
# players_to_be_kept = P.copy()
# players_to_be_kept["Value"] = players_to_be_kept['Score']/players_to_be_kept['Price']
# players_to_be_kept = players_to_be_kept.groupby('Name').mean()[['Score','Value']]

# players_to_keep = []
# # num_players_to_keep = int(len(players_to_be_kept)*0.2)
# num_players_to_keep = int(len(players_to_be_kept)*0.5)
# players_to_keep += list(players_to_be_kept.sort_values('Score', ascending = False).index[:num_players_to_keep])
# players_to_keep += list(players_to_be_kept.sort_values('Value', ascending = False).index[:num_players_to_keep])

# P = P[P['Name'].isin(players_to_keep)]


# %%
P['id'] = "R:"+P['Round'].astype(str)+"_P:"+P['Name']

player_contraints = {}
prob = LpProblem("aflProblem", LpMaximize)

overall_score = LpVariable('OverallScore',0)
player_contraints = LpVariable.dicts("Player Contraints", P['id'].unique(), 0, 1, cat='Binary')
prob += lpSum([player_contraints[p_id[1]]*score[1] for p_id,score in zip(P['id'].iteritems(),P['Score'].iteritems())]), "Total score is maximized"


# %%
P_prev_round = P.copy()
P_prev_round['Round_prev'] = P_prev_round['Round'] - 1

P_transfers = P[['id','Position','Round','Name']].merge(P_prev_round[['id','Position','Round_prev','Name']],left_on=['Position','Round'],right_on=['Position','Round_prev'], suffixes = ('_prev','_next'))

P_transfers['Transfer'] = P_transfers['id_prev']+'-'+P_transfers['id_next']
P_transfers=P_transfers[['id_prev',	'Round',	'Name_prev',	'id_next',	'Round_prev',	'Name_next',	'Transfer']].drop_duplicates()

transfer_contraints = LpVariable.dicts("Transfer contraints", P_transfers['Transfer'], 0, 1, cat='Binary')

# map rounds end to itermediary
for prev_player,trans in P_transfers.groupby(['id_prev'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[prev_player], f"Previous player equals transfer intermediatary for {prev_player} equals {trans}"

# map intermediary to next round
for next_player,trans in P_transfers.groupby(['id_next'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in trans]) == player_contraints[next_player], f"Next player equals transfer intermediatary for {next_player} equals {trans}"

for round, transfers in P_transfers[P_transfers['Name_prev'] != P_transfers['Name_next']][['Round','Transfer']].drop_duplicates().groupby(['Round'])['Transfer'].apply(list).iteritems():
    prob += lpSum([transfer_contraints[t_id] for t_id in transfers]) <= 2, f"Round: {round}, has less than or equal to 4 transfers"

# # END TRANFERS


# %%
# START money contraint
money = 13 * 10**6
for round,player in P[['id','Price','Round']].drop_duplicates().groupby(['Round'])['id'].apply(list).iteritems():
    prob += lpSum([player_contraints[p_id]*P[P['id']==p_id]['Price'].values[0] for p_id in player]) <= money, f"Round: {round}, has less than ${money}"
# # END money contraint


# %%
#START Contrain each player to a single position per round
P_Position_Casting = P.copy()
P_Position_Casting['pl_pos_id'] = (P['id']+'_'+P['Position'])
P_Position_Casting = P_Position_Casting[['id','pl_pos_id','Round','Position']].drop_duplicates()

Players_Position_Casting = LpVariable.dicts("Players_Position_Casting", P_Position_Casting['pl_pos_id'], 0, 1, cat='Binary')

Players_Position_Casting_Series = P_Position_Casting[['id','pl_pos_id','Round']].drop_duplicates().groupby(['id','Round'])['pl_pos_id'].apply(list).iteritems()

for (p_id,round), pl_pos_ids in Players_Position_Casting_Series:
    player_possible_positions = [Players_Position_Casting[pl_pos_id] for pl_pos_id  in pl_pos_ids] 
    prob += lpSum(player_possible_positions) <= player_contraints[p_id], f"Must have only have equal to {p_id} (0 or 1) of {player_possible_positions} positions in round {round}"
  #END Contrain each player to a single position per round


# %%
# # START max players from each position
allowed_holds_per_position = {'DE': 8, "MI" : 10, 'RU' : 3, 'FO':9}
# allowed_holds_per_position = {'DE': 1, "MI" : 0, 'RU' : 0, 'FO':0}
for (position,round), pl_pos_ids in P_Position_Casting[['pl_pos_id','Position','Round']].drop_duplicates().groupby(['Position', 'Round'])['pl_pos_id'].apply(list).iteritems():
    prob += lpSum([Players_Position_Casting[pl_pos_id] for pl_pos_id in pl_pos_ids]) == allowed_holds_per_position[position], f"Position: {position}, has less than {allowed_holds_per_position[position]} in round {round}"
# END max players from each position


# %%
# solver = getSolver('COIN_CMD', timeLimit=1000, msg=True,gapRel = 0.2)
# prob.solve(solver)

solver = getSolver('COIN_CMD', msg=True)
prob.solve(solver)

# solver = getSolver('PULP_CBC_CMD', maxSeconds=1000, msg=True, fracGap = 0.9)
# prob.solve(solver)


# %%
prob.status


# %%
results = []
for name, player_position in Players_Position_Casting.items():
    if player_position.value() != 0:
        results.append([name,1])
Players_selected = pd.DataFrame(results, columns=['pl_pos_id', 'is_selected'])
P_Position_Casting = P_Position_Casting.merge(Players_selected, on = ['pl_pos_id'])
P_Position_Casting.to_csv('solution.csv')
P_Position_Casting
# print(P_Position_Casting.groupby(['Round','Position']).sum()['is_selected'])


# %%



