import pandas as pd

df_jug = pd.read_csv('0825_jug_1000.csv')
df_top = pd.read_csv('0825_top_1000.csv')
df_mid = pd.read_csv('0825_mid_1000.csv')
df_spt = pd.read_csv('0825_spt_1000.csv')
df_adc = pd.read_csv('0825_adc_1000.csv')

to_extract = ['kda', 'tier', 'dpm', 'win']
df_list = []
df_id_list = []

for j in range(0, len(df_jug), 2):  # for j in range(0, len(df_jug), 2):
     # blue team
    df_team_list = []
    df_team_id_list = []
    data_table = []
    for item in to_extract:
        data_table.append([df_jug.iloc[j][item], df_top.iloc[j][item], df_mid.iloc[j][item],
                           df_spt.iloc[j][item], df_adc.iloc[j][item]])

    df = pd.DataFrame(data_table, columns=['JUG', 'TOP', 'MID', 'SPT', 'ADC'], index=to_extract)
    df_team_list.append(df)
    df_team_id_list.append(df_jug.iloc[j]['tid'])

    data_table = []
    for item in to_extract:
        data_table.append([df_jug.iloc[j+1][item], df_top.iloc[j+1][item], df_mid.iloc[j+1][item],
                            df_spt.iloc[j+1][item], df_adc.iloc[j+1][item]])
    df = pd.DataFrame(data_table, columns=['JUG', 'TOP', 'MID', 'SPT', 'ADC'], index=to_extract)
    df_team_list.append(df)
    df_team_id_list.append(df_jug.iloc[j+1]['tid'])
    df_team = pd.concat(df_team_list, keys=df_team_id_list, names=['team', 'feature'])
    df_list.append(df_team)
    df_id_list.append(df_jug.iloc[j]['id'] // 10)


df_final = pd.concat(df_list, keys=df_id_list, names=['id'])
df_final.to_pickle('0825_game_data_1000.pkl')
print(df_final)

