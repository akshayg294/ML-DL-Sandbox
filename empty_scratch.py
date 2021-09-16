ldataset = list()
ltargetset = list()

for i in range(len(finallist)):
    out = finallist[i]
    labeltim = timelist[indlist[i]]
    label = labellist[indlist[i]]
    ind = out.index[out.label != 'no_event'].tolist()[0]
    out['apparent_power'] = (out.apparent_power_S1 + out.apparent_power_S2 + out.apparent_power_S3)/1000
    out['mean_app_power'] = out.apparent_power.rolling(30).mean()
    for lab_ind in range(ind-20, ind):
        data = out[lab_ind-20:lab_ind+60]
        data['norm_main_power'] = data.mains_power - data.before_mean.tolist()[0]
        data['norm_apparent_power'] = data.apparent_power - data.mean_app_power.tolist()[0]
        data = data[['mains_power', 'apparent_power', 'norm_main_power', 'norm_apparent_power']].values
        ldataset.append(data)
        ltargetset.append(label)
