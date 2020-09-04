import pandas as pd


from local_datasets import load_banana, load_vertebral_column_2c, load_ripley, \
        load_german_stalog, load_habermans_survivor, load_statlog_heart, \
        load_ionosphere, load_breast_cancer, load_two_moon, load_pima_indians

datasets = {
    'BAN' : load_banana(),
    'BCW' : load_breast_cancer(),
    'GER' : load_german_stalog(),
    'HEA' : load_statlog_heart(),
    'HAB' : load_habermans_survivor(),
    'ION' : load_ionosphere(),
    'PID' : load_pima_indians(),
    'RIP' : load_ripley(),
    'TMN' : load_two_moon(),
    'VCP' : load_vertebral_column_2c()
}


bb_res = pd.read_csv('results_thesis.csv', index_col=0)


from ismael_fuaz import draw_cd_diagram


# normalize with respect to full-mlm: the  ``max(by_dataset['norm'])``
for dataset_name in datasets.keys():
    by_dataset = bb_res[bb_res['dataset_name'] == dataset_name]
    norm_values_by_dataset = 1 - by_dataset['norm'] / max(by_dataset['norm'])

    bb_res.loc[bb_res['dataset_name'] == dataset_name, 'norm'] = norm_values_by_dataset


print(bb_res)
bb_res.to_csv('for_R.csv', index=False)

print('cd diagram for norm')
draw_cd_diagram(df_perf=bb_res, title='norm', labels=False, column='norm', filename='cd-diagram.pdf')


print('cd diagram for acc')
draw_cd_diagram(df_perf=bb_res, title='acc', labels=False, column='accuracy', filename='acc-diagram.pdf')



