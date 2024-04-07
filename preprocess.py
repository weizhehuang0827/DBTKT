from dataset import assist2009, assist2017, assist2012, algebra2005, nips2020, aaai2023, ednet
import fire

def preprocess(dataset=''):
    if dataset == 'assist2017':
        assist2017('data/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv', encoding='utf-8', save=True, save_argument=False)
    elif dataset == 'assist2012':
        assist2012('data/2012-2013-data-with-predictions-4-final/2012-2013-data-with-predictions-4-final.csv', encoding='utf-8', save=True, save_argument=False)
    elif dataset == 'nips2020':
        nips2020('data/NIPS2020/public_data/train_data/train_task_3_4.csv', 'data/NIPS2020/public_data/metadata', "task_3_4", 'data/processed/nips2020/result.txt', encoding='utf-8', save=True, save_argument=False)
if __name__ == '__main__':
    fire.Fire(preprocess)