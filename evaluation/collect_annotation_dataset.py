import json
from tqdm import tqdm

if __name__ == "__main__":

    for dataset in tqdm(['dailydialog', 'convai2', 'empatheticdialogues']):
        for model in tqdm(['transformer_generator', 'transformer_ranker']):
            data_ctx_folder = f'eval_data/{dataset}/{model}/human_ctx.txt'
            data_res_folder = f'eval_data/{dataset}/{model}/human_ref.txt'
            data_hyp_folder = f'eval_data/{dataset}/{model}/human_hyp.txt'
            score_folder = f'human_score/{dataset}/{model}/human_score.txt'

            with open(data_ctx_folder) as f:
                ctx = [line.strip() for line in f.readlines() if line.strip()]
            with open(data_res_folder) as f:
                res = [line.strip() for line in f.readlines() if line.strip()]
            with open(data_hyp_folder) as f:
                hyp = [line.strip() for line in f.readlines() if line.strip()]
            with open(score_folder) as f:
                score = [float(line.strip()) for line in f.readlines() if line.strip()]
            assert len(ctx) == len(res) == len(score) == len(hyp)

            with open(f'{dataset}_{model}.txt', 'w') as f:
                for c, r, h, s in zip(ctx, res, hyp, score):
                    item = {'context': c, 'ground_truth': r, 'response': h, 'score': s}
                    string = json.dumps(item)
                    f.write(f'{string}\n')
