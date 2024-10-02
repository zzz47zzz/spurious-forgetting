import wandb
from training.utility import set_seed
import random


def get_random_table(seed, x_begin, x_end, x_axis_name, y_axis_name):
    x_axis = [i for i in range(x_begin, x_end)]
    y_axis = [random.random() for _ in range(1000)]
    data = [[x, y] for x, y in zip(x_axis, y_axis)]
    table = wandb.Table(data=data, columns=[x_axis_name, y_axis_name])
    return table


def log_task_pretraining(task_id):
    set_seed(task_id)
    first_token_accuracy__soft__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__hard__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__soft__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__hard__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    for table_title in ["first_token_accuracy__soft__birthday", "first_token_accuracy__hard__birthday",
                        "first_token_accuracy__soft__birth_city", "first_token_accuracy__hard__birth_city"]:
        table = eval(table_title)
        wandb.log({
            table_title: wandb.plot.line(table,
                                         table.columns[0],
                                         table.columns[1],
                                         title=table_title),
        })


def log_task_fine_tuning(task_id):
    set_seed(task_id)
    first_token_accuracy__soft__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__hard__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__soft__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    first_token_accuracy__hard__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    exact_match__soft__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    exact_match__hard__birthday = get_random_table(task_id, 0, 10, "step", "accuracy")
    exact_match__soft__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    exact_match__hard__birth_city = get_random_table(task_id, 0, 10, "step", "accuracy")
    for table_title in ["first_token_accuracy__soft__birthday", "first_token_accuracy__hard__birthday",
                        "first_token_accuracy__soft__birth_city", "first_token_accuracy__hard__birth_city",
                        "exact_match__soft__birthday", "exact_match__hard__birthday",
                        "exact_match__soft__birth_city", "exact_match__hard__birth_city"]:
        table = eval(table_title)
        wandb.log({
            table_title: wandb.plot.line(table,
                                         table.columns[0],
                                         table.columns[1],
                                         title=table_title),
        })


def update_wandb_config():
    # https://docs.wandb.ai/guides/track/public-api-guide#update-config-for-an-existing-run
    api = wandb.Api()
    run = api.run("xidicai067/forgetting/c529qvl0")
    run.config["all_config"] = {'wandb': {
        'continual_learning_exp_id': 'processed_0720_v0730__v0806_v0806__multi5_permute_fullname',
        'phase': 'pre_training',
        'run_name': 'task_0'
    }}
    run.update()


def main():
    # https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA
    wandb.login(key='5139c64ae54ccc30c6ab755a670a5d35a2666560')

    wandb.init(project='debug-ignore', name=f'task_0', group='g2', job_type='pre_training')
    log_task_pretraining(0)
    wandb.finish()

    wandb.init(project='debug-ignore', name=f'task_1', group='g2', job_type='pre_training')
    log_task_pretraining(1)
    wandb.finish()

    wandb.init(project='debug-ignore', name=f'task_0_0', group='g2', job_type='fine_tuning', config={
        'beginning': 'task_0',
        'tuning': 'task_0'
    })
    log_task_fine_tuning(0)
    wandb.finish()

    wandb.init(project='debug-ignore', name=f'task_1_0', group='g2', job_type='fine_tuning', config={
        'beginning': 'task_1',
        'tuning': 'task_0'
    })
    log_task_fine_tuning(1)
    wandb.finish()

    wandb.init(project='debug-ignore', name=f'task_1_1', group='g2', job_type='fine_tuning', config={
        'beginning': 'task_1',
        'tuning': 'task_1'
    })
    log_task_fine_tuning(2)
    wandb.finish()


if __name__ == '__main__':
    # main()
    update_wandb_config()
