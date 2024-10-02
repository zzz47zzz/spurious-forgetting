import json
import os


class Collector:
    def __init__(self, ):
        self.result_dict = dict()
        self.output_path = './temp/config_v0903_result.json'
        if os.path.exists(self.output_path):
            self.result_dict = json.load(open(self.output_path, 'r'))

    def collect(self,
                task_id: int,
                total_step: int,
                before_recovery_step_interval: list[int],
                after_recovery_step_interval: list[int]):
        assert len(before_recovery_step_interval) == len(set(before_recovery_step_interval))
        assert len(after_recovery_step_interval) == len(set(after_recovery_step_interval))
        fine_tuning_output_dir = os.path.join(
            './model/gpt-neox/processed_0720_v0730/config_v0903/multi5_permute_fullname',
            f'task{task_id}_fine_tuning_{total_step}step'
        )
        current_task_first_token_accuracy_file_path = os.path.join(
            fine_tuning_output_dir, 'train__step__1000__first_token_accuracy_history.json')
        previous_task_first_token_accuracy_file_path = os.path.join(
            fine_tuning_output_dir, f'task_{task_id - 1}__step__1000__first_token_accuracy_history.json')
        current_task_first_token_accuracy = json.load(open(current_task_first_token_accuracy_file_path, 'r'))
        previous_task_first_token_accuracy = json.load(open(previous_task_first_token_accuracy_file_path, 'r'))
        task_result = {'before_recovery': dict(), 'after_recovery': dict()}
        for step_id in before_recovery_step_interval:
            task_result['before_recovery'][str(step_id)] = dict()
            for task_type, accuracy_dict in [
                ('current', current_task_first_token_accuracy),
                ('previous', previous_task_first_token_accuracy)
            ]:
                attribute_sum, hard_correct_sum, soft_correct_sum = 0, 0, 0
                for _, value in accuracy_dict[str(step_id)].items():
                    attribute_sum += value['total']
                    hard_correct_sum += value['hard_correct']
                    soft_correct_sum += value['soft_correct']
                task_result['before_recovery'][str(step_id)].update({
                    f'{task_type}_task_hard_first_token_accuracy_average': hard_correct_sum / attribute_sum,
                    f'{task_type}_task_soft_first_token_accuracy_average': soft_correct_sum / attribute_sum,
                })
        for step_id in after_recovery_step_interval:
            task_result['after_recovery'][str(step_id)] = dict()
            recovery_result_dir = os.path.join(fine_tuning_output_dir, f'checkpoint-{step_id}', 'recovery')
            previous_task_train_exact_match_accuracy_dict = json.load(
                open(os.path.join(recovery_result_dir, 'train__exact_match_accuracy_result.json'), 'r')
            )
            previous_task_test_exact_match_accuracy_dict = json.load(
                open(os.path.join(recovery_result_dir, f'task_{task_id - 1}__exact_match_accuracy_result.json'), 'r')
            )
            for data_type, accuracy_dict in [
                ('train', previous_task_train_exact_match_accuracy_dict),
                ('test', previous_task_test_exact_match_accuracy_dict)
            ]:
                attribute_sum, correct_sum = 0, 0
                for _, value in accuracy_dict.items():
                    attribute_sum += value['total']
                    correct_sum += value['correct']
                task_result['after_recovery'][str(step_id)].update({
                    f'previous_task_{data_type}_exact_match_accuracy_average': correct_sum / attribute_sum
                })
        if str(task_id) in self.result_dict:
            assert self.result_dict[str(task_id)] == task_result
        self.result_dict[str(task_id)] = task_result

    def save(self):
        json.dump(self.result_dict, open(self.output_path, 'w'), indent=2)


def main():
    collector = Collector()
    collector.collect(1,
                      62500,
                      list(range(5, 201, 5)) + list(range(1000, 62501, 1000)),
                      list(range(5, 201, 5)),
                      )
    # collector.collect(2,
    #                   62500,
    #                   list(range(5, 201, 5)) + list(range(1000, 62501, 1000)),
    #                   list(range(5, 201, 5)) + list(range(6250, 62501, 6250)),
    #                   )
    # collector.collect(3,
    #                   62500,
    #                   list(range(5, 201, 5)) + list(range(1000, 62501, 1000)),
    #                   list(range(5, 201, 5)) + list(range(6250, 62501, 6250)),
    #                   )
    collector.save()


if __name__ == '__main__':
    main()
