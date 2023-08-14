import argparse
import collections
import os
from typing import List, Tuple, Union

from modules.block_kd.search.searcher import BlockKDSearcher
from modules.search_space.superspace import get_available_superspaces

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


FILE_DIR = os.path.dirname(__file__)


class Individual:

    def __init__(self, block_choice, width_window_choice, score, lat_loss, top_loss) -> None:
        self.block_choice = block_choice
        self.width_window_choice = width_window_choice
        self.score = score
        self.lat_loss = lat_loss
        self.top_loss = top_loss

    @property
    def arch_str(self) -> int:
        return ''.join([str(v) for v in self.block_choice]) + '_' + ''.join([str(v) for v in self.width_window_choice])

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self) -> str:
        arch_str = 'arch:' + self.arch_str
        score_str = f'score:{self.score:.4f}'
        lat_loss_str = f'lat_loss:{self.lat_loss:.2f}'
        top_loss_str = f'top_loss:{self.top_loss:.4f}'
        return '-'.join([arch_str, score_str, lat_loss_str, top_loss_str])

    def __eq__(self, other) -> bool:
        if isinstance(other, Tuple):
            block_choice, width_window_choice = other
            return list(self.block_choice) == list(block_choice) and list(self.width_window_choice) == list(width_window_choice)
        else:
            return list(self.block_choice) == list(other.block_choice) and list(self.width_window_choice) == list(other.width_window_choice)


class History:

    def __init__(self):
        self._history = []

    def append(self, indv: Individual):
        self._history.append(indv)

    def __len__(self) -> int:
        return len(self._history)

    def __contains__(self, x: Union[Individual, Tuple]) -> bool:
        return x in self._history

    def __iter__(self):
        return iter(self._history)

    def history(self) -> List[Individual]:
        return self._history


def evolution_search(args):
    searcher = BlockKDSearcher(
        lut_dir=args.lut_path, 
        superspace=args.superspace,
        platform=args.platform,
        latency_constraint=args.latency_constraint,
        lat_loss_t=args.latency_loss_t,
        lat_loss_a=args.latency_loss_a
    )

    latency_constraint_str = '+'.join([str(v) for v in args.latency_constraint])
    search_config_str = f'c{latency_constraint_str}_t{args.latency_loss_t}_a{args.latency_loss_a}'
    output_dir = os.path.join(args.output_dir, search_config_str)
    log_path = os.path.join(output_dir, 'output.log')
    distribution_fig_path = os.path.join(output_dir, '{}_distribution.png')
    os.makedirs(output_dir, exist_ok=True)

    with open(log_path, 'w') as f:
        pass 

    def log(text):
        print(text)
        with open(log_path, 'a') as f:
            f.write(text + '\n')

    population = collections.deque()
    history = History()

    log('Generating population')
    # initial specific population
    for i in range(searcher.num_block_choices):
        for j in range(max(searcher.num_width_window_choices_list)):
            block_choices, width_window_choices = [i] * searcher.num_stages, [min(j, searcher.num_width_window_choices_list[stage_idx] - 1) for stage_idx in range(searcher.num_stages)]
            score, lat_loss, top_loss = searcher.eval_individual(block_choices, width_window_choices)
            indv = Individual(block_choices, width_window_choices, score, lat_loss, top_loss)
            population.append(indv)
            history.append(indv)

    # Initialize the population with random models.
    with tqdm(total=args.population_size-len(population)) as t:
        while len(population) < args.population_size:
            block_choices, width_window_choices = searcher.random_sample()
            if (block_choices, width_window_choices) in history:
                pass
            score, lat_loss, top_loss = searcher.eval_individual(block_choices, width_window_choices)
            indv = Individual(block_choices, width_window_choices, score, lat_loss, top_loss)
            population.append(indv)
            history.append(indv)
            t.update()

    log('Initial population:')
    for indv in history:
        log(str(indv))

    log('Start Evolution Search')
    # Carry out evolution in cycles. Each cycle produces a model and removes another.
    while len(history) < args.cycles:
        sample = np.random.choice(population, args.parent_size, replace=False)

        parent = max(sample)
        # mutate block type
        child_block_arch, _ = searcher.mutate_block(parent.block_choice, parent.width_window_choice)
        if (child_block_arch, parent.width_window_choice) not in history:
            score1, lat_loss1, top_loss1 = searcher.eval_individual(child_block_arch, parent.width_window_choice)
            child1 = Individual(child_block_arch, parent.width_window_choice, score1, lat_loss1, top_loss1)
            population.append(child1)
            history.append(child1)
            population.popleft()
            log(f'Iter {len(history) - len(population):4d}  parent: {parent}  child: {child1}')

        # mutate width window
        _, child_width_window_arch = searcher.mutate_width_window(parent.block_choice, parent.width_window_choice)
        if (parent.block_choice, child_width_window_arch) not in history:
            score2, lat_loss2, top_loss2 = searcher.eval_individual(parent.block_choice, child_width_window_arch)
            child2 = Individual(parent.block_choice, child_width_window_arch, score2, lat_loss2, top_loss2)
            population.append(child2)
            history.append(child2)
            population.popleft()
            log(f'Iter {len(history) - len(population):4d}  parent: {parent}  child: {child2}')


    history = sorted(history.history(), reverse=True)
    log(f'TOP individual:')
    for i in range(50):
        max_ms, min_ms = searcher.pred_min_max_latency(history[i].block_choice, history[i].width_window_choice)
        log(str(history[i]) + f' min_ms {min_ms:.2f} max_ms {max_ms:.2f}')
    best_indiv = history[0]
    draw_latency_loss_cdf(searcher, best_indiv.block_choice, best_indiv.width_window_choice, distribution_fig_path.format(best_indiv.arch_str))
    return history


def draw_latency_loss_cdf(searcher: BlockKDSearcher, block_choice: List[int], width_window_choice: List[int], fig_path):
    loss_list, latency_list = searcher.sample_individual(block_choice, width_window_choice)
    latency_list = np.sort(latency_list)
    loss_list = np.sort(loss_list)

    # get the cdf values of y
    y = np.arange(len(loss_list)) / float(len(loss_list))
    
    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].plot(latency_list, y)
    ax[1].plot(loss_list, y)
    ax[0].set_title('latency cdf')
    ax[1].set_title('loss cdf')

    plt.savefig(fig_path)


def eval_spec_candidates(args):
    searcher = BlockKDSearcher(
        lut_dir=args.lut_path, 
        superspace=args.superspace,
        platform=args.platform,
        latency_constraint=args.latency_constraint,
        lat_loss_t=args.latency_loss_t,
        lat_loss_a=args.latency_loss_a
    )
    line = [''.join([str(v) for v in args.latency_constraint])]
    for candidate in args.supernet_choices:
        block_choice_str, width_window_choice_str = candidate.replace('_', '-').split('-')
        block_choice = [int(v) for v in block_choice_str]
        width_window_choice = [int(v) for v in width_window_choice_str]
        score, lat_loss, top_loss = searcher.eval_individual(block_choice, width_window_choice)
        indv = Individual(block_choice, width_window_choice, score, lat_loss, top_loss)
        max_ms, min_ms = searcher.pred_min_max_latency(block_choice, width_window_choice)
        print(indv, f'min_ms {min_ms:.2f} max_ms {max_ms:.2f}')
        output_fig_path = os.path.join(args.output_dir, 'cdf', f'{candidate}.png')
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        draw_latency_loss_cdf(searcher, block_choice, width_window_choice, output_fig_path)
        line.append(f'{score:.4f}')
    with open('tmp_eval_block_kd_candidates.csv', 'a') as f:
        f.write(','.join(line) + '\n')
        print('score written to tmp_eval_block_kd_candidates.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lut_path', default=os.path.join(FILE_DIR, 'data/block_lut'), type=str, help='block_kd lut path')
    parser.add_argument('--output_dir', default=os.path.join(FILE_DIR, 'checkpoints/block_kd/search'))
    parser.add_argument('--superspace', required=True, choices=get_available_superspaces())
    parser.add_argument('--platform', type=str, required=True)
    parser.add_argument('--supernet_choices', nargs='*',
                        help='The search spaces (supernets) to calculate score.'
                             'If set, only do evaluation instead of search')
    parser.add_argument('--latency_constraint', nargs='+')
    parser.add_argument('--latency_loss_t', default=0.1, type=float)
    parser.add_argument('--latency_loss_a', default=0.01, type=float)
    # --- args for evolution search ---
    parser.add_argument('--population_size', default=500, type=int, help='evolution population size')
    parser.add_argument('--parent_size', default=125, type=int, help='evolution sample size')
    parser.add_argument('--cycles', default=5000, type=int, help='evolution cycles')
    args = parser.parse_args()

    args.latency_constraint = [int(v) for v in args.latency_constraint]
    args.output_dir = os.path.join(args.output_dir, args.superspace)
    args.lut_path = os.path.join(args.lut_path, args.superspace)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.supernet_choices:
        evolution_search(args)
    else:
        eval_spec_candidates(args)


if __name__ == '__main__':
    main()
