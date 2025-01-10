#!/usr/bin/env python
# Created by "Thieu" at 10:02, 16/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import platform
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from src.models.problem import get_model_history


def draw_model_figure(model, path_save="results"):
    list_completed_tasks, list_total_benefits, list_global_completed_tasks, list_global_total_benefits = get_model_history(model)

    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)
    plt.close()

    plt.plot(list_completed_tasks, label="list_completed_tasks")
    plt.legend()
    plt.savefig(f"{path_save}/list_completed_tasks.png", dpi=300)
    plt.close()

    plt.plot(list_total_benefits, label="list_total_benefits")
    plt.legend()
    plt.savefig(f"{path_save}/list_total_benefits.png", dpi=300)
    plt.close()

    plt.plot(list_global_completed_tasks, label="list_global_completed_tasks")
    plt.legend()
    plt.savefig(f"{path_save}/list_global_completed_tasks.png", dpi=300)
    plt.close()

    plt.plot(list_global_total_benefits, label="list_global_total_benefits")
    plt.legend()
    plt.savefig(f"{path_save}/list_global_total_benefits.png", dpi=300)
    plt.close()

    ## You can access them all via object "history" like this:
    model.history.save_global_best_fitness_chart(filename=f"{path_save}/gbfc")
    model.history.save_local_best_fitness_chart(filename=f"{path_save}/lbfc")
    model.history.save_runtime_chart(filename=f"{path_save}/rtc")

    with open(f"{path_save}/model.pkl", "wb") as file:
        pickle.dump(model, file)


def draw_result_trials(list_result_trials, y_label=None, title=None, filename=None,
                       exts=(".png", ".pdf"), path_save="results", verbose=False):
    # Plot the results for each trial
    epoch = list(range(1, len(list_result_trials[0])+1))
    for idx, res in enumerate(list_result_trials):
        plt.plot(epoch, res, label=f'Trial {idx + 1}')

    # Set labels and title
    plt.xlabel('Generations')
    plt.ylabel(y_label)
    plt.title(title)

    # Add legend
    plt.legend()

    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)
    for idx, ext in enumerate(exts):
        plt.savefig(f"{path_save}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_average_trials(list_results, y_label=None, legends=None, title=None, filename=None,
                        exts=(".png", ".pdf"), path_save="results", verbose=False):
    # Plot the results for each trial
    colors = [
     "green", "purple",
     "brown", "gray", "blue","red" ,"cyan", "magenta", "lime"
    ]
    epoch = list(range(1, len(list_results[0]) + 1))
    for idx, res in enumerate(list_results):
        plt.plot(epoch, res, label=f'{legends[idx]}', color = colors[idx])

    # Set labels and title
    plt.xlabel('Generations')
    plt.ylabel(y_label)
    plt.title(title)

    # Add legend
    plt.legend()

    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)
    for idx, ext in enumerate(exts):
        plt.savefig(f"{path_save}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()

def draw_bar(drl, metaheuristics, types ='', y_label = ''):
    if len(drl) >len(metaheuristics):
        drl = drl[0:len(metaheuristics)]        
    x = [f'trail-{i+1}' for i in range(len(drl))]
    width = 0.35
    x_pos = np.arange(len(x))
    plt.bar(x_pos - width/2, drl, width=width, label='MA-DDQN', align='center', color = 'r')
    plt.bar(x_pos + width/2, metaheuristics, width=width, label='CCG-ARO', align='center', color = 'blue')

    # plt.xlabel('Trails')
    plt.xticks(x_pos, x, rotation=45)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")
    plt.savefig(f"drl-meta-heuristic-{types}.png", dpi = 300)
    plt.close()
    
def draw_bar_rand_greedy(drl, random, greedy, types='', y_label=''):
    # Đảm bảo rằng tất cả các danh sách có độ dài bằng nhau
    if len(drl) > len(random):
        drl = drl[:len(random)]
    if len(random) > len(greedy):
        greedy = greedy[:len(random)]
    
    # Tạo nhãn cho trục x
    x = [f'trail-{i+1}' for i in range(len(drl))]
    x_pos = np.arange(len(x))  # Các vị trí trên trục x cho các nhóm

    width = 0.25  # Điều chỉnh chiều rộng của các cột để chúng không trùng lên nhau

    # Vẽ các cột
    plt.bar(x_pos - width, drl, width=width, label='MA-DDQN', align='center', color='r')
    plt.bar(x_pos, random, width=width, label='Random', align='center', color='b')
    plt.bar(x_pos + width, greedy, width=width, label='Greedy Distance', align='center', color='g')

    # Đặt các nhãn cho trục x và y
    plt.xticks(x_pos, x, rotation=45)
    plt.ylabel(y_label)

    # Thêm chú thích
    plt.legend(loc="upper left")

    # Lưu ảnh vào file
    plt.savefig(f"drl-compare-{types}.png", dpi=300)
    plt.close()