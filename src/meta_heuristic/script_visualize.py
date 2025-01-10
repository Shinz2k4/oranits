#!/usr/bin/env python
# Created by "Thieu" at 09:53, 16/06/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from models.problem import get_model_history
from models.visualize import draw_result_trials, draw_average_trials, draw_bar, draw_bar_rand_greedy
from configs.config import ParaConfig
from configs.systemcfg import mission_cfg
import pickle
import os, sys
import pandas as pd
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 

def draw_results():

    list_task_averages = []
    list_benefit_averages = []
    list_fitness_averages = []
    list_legends = []
    for algorithm in ParaConfig.models:

        list_global_tasks = []
        list_global_benefits = []
        list_global_fits = []
        for trial in range(0, ParaConfig.N_TRIALS):
            # Load the pickled model from a file
            with open(f"{ParaConfig.PATH_SAVE}/trial-{trial}/{algorithm['name']}/model.pkl", 'rb') as file:
                model = pickle.load(file)

            list_completed_tasks, list_total_benefits, list_global_completed_tasks, list_global_total_benefits = get_model_history(model)
            list_global_tasks.append(list_global_completed_tasks)
            list_global_benefits.append(list_global_total_benefits)
            list_global_fits.append(model.history.list_global_best_fit)

        draw_result_trials(list_global_fits, y_label="Fitness", title="Fitness value",
                                         filename="fitness_values", path_save=f"{ParaConfig.PATH_SAVE}/visualize/{algorithm['name']}")
        draw_result_trials(list_global_tasks, y_label="The number of completed tasks", title="Number of completed tasks",
                                         filename="completed_tasks", path_save=f"{ParaConfig.PATH_SAVE}/visualize/{algorithm['name']}")
        draw_result_trials(list_global_benefits, y_label="Total benefits", title="Total of benefits",
                                         filename="benefit_values", path_save=f"{ParaConfig.PATH_SAVE}/visualize/{algorithm['name']}")

        list_benefit_averages.append(np.mean(list_global_benefits, axis=0))
        list_task_averages.append(np.mean(list_global_tasks, axis=0))
        list_fitness_averages.append(np.mean(list_global_fits, axis=0))
        list_legends.append(algorithm["name"])

    draw_average_trials(list_benefit_averages, y_label="Average benefits", legends=list_legends, title="",
                        filename="benefits", path_save=f"{ParaConfig.PATH_SAVE}/visualize")
    draw_average_trials(list_task_averages, y_label="Average completed tasks", legends=list_legends, title="",
                        filename="tasks", path_save=f"{ParaConfig.PATH_SAVE}/visualize")
    draw_average_trials(list_fitness_averages, y_label="Average fitness value", legends=list_legends, title="",
                        filename="fitness", path_save=f"{ParaConfig.PATH_SAVE}/visualize")


def count_tasks_in_epochs_and_sum(file_path):
    epoch_start_pattern = r"---------> reset " 
    epoch_end_pattern = r"---------> reset True"
    task_pattern = r"-------> \d+ (\d+).*?\[(.*?)\]"
    with open(file_path, 'r') as file:
        data = file.read()
    epoch_starts = re.finditer(epoch_start_pattern, data)
    # epoch_starts = list(re.finditer(epoch_start_pattern, data))
    task_matches = re.finditer(epoch_end_pattern, data)

    end_positions = [match.start() for match in task_matches]
    
    if len(end_positions) > 1:
        end_signal = end_positions[1]
        data = data[0:end_signal]
  
    task_matches_in_epoch = re.findall(task_pattern, data, re.MULTILINE)
    tasks = list(set([int(match[0]) for match in task_matches_in_epoch]))
    benefits = []
    cnt = 0
    for task in tasks:
        benefit = 0
        for i in range(task):
            benefit +=float(task_matches_in_epoch[cnt][1])
            cnt += 1
        benefits.append(benefit) 
    return max(tasks), max(benefits)

def cal_avg_predict_time_drl(num):
    avg_tasks = []
    avg_benefits = []
    for i in range(num):
        path = f"{ParaConfig.EVAL_PATH_SAVE}/drl_meta_result_{i}.txt"
        task, benefit = count_tasks_in_epochs_and_sum(path)
        avg_tasks.append(task)
        avg_benefits.append(benefit)
    return avg_tasks, avg_benefits


def draw_eval_results():

    list_results = os.listdir(ParaConfig.EVAL_PATH_SAVE)
    list_results = sorted(list_results)
    for algorithm in ParaConfig.models:
        avg_global_tasks = []
        avg_global_benefits = []
        avg_global_fits = []
        for data in list_results:
            if not os.path.isdir(f"{ParaConfig.EVAL_PATH_SAVE}/{data}"):
                continue
            global_tasks = []
            global_benefits = []
            global_fits = []
            for trial in range(0, ParaConfig.N_TRIALS):
                # Load the pickled model from a file
                try:
                    with open(f"{ParaConfig.EVAL_PATH_SAVE}/{data}/trial-{trial}/{algorithm['name']}/model.pkl", 'rb') as file:
                        model = pickle.load(file)
                except:
                    continue
                list_completed_tasks, list_total_benefits, list_global_completed_tasks, list_global_total_benefits = get_model_history(model)
                global_tasks.append(list_global_completed_tasks[-1])
                global_benefits.append(list_global_total_benefits[-1])
                global_fits.append(model.history.list_global_best_fit[-1])
            avg_global_tasks.append(np.mean(global_tasks))
            avg_global_benefits.append(np.mean(global_benefits))
            avg_global_fits.append(np.mean(global_fits))
        # df = count_tasks_in_epochs_and_sum('drl_meta_result.txt')
        # print(df['completed_tasks'])
        # print(df['benefits'])
        # draw_bar(df['completed_tasks'], avg_global_tasks, types="tasks",y_label = "The number of completed tasks")
        # draw_bar(df['benefits'], avg_global_benefits, types="benefits", y_label= "Benefits")
        # draw_bar(df['benefits'], avg_global_fits, types="total_benefits", y_label= "Benefits")
        avg_tasks, avg_benefits = cal_avg_predict_time_drl(10)
        draw_bar(avg_tasks, avg_global_tasks, types="tasks",y_label = "The number of completed tasks")
        # draw_bar(df['benefits'], avg_global_benefits, types="benefits", y_label= "Benefits")
        draw_bar(avg_benefits, avg_global_fits, types="total_benefits", y_label= "Benefits")

def read_data_from_file_greedy_random(path):
    with open(path, 'r') as file:
        file_data = file.read()
    pattern_benefit_task_profit = r"(\d+\.\d+)\s+(\d+)\s+(\d+)"
    matches_benefit_task_profit = re.findall(pattern_benefit_task_profit, file_data)
    data = []
    for match in matches_benefit_task_profit:
        benefit, completed_tasks, profit = map(float, match)        
        data.append([benefit, completed_tasks, profit])
    df = pd.DataFrame(data, columns=["Total Benefits", "Completed Tasks", "Profit"])
    return df
def draw_compared_greedy_random_drl():
    greedy_path = 'greedy_distance_result_.txt'
    random_path = 'random_result_.txt'
    df_greedy = read_data_from_file_greedy_random(f"{ParaConfig.EVAL_PATH_SAVE}/{greedy_path}")
    df_random = read_data_from_file_greedy_random(f"{ParaConfig.EVAL_PATH_SAVE}/{random_path}")
    avg_tasks, avg_benefits = cal_avg_predict_time_drl(10)
    draw_bar_rand_greedy(avg_tasks, df_random['Completed Tasks'], df_greedy['Completed Tasks'], types="tasks",y_label = "The number of completed tasks")
    draw_bar_rand_greedy(avg_benefits, df_random['Total Benefits'], df_greedy['Total Benefits'], types="total_benefits",y_label = "Benefits")
    
    