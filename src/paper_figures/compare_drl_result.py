import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
matplotlib.rcParams['font.size']=11
plt.style.use(["science", "ieee"])
import os
current_file_path = os.path.abspath(__file__)
pwd = os.path.dirname(current_file_path) 


# Đọc dữ liệu từ 2 file CSV
#saved_files_global_combine_decay_0.99_lr_1e-05_batch_size_512_modify_reward_False_combine_0.05_more/scores.png
path_2 = 'saved_files_global_combine_decay_0.99_lr_1e-05_batch_size_512_modify_reward_False_combine_0.05_more/_reward.csv'
path_1 = 'saved_files_global_combine_decay_0.99_lr_1e-05_batch_size_512_modify_reward_True_combine_0.1/_reward.csv'
df1 = pd.read_csv(path_1)
df2 = pd.read_csv(path_2)

df1['sum_agents'] = df1[['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']].sum(axis=1)
df2['sum_agents'] = df2[['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']].sum(axis=1)

# Cắt dữ liệu để có số dòng bằng nhau (theo file ít dòng hơn)
min_length = min(50000,min(len(df1), len(df2)))
df1 = df1.iloc[:min_length]
df2 = df2.iloc[:min_length]

window_size = 300
df1['Max_smoothed'] = df1['Max'].rolling(window=window_size, min_periods=1).mean()
df2['Max_smoothed'] = df2['Max'].rolling(window=window_size, min_periods=1).mean()
df1['sum_agents_smoothed'] = df1['sum_agents'].rolling(window=window_size, min_periods=1).mean()
df2['sum_agents_smoothed'] = df2['sum_agents'].rolling(window=window_size, min_periods=1).mean()

# Vẽ biểu đồ cho giá trị 'Max' đã mượt
plt.figure(figsize=(10,7.5))

# So sánh giá trị max đã mượt của cả 2 file
plt.plot(df1['Max_smoothed'], label='Proposed Scheme', color='b', linestyle='--')
plt.plot(df2['Max_smoothed'], label='Not Modify Reward', color='r', linestyle='-')

# Thêm nhãn và tiêu đề
# plt.title('Compare max benifi')
plt.xlabel('Epoches')
plt.ylabel('Maximum System Benifit')
plt.legend()


# Lưu và đóng biểu đồ
plt.grid(True)
plt.xlim(0,50000)
plt.savefig(pwd+"/DRL_compare_Maximum_System_Benifit.png", dpi=300)
plt.close()

# Vẽ biểu đồ cho tổng giá trị các agents đã mượt
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(6,4.75))
# So sánh tổng giá trị của các agents (Agent 0 -> Agent 5) đã mượt
plt.plot(df1['sum_agents_smoothed'], label='Proposed Scheme', color='g', linestyle='-.')
plt.plot(df2['sum_agents_smoothed'], label='Not Modify Reward', color='orange', linestyle=':')

# Thêm nhãn và tiêu đề
# plt.title('Comparison of Smoothed Max and Sum of Agents (0-5) from Two CSV Files')
plt.xlabel('Epoches', fontsize = 11)
plt.ylabel('Total System Benifit', fontsize = 11)
plt.xlim(0,50000)
plt.legend(fontsize=11)

# Lưu và đóng biểu đồ
plt.grid(True)
plt.savefig(pwd+"/DRL_compare_Total_System_Benifit.png", dpi=300)
plt.close()