import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit

data = []

f = open('../results/result.txt', 'r')
lines = f.readlines()
for i, line in enumerate(lines):
    if i % 15 == 0:
        print('line', line)
        filename = line.strip()
        f = open('../finished_CNF/'+filename)
        f_lines = f.readlines()
        stripped_line = f_lines[6].strip().split(' ')
        nodes = int(stripped_line[2])
        edges = int(stripped_line[3])
        bits = int(line.split('_')[0])
        instance = int(line.split('_')[-1].split('.')[0])
    if i % 15 == 3:
        treewidth = float(line.split(': ')[1].strip())
    if i % 15 == 8:
        construction_time = float(line.split(': ')[1].strip())
    if i % 15 == 9:
        tree_time = float(line.split(': ')[1].strip())
    if i % 15 == 10:
        contraction_time = float(line.split(': ')[1].strip())
    if i % 15 == 12:
        total_time = float(line.split(': ')[1].strip())
    if i % 15 == 13:
        solution_count = float(line.split(': ')[1].strip())
    if i % 15 == 14:
        data.append({'filename':filename, 'bits':bits, 'instance':instance, 'treewidth':treewidth, 'construction_time':construction_time, 'tree_time':tree_time, 'contraction_time':contraction_time, 'total_time':total_time, 'solution_count':solution_count, 'nodes':nodes, 'edges':edges})

df = pd.DataFrame(data)
df.to_csv('../results/curated_result.csv')
colors = mpl.cm.get_cmap('YlGn')(np.linspace(0.2, 1, 17))

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count'], df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to complete (s)')
plt.xlabel('Solution count')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Solution count vs time to complete on all bits')
plt.savefig('../results/time_solution')
plt.clf()

'''satisfiability_x = []
time_y = []
for j, i in enumerate(list(df['bits'].unique())):
    #satisfiability_x += list(1/(df[df['bits'] == i]['solution_count']/2**i))
    #print('first',satisfiability_x)
    #time_y += list(df[df['bits'] == i]['total_time'])
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
#plt.axline((4, 2), 5)
#y = func(np.unique(satisfiability_x))
#plt.plot(np.unique(satisfiability_x), y)
#satisfiability_x = np.array(satisfiability_x)
#time_y = np.array(time_y)

#popt, pcov = curve_fit(func1, satisfiability_x, time_y)
#plt.plot(np.unique(satisfiability_x), func1(np.unique(satisfiability_x), *popt))
'''

def func_sqrt(x, b):
    return np.sqrt(x) + b

def func_lin(x, b):
    return x + b

min_1_P = 10**6
max_1_P = 10**8
data_complexity_1_P = []
for j, i in enumerate(list(df['bits'].unique())):
    for k in list(df['instance'].unique()):
        data_complexity_1_P.append({'bits': df[(df['bits'] == i) & (df['instance'] == k)]['bits'], 'instance':df[(df['bits'] == i) & (df['instance'] == k)]['instance'], 'treewidth':df[(df['bits'] == i) & (df['instance'] == k)]['treewidth'], 'time':1/(df[(df['bits'] == i) & (df['instance'] == k)]['solution_count']/2**i), '1_P':df[(df['bits'] == i) & (df['instance'] == k)]['total_time']})
            # satisfiability_x += list(1/(df[df['bits'] == i]['solution_count']/2**i))
            # time_y += list(df[df['bits'] == i]['total_time'])
    min_1_P_i = min(1/(df[df['bits'] == i]['solution_count']/2**i))
    #max_1_P_i = max(1/(df[df['bits'] == i]['solution_count']/2**i))
    if min_1_P_i < min_1_P:
        min_1_P = min_1_P_i
    #if max_1_P_i > max_1_P:
    #    max_1_P = max_1_P_i
    plt.scatter(1/(df[(df['bits'] == i)]['solution_count']/2**i), np.exp((df[df['bits'] == i]['treewidth']-1)) * df[df['bits'] == i]['nodes'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)

print(min_1_P, max_1_P)
x_axis = np.linspace(min_1_P, max_1_P, 2)
y_axis_lin = func_lin(x_axis, 0)
#y_axis_sqrt = func_sqrt(x_axis, 0)
plt.plot(x_axis, y_axis_lin, '--', label='$p = 1/P$', c='purple')
#plt.plot(x_axis, y_axis_sqrt, '--', label='$p = 1/\\sqrt{P}$', c='orange')

plt.ylabel('Time complexity (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
#plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
#plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.minorticks_off()
plt.title('Satisfiability vs time complexity on all bits')
plt.savefig('../results/satisfiability_complexity_inverse')
plt.clf()

data_time_1_P = []
#satisfiability_x = []
#time_y = []
for j, i in enumerate(list(df['bits'].unique())):
    for k in list(df['instance'].unique()):
        data_time_1_P.append({'bits': df[(df['bits'] == i) & (df['instance'] == k)]['bits'], 'instance':df[(df['bits'] == i) & (df['instance'] == k)]['instance'], 'time':1/(df[(df['bits'] == i) & (df['instance'] == k)]['solution_count']/2**i), '1_P':df[(df['bits'] == i) & (df['instance'] == k)]['total_time']})
        # satisfiability_x += list(1/(df[df['bits'] == i]['solution_count']/2**i))
        # time_y += list(df[df['bits'] == i]['total_time'])
    plt.scatter(1/(df[(df['bits'] == i)]['solution_count']/2**i), df[(df['bits'] == i)]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
    
x_axis = np.linspace(min_1_P, max_1_P, 2)
y_axis_lin = func_lin(x_axis, 0)
y_axis_sqrt = func_sqrt(x_axis, 0)
plt.plot(x_axis, y_axis_lin, '--', label='$p = 1/P$', c='purple')
plt.plot(x_axis, y_axis_sqrt, '--', label='$p = 1/\\sqrt{P}$', c='orange')

plt.ylabel('Time to complete (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
#plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
#plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.minorticks_off()
plt.title('Satisfiability vs time to complete on all bits')
plt.savefig('../results/satisfiability_time_2')
plt.clf()

df_time_1_P = pd.DataFrame(data_time_1_P)
df_time_1_P.to_csv('../results/time_1_P.csv')
#print(df_time_1_P.head().to_markdown())

plt.plot(list(df['bits'].unique()), [np.mean([np.mean(df[(df['bits'] == i) & (df['instance'] == k)]['total_time']) for k in list(df['instance'].unique())]) for i in list(df['bits'].unique())])
plt.ylabel('Mean time to complete (s)')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean time to complete')
plt.savefig('../results/nodes_mean_time')
plt.clf()

plt.plot(list(df['bits'].unique()), [np.mean([np.mean(df[(df['bits'] == i) & (df['instance'] == k)]['construction_time']) for k in list(df['instance'].unique())]) for i in list(df['bits'].unique())])
plt.ylabel('Mean construction time (s)')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean construction time')
plt.savefig('../results/nodes_mean_time_construction')
plt.clf()

plt.plot(list(df['bits'].unique()), [np.mean([np.mean(df[(df['bits'] == i) & (df['instance'] == k)]['tree_time']) for k in list(df['instance'].unique())]) for i in list(df['bits'].unique())])
plt.ylabel('Mean tree time (s)')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean tree time')
plt.savefig('../results/nodes_mean_time_tree')
plt.clf()

plt.plot(list(df['bits'].unique()), [np.mean([np.mean(df[(df['bits'] == i) & (df['instance'] == k)]['contraction_time']) for k in list(df['instance'].unique())]) for i in list(df['bits'].unique())])
plt.ylabel('Mean contraction time (s)')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean contraction time')
plt.savefig('../results/nodes_mean_time_contraction')
plt.clf()
