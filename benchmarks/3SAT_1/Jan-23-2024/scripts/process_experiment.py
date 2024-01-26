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
        stripped_line = f_lines[0].strip().split(' ')
        nodes = int(stripped_line[2])
        edges = int(stripped_line[3])
        bits = int(line.split('_')[0])
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
        data.append({'filename':filename, 'bits':bits, 'treewidth':treewidth, 'construction_time':construction_time, 'tree_time':tree_time, 'contraction_time':contraction_time, 'total_time':total_time, 'solution_count':solution_count, 'nodes':nodes, 'edges':edges})

df = pd.DataFrame(data)
df.to_csv('../results/curated_result.csv')
colors = mpl.cm.get_cmap('RdPu')(np.linspace(0.2, 1, 17))

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

def func_sqrt(x, a, b):
    return a*np.sqrt(x) + 2**(-6)

def func_lin(x, a, b):
    return a*(x) + 2**(-6)

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

data_complexity_1_P = []
for j, i in enumerate(list(df['bits'].unique())):
    data_complexity_1_P.append({'bits': df[df['bits'] == i]['bits'], 'time':1/(df[df['bits'] == i]['solution_count']/2**i), '1_P':np.exp((df[df['bits'] == i]['treewidth']-1) * df[df['bits'] == i]['nodes'])})
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), np.exp((df[df['bits'] == i]['treewidth']-1)) * df[df['bits'] == i]['nodes'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)

plt.ylabel('Time complexity')
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

df_complexity_1_P = pd.DataFrame(data_complexity_1_P)
df_complexity_1_P.to_csv('../results/complexity_1_P.csv')
print(df_complexity_1_P.head().to_markdown())

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['construction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to construct (s)')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to construct on all bits')
plt.savefig('../results/satisfiability_construction_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['construction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to construct (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
#plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
#plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to construct on all bits')
plt.savefig('../results/satisfiability_construction_time_2')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['tree_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Tree time (s)')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs tree time on all bits')
plt.savefig('../results/satisfiability_tree_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['tree_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Tree time (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.title('Satisfiability vs tree time on all bits')
plt.savefig('../results/satisfiability_tree_time_2')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['contraction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to contract (s)')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to contract on all bits')
plt.savefig('../results/satisfiability_contraction_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['contraction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to contract (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
#plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
#plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to contract on all bits')
plt.savefig('../results/satisfiability_contraction_time_2')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to complete (s)')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs total time on all bits')
plt.savefig('../results/satisfiability_total_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to complete (s)')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
#plt.xticks([4**(i+1) for i in range(9)], ['']+[4**(i+1) for i in range(1,9)])
#plt.yticks([2**(i-6) for i in range(6)], ['$2^{{ {0} }}$'.format(i-6) for i in range(6)])
plt.grid()
plt.legend()
plt.title('Satisfiability vs total time on all bits')
plt.savefig('../results/satisfiability_total_time_2')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['edges']/df[df['bits'] == i]['nodes'], df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.axvline(3.8, color='black', linestyle='--')
plt.axvline(4.26, color='black', linestyle='--')
plt.ylabel('Time to complete (s)')
plt.xlabel('Density')
plt.yscale('log')
plt.grid()
plt.legend()
plt.title('Density vs time to complete')
plt.savefig('../results/density_time')
plt.clf()




#for j, i in enumerate(list(df['bits'].unique())):
#    plt.scatter(i, df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)

for j, i in enumerate(list(df['bits'].unique())):
    density = df[df['bits'] == i]['edges']/df[df['bits'] == i]['nodes']
    total_time = df[df['bits'] == i]['total_time']
    plt.plot(np.unique(density), np.poly1d(np.polyfit(density, total_time, 2))(np.unique(density)), label='{0} bits'.format(i))
    #plt.plot(df[df['bits'] == i]['edges']/df[df['bits'] == i]['nodes'], df[df['bits'] == i]['total_time'])
plt.xlabel('Density')
plt.ylabel('Time to complete')
plt.grid()
plt.legend()
plt.title('Density vs time to complete per number of nodes')
plt.savefig('../results/density_time_nodes')
plt.clf()

plt.plot(list(df['bits'].unique()), [np.mean(df[df['bits'] == i]['total_time']) for i in list(df['bits'].unique())])
plt.ylabel('Mean time to complete')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean time to complete')
plt.savefig('../results/nodes_mean_time')
plt.clf()

plt.plot(list(df['bits'].unique()), [np.mean(df[df['bits'] == i]['total_time']/(df[df['bits'] == i]['edges']/df[df['bits'] == i]['nodes'])) for i in list(df['bits'].unique())])
plt.ylabel('Mean time to complete over mean density')
plt.xlabel('Number of nodes')
plt.grid()
plt.title('Nodes vs mean time to complete over mean density')
plt.savefig('../results/nodes_mean_time_density')
plt.clf()
