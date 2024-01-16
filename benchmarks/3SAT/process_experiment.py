import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data = []

f = open('result.txt', 'r')
lines = f.readlines()
for i, line in enumerate(lines):
    if i % 15 == 0:
        filename = line.strip()
        f = open('./CNF/'+filename)
        f_lines = f.readlines()
        stripped_line = f_lines[0].strip().split(' ')
        nodes = int(stripped_line[2])
        edges = int(stripped_line[3])
        bits = int(line.split('_')[0])
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
        data.append({'filename':filename, 'bits':bits, 'construction_time':construction_time, 'tree_time':tree_time, 'contraction_time':contraction_time, 'total_time':total_time, 'solution_count':solution_count, 'nodes':nodes, 'edges':edges})

df = pd.DataFrame(data)
df.to_csv('results_CNF/curated_result.csv')
colors = mpl.cm.get_cmap('RdPu')(np.linspace(0.2, 1, 8))

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count'], df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to complete')
plt.xlabel('Solution count')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Solution count vs time to complete on all bits')
plt.savefig('results_CNF/time_solution')
plt.clf()

data = []

for j, i in enumerate(list(df['bits'].unique())):
    data.append({'bits':i, 'time':1/(df[df['bits'] == i]['solution_count']/2**i), '1_P':df[df['bits'] == i]['total_time']})
    plt.scatter(1/(df[df['bits'] == i]['solution_count']/2**i), df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to complete')
plt.xlabel('$\\frac{1}{p}$')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to complete on all bits')
plt.savefig('results_CNF/satisfiability_time')
plt.clf()

df = pd.DataFrame(data)
df.to_csv('results_CNF/time_1_P.csv')

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['construction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to construct')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to construct on all bits')
plt.savefig('results_CNF/satisfiability_construction_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['tree_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Tree time')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs tree time on all bits')
plt.savefig('results_CNF/satisfiability_tree_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['solution_count']/2**i, df[df['bits'] == i]['contraction_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.ylabel('Time to contract')
plt.xlabel('Satisfiability')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.legend()
plt.title('Satisfiability vs time to contract on all bits')
plt.savefig('results_CNF/satisfiability_contraction_time')
plt.clf()

for j, i in enumerate(list(df['bits'].unique())):
    plt.scatter(df[df['bits'] == i]['edges']/df[df['bits'] == i]['nodes'], df[df['bits'] == i]['total_time'], label='{0} bits'.format(i), facecolors='none', edgecolors=colors[j], s=10)
plt.axvline(3.8, color='black', linestyle='--')
plt.axvline(4.26, color='black', linestyle='--')
plt.ylabel('Time to complete')
plt.xlabel('Density')
plt.yscale('log')
plt.grid()
plt.legend()
plt.title('Density vs time to complete')
plt.savefig('results_CNF/density_time')
plt.clf()