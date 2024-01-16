import json
import numpy as np

indices = [i for i in range(10,19)]
file_names = [str(index) + '_bit_instance.dat' for index in indices]

for i, file_name in enumerate(file_names):
    f_in = open(file_name, 'r')
    all_lines = f_in.readlines()
    for lines in all_lines:
        lines_split = lines.split('  ')
        cnf_number = lines_split[0][:-2]
        variables = lines_split[0][-2:]
        #clauses = lines_split[1]

        #matrix_string = ''.join(lines_split[3:])
        #matrix_string_split = matrix_string.split(']  [')
        #matrix1_string = matrix_string_split[0]+']'
        #matrix2_string = '['+matrix_string_split[1]

        matrix_vars = np.array(json.loads(lines_split[3]))
        matrix_signs = np.array(json.loads(lines_split[4]))
        #matrix = [[(matrix_vars[i][j] + 1) * matrix_signs[i][j] for i in range(len(matrix_vars))] for j in range(len(matrix_vars[0]))]
        matrix = matrix_signs*(matrix_vars+1)
        #matrix = matrix_vars+1

        clauses = matrix.shape[0]

        print('CNF: '+cnf_number)
        print('Variables: '+variables)
        print('Clauses: ', clauses)
        print('Matrix 1: ', matrix_vars)
        print('Matrix 2: ', matrix_signs)
        print('Matrix 3: ', matrix, matrix.shape)

        new_file_name = '../CNF/{0}_bit_{1}.cnf'.format(indices[i], cnf_number)
        f_out = open(new_file_name, 'w')
        new_lines = []
        new_lines.append( 'p cnf {0} {1}\n'.format(variables, clauses))
        for j in range(matrix.shape[0]):
            new_lines.append( '{0} {1} {2} 0\n'.format(matrix[j][0], matrix[j][1], matrix[j][2]) )
        f_out.writelines(new_lines)
        

