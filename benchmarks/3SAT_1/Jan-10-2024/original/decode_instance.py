f = open("11_bit_instance.dat","r")
lines = f.readlines()
num_instance = len(lines)
print(num_instance)

clauses = []
signs = []
for i in range(num_instance):
    clauses.append([])
    signs.append([])
    line = lines[i].strip().split()
    str_length = len(line)
    num_clause = (str_length -3)//2
    k = 2
    for j in range(3,3+num_clause):
        k += 1
        k %= 3
        if k == 0:
            clauses[i].append([])
            signs[i].append([])
        clauses[i][-1].append(int(''.join(c for c in line[j] if (c.isdigit() or c =='+' or c =='-'))))
        signs[i][-1].append(int(''.join(c for c in line[j+num_clause] if (c.isdigit() or c =='+' or c =='-'))))
    print(clauses[i])
    print(signs[i])
