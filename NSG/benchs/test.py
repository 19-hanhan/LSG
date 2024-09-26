with open('count.txt', 'r') as file:
    # 读取每一行并将其转化为整数数组
    array = [int(line.strip()) for line in file.readlines()]

print(array)
import os
os.remove('count.txt')

