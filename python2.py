def hi():
    return 2,6

print(hi())

def sum(int1=0, int2=0, int3=0):
    result = int1 + int2 + int3
    return result

print(sum(*hi()))