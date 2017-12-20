
#求number以内所有的质数


def PrimeNumber(number):
    result = []
    for i in range(2, number):
        for j in range(2, i+1):
            if i % j == 0:
                break
        if j == i:
            result.append(i)
    return result


print(PrimeNumber(10))