#排序，Sort为选择的排序方式


def NumberSort(num):
    for i in range(len(num)):
        for j in range(len(num)-i-1):
            if num[j] > num[j+1]:
                tem = num[j]
                num[j] = num[j+1]
                num[j+1] = tem
    return num


def SlectSort(temp, Sort):
    return Sort(temp)


temp = [5, 6, 8, 2, 1, 4]
print(SlectSort(temp, NumberSort))