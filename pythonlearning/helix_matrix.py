a=[[1,2,3],[4,5,6],[7,8,9]]
def  HelixMatrix(matrix):
    result=[]
    while len(matrix)>=1:
        result.extend(matrix[0])
        del matrix[0]
        if len(matrix)>=2:
            for i in range(len(matrix)-1):
                result.append(matrix[i][-1])
                del matrix[i][-1]
        if len(matrix)>=1:
            result.extend(ListOverturn(matrix[-1]))
            del matrix[-1]
        if len(matrix)>=1:
            for i in range(len(matrix)):
                result.append(matrix[len(matrix)-i-1][0])
                del matrix[len(matrix)-i-1][0]
    return result


def ListOverturn(arry):
    result=[]
    for i in range(len(arry)):
        result.append(arry[len(arry)-i-1])
    return result


print(HelixMatrix(a))