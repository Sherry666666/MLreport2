


def WordOverturn(S):
    Stemp= S.split(' ')#Stemp=list(S)
    Stemp_Overturn=[]
    for i in range(len(Stemp)):
            Stemp_Overturn.append(Stemp[len(Stemp)-i-1])

    return ' '.join(Stemp_Overturn)


print(WordOverturn('I love China!'))