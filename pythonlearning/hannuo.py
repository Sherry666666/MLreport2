#解决汉诺塔问题


def HanNuo(n, A= 'A', B= 'B', C= 'C'):
    if n == 1:
        print(A+'->'+B)
    elif n == 2:
        print(A+'->'+C)
        print(A+'->'+B)
        print(C+'->'+B)
    else:
        HanNuo(n-1, A='A', B='C', C='B')
        print(A+'->'+B)
        HanNuo(n-1, A='C', B='B', C='A')


HanNuo(3)