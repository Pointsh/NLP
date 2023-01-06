def solution(s):
    equl=0
    n_equl=0
    char=''
    result=0
    for i in s:
        if equl==n_equl:
            result+=1
            char=i
        if i==char:
            equl+=1
        else:
            n_equl+=1
    return result



    
    
        
        

