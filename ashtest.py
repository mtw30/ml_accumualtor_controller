print("Matt is Gay")
num1 = input("PLease enter your first number")
print(num1)
num2 = input("PLease enter your first number")
print(num2)
num3 = input("PLease enter your first number")
print(num3)
#work out the biggest number.
if num2 < num1:
    print("Number One is bigger than number two")
    if num1 > num3:
        print("Number One Is biggest:"+str(num1))
    else:
        print("Number Three Is biggest:" + str(num3))
else:
    print("Number Two is bigger than number One")
    if num2 > num3:
        print("Number Two biggest:" + str(num2))
    else:
        print("Number Three Is biggest:" + str(num3))
