user_input = input("Type something: ")
substr = user_input[0:-2]
print("Now we take off the last 2 characters and print it reversed.")
print(substr[::-1])
