numerical_grade = float(input("Provide the numerical grade from 0-100: "))
letter_grade = "A"
if numerical_grade >= 80 and numerical_grade < 90:
    letter_grade = "B"
elif numerical_grade >= 70 and numerical_grade < 80:
    letter_grade = "C"
elif numerical_grade >= 60 and numerical_grade < 70:
    letter_grade = "D"
elif numerical_grade < 60:
    letter_grade = "F"

print(f"The letter grade is {letter_grade}")
