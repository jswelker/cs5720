import os

lines = open(os.path.join(os.getcwd(), "input.txt"), "r").readlines()
words = {}
for line in lines:
    for word in line.strip().split(" "):
        clean_word = word.strip()
        if clean_word != "" and clean_word not in words:
            words[clean_word] = 0
        words[clean_word] += 1

output = []
for line in lines:
    output.append(line.strip() + os.linesep)
for word in words:
    output.append(f"{word}: {words[word]}{os.linesep}")
with open(os.path.join(os.getcwd(), "output.txt"), "w") as f:
    f.writelines(output)
