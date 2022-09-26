import re

TXT_DIR_NAME = "./datasets/children/asd/"
TXT_FILE_NAME = "KKI_0050792.txt"
CSV_DIR_NAME = "./datasets/csv/"
CSV_FILE_NAME = "child_asd.csv"

with open(TXT_DIR_NAME + TXT_FILE_NAME,"r") as file: 
    text = file.readlines()

# Assuming brain network format of 116x116 matrix of 0s and 1s separated by spaces.
first_row = "_"
for i in range(116):
    first_row += f",{i}"
first_row += "\n"

for i in range(116):
    text[i] = re.sub(" ", ",", text[i])
    text[i] = f"{i}," + text[i]

new_text = [first_row] + text

with open(CSV_DIR_NAME + CSV_FILE_NAME, "w") as file: 
    file.writelines(new_text)