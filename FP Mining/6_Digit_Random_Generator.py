import random
import csv

# Generate 1000 unique 6-digit numbers
unique_numbers = random.sample(range(100000, 1000000), 1000)

# Save to CSV file
with open("FP Mining/unique_6_digit_numbers.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["6_digit_number"])  # Optional header
    for number in unique_numbers:
        writer.writerow([number])

print("CSV file 'unique_6_digit_numbers.csv' created successfully.")
