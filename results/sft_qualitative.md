# Base vs SFT: 20 fixed prompts

- **Base:** `Qwen/Qwen2.5-0.5B` (4-bit NF4)
- **SFT:** same base + LoRA r=64, CodeAlpaca-20k (5000 examples), 2 epochs
- **Decoding:** greedy (`do_sample=False`), `max_new_tokens=200`, identical prompt template

---

## 1. Write a Python function that takes a list of numbers and returns the sum of all even numbers.

**Base**

```
```python
def sum_even_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total
```
```

**SFT**

```
def even_sum(numbers):
    even_sum = 0
    for num in numbers:
        if num % 2 == 0:
            even_sum += num
    return even_sum
```

---

## 2. Write a Python function to reverse a string without using slicing.

**Base**

```
def reverse_string(s):
    # Initialize an empty string to store the reversed string
    reversed_string = ""
    
    # Iterate through each character in the input string
    for i in range(len(s)):
        # Append each character to the reversed string
        reversed_string = reversed_string + s[i]
    
    # Return the reversed string
    return reversed_string
```

**SFT**

```
def reverse_string(string):
    rev_string = ""
    for i in range(len(string)-1, -1, -1):
        rev_string += string[i]
    return rev_string
```

---

## 3. Write a SQL query to find the second highest salary from an Employees table.

**Base**

```
```sql
SELECT MAX(salary) AS second_highest_salary
FROM Employees
ORDER BY salary DESC
LIMIT 1;
```
```

**SFT**

```
SELECT MAX(salary) AS second_highest_salary
FROM Employees;
```

---

## 4. Write a JavaScript function that debounces another function by a given delay.

**Base**

```
```javascript
function debounce(func, delay) {
  let timeout = null;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      func.apply(context, args);
    }, delay);
  };
}
```
```

**SFT**

```
function debounce(func, delay) {
    let timeout;
    return function() {
        clearTimeout(timeout);
        timeout = setTimeout(func, delay);
    };
}
```

---

## 5. Implement binary search in Python. Return the index of the target, or -1 if it is not present.

**Base**

```
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
```

**SFT**

```
def binary_search(arr, target):
    start = 0
    end = len(arr) - 1

    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            start = mid + 1
        else:
            end = mid - 1
    return -1
```

---

## 6. Write a Python class Stack with push, pop, and peek methods.

**Base**

```
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            return None
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0
```

**SFT**

```
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def peek(self):
        return self.stack[-1]
```

---

## 7. Convert this loop into a list comprehension:

result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x * x)

**Base**

```
```python
result = [x * x for x in range(10) if x % 2 == 0]
```
```

**SFT**

```
result = [x * x for x in range(10) if x % 2 == 0]
```

---

## 8. Fix this function so it handles division by zero:

def divide(a, b):
    return a / b

**Base**

```
def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    else:
        return a / b
```

**SFT**

```
def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    else:
        return a / b
```

---

## 9. Write a bash command to find all files larger than 100MB in the current directory.

**Base**

```
```bash
#!/bin/bash

# Find all files larger than 100MB in the current directory
find . -type f -size +100M -exec ls -l {} \;
```
```

**SFT**

```
find / -type f -size +100M
```

---

## 10. Write a Python function that reads a CSV file and returns a list of dictionaries.

**Base**

```
```python
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split('\n')
        data = [line.split(',') for line in data]
        return data

def create_dict(data):
    return {row[0]: {row[1]: row[2]} for row in data}

def main():
    file_path = 'data.csv'
    data = read_csv(file_path)
    dict_data = create_dict(data)
    print(dict_data)

if __name__ == '__main__':
    main()
```
```

**SFT**

```
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data = [dict(zip(file.fieldnames, file.readlines())) for file in csv.reader(file)]
    return data
```

---

## 11. Write a C function that swaps two integers using pointers.

**Base**

```
```c
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int a = 10, b = 20;
    swap(&a, &b);
    printf("a = %d, b = %d\n", a, b);
    return 0;
}
```
```

**SFT**

```
void swap(int *a, int *b) 
{ 
    int temp = *a; 
    *a = *b; 
    *b = temp; 
}
```

---

## 12. Write a regular expression that matches a valid email address.

**Base**

```
```python
import re

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._-]+@([a-zA-Z0-9._-]+\.[a-zA-Z]{2,})$'
    if re.match(pattern, email):
        return True
    else:
        return False

# Test the function
print(is_valid_email("example@example.com"))  # Output: True
print(is_valid_email("example@.com"))  # Output: False
```
```

**SFT**

```
^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$
```

---

## 13. Sort this list of dictionaries by the 'age' key:

people = [{'name': 'A', 'age': 30}, {'name': 'B', 'age': 25}]

**Base**

```
```python
people = [{'name': 'A', 'age': 30}, {'name': 'B', 'age': 25}]
people.sort(key=lambda x: x['age'])
print(people)
```
```

**SFT**

```
sorted_people = sorted(people, key=lambda k: k['age'])
```

---

## 14. Write a pytest unit test for a function add(a, b) that returns the sum of its arguments.

**Base**

```
```python
import pytest

def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(10, 20) == 30
    assert add(100, 200) == 300
```
```

**SFT**

```
import pytest

def test_add():
    assert add(1, 2) == 3
    assert add(10, 20) == 30
```

---

## 15. Write a Python decorator that measures how long a function takes to run.

**Base**

```
```python
import time
import functools

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time} seconds to run.")
        return result
    return wrapper
```
```

**SFT**

```
def measure_time(func):
    def inner_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return inner_func
```

---

## 16. Explain what a Python decorator is in two sentences.

**Base**

```
A decorator is a function that takes another function as an argument and returns a new function. It is used to modify the behavior of a function without changing its source code.
```

**SFT**

```
A Python decorator is a function that takes another function as an argument and adds some functionality to it. It can be used to extend the functionality of a function without modifying the original code.
```

---

## 17. Explain the difference between a list and a tuple in Python.

**Base**

```
A list is a collection of items, while a tuple is a collection of items with a fixed order. In Python, a list is a mutable data structure, while a tuple is a fixed order data structure. Lists are created using square brackets, while tuples are created using parentheses. Lists can be empty, while tuples cannot. Lists can be nested, while tuples cannot. Lists are mutable, while tuples are immutable. Lists are used to store collections of items, while tuples are used to store collections of items with a fixed order.
```

**SFT**

```
A list is a collection of items that are stored in an ordered sequence, while a tuple is a collection of items that are stored in an ordered sequence, but the order of the items is fixed. Lists are mutable, meaning they can be changed after they are created, while tuples are immutable, meaning they cannot be changed once they are created. Tuples are often used when we need to store a collection of items that need to be accessed by their index, while lists are used when we need to store a collection of items that need to be accessed by their position.
```

---

## 18. What is the average time complexity of inserting into a hash table, and why?

**Base**

```
The average time complexity of inserting into a hash table is O(1), as the hash table is a linear data structure and each element is inserted in a sequential manner. This means that the time complexity of inserting an element into the hash table is constant and does not depend on the size of the hash table or the number of elements in the hash table.
```

**SFT**

```
The average time complexity of inserting into a hash table is O(1), as the hash table is a linear data structure and each element is accessed sequentially.
```

---

## 19. Do the thing with the data.

**Base**

```
# In this exercise, we will be working with the data from the Titanic dataset. The Titanic dataset is a dataset containing information about passengers who boarded the Titanic. The dataset contains information about the passengers' age, gender, and whether they survived or not. We will be working with the data to answer questions about the survival of the passengers.

# To start, we will need to import the necessary libraries. In this case, we will be importing the pandas library, which is a powerful data manipulation library that allows us to work with data in a variety of formats.

# Next, we will need to load the Titanic dataset into a pandas DataFrame. We can do this by using the pandas library's read_csv function, which takes in the file path to the dataset as an argument and returns a DataFrame containing the data.

# Once we have loaded the data into a DataFrame, we can start by examining the data. We can do this by using the DataFrame's head function, which returns the first few rows
```

**SFT**

```
The data should be cleaned and transformed to a usable format.
```

---

## 20. Write a function that returns the largest prime number.

**Base**

```
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def largest_prime(n):
    if n < 2:
        return None
    for i in range(2, int(n**0.5) + 1):
        if is_prime(i):
            return i
    return None

def largest_prime(n):
    if n < 2:
        return None
    for i in range(2, int(n**0.5) + 1):
        if is_prime(i):
            return i
    return None
```
```

**SFT**

```
def largest_prime(n):
    largest = 0
    for i in range(2, n):
        if is_prime(i):
            largest = max(largest, i)
    return largest

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True
```

