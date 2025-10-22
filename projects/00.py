"""
R-to-Python Basics Demonstration
--------------------------------

This script illustrates basic R operations translated to Python, using:
- Arithmetic and division
- List/vector creation and concatenation
- Sequences and repetition
- Indexing, filtering, and logical operations
- Vectorized math operations with NumPy
- String operations
- Lists of lists and dictionaries
- Type checking and type conversion
- Complex numbers and floating-point sequences

Author: Your Name
Date: 2025-10-22
"""

import math
import string
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 1: Basic Arithmetic
# -----------------------------------------------------------------------------
print("# --- Basic Arithmetic ---")
a = 2
b = 3
c = a * b
d = a + b
print(c)  # 6
print(d)  # 5

print("\n# --- Division ---")
a = 3
b = 2
print(a / b)  # 1.5

# -----------------------------------------------------------------------------
# SECTION 2: Lists / Vectors (NumPy for R-like behavior)
# -----------------------------------------------------------------------------
print("\n# --- List Creation ---")
a_np = np.array([1, 4, 6, 7])
print(a_np)

print("\n# --- List Concatenation ---")
a_np1 = np.array([1, 2, 3])
b_np1 = np.array([4, 5, 6])
c_np1 = np.concatenate((a_np1, b_np1))
print(c_np1)
print("Type:", type(c_np1))
print("Element dtype:", c_np1.dtype)

print("\n# --- Range / Sequence Generation ---")
b_seq = np.arange(1, 11)  # Similar to R's 1:10
print(b_seq)

print("\n# --- Multi-line Calculation ---")
# Demonstrating Python multi-line expressions
result_multiline = 5 - \
                   + \
                   + 1
print(result_multiline)

result_simple = 5 - 1
print(result_simple)  # 4

# -----------------------------------------------------------------------------
# SECTION 3: Predefined Constants
# -----------------------------------------------------------------------------
print("\n# --- Predefined Constants ---")
print("Pi:", math.pi)
print("Lowercase letters:", list(string.ascii_lowercase))
print("Uppercase letters:", list(string.ascii_uppercase))

# -----------------------------------------------------------------------------
# SECTION 4: Boolean Operations and Type Checking
# -----------------------------------------------------------------------------
print("\n# --- Boolean Operations ---")
w = 5 > 2
print(w)

print("\n# --- Type Checking ---")
print("Is 2 numeric?", isinstance(2, (int, float)))
print("Is 'a' numeric?", isinstance("a", (int, float)))
print("Is 2.6 integer?", isinstance(2.6, int))

# -----------------------------------------------------------------------------
# SECTION 5: Complex Numbers
# -----------------------------------------------------------------------------
print("\n# --- Complex Numbers ---")
d_complex = 2 + 3j  # Python uses 'j' for imaginary unit
print("Complex number:", d_complex)
print("Type:", type(d_complex))
print("Real part as int:", int(d_complex.real))
print("String conversion:", str(d_complex))

# -----------------------------------------------------------------------------
# SECTION 6: Floating Point Sequences
# -----------------------------------------------------------------------------
print("\n# --- Floating Point Sequence ---")
fp_seq = np.arange(2.2, 8.3, 1.0)  # Includes endpoint approximately
print(fp_seq)

# -----------------------------------------------------------------------------
# SECTION 7: Sequence Generation and Repetition
# -----------------------------------------------------------------------------
print("\n# --- Sequence Generation and Repetition ---")
w_seq = np.arange(2, 21, 4)  # Similar to R's seq(2,20,4)
print(w_seq)
w_rep = np.tile(w_seq, 4)  # Repeat the sequence 4 times
print(w_rep)

# -----------------------------------------------------------------------------
# SECTION 8: Vector Indexing and Filtering
# -----------------------------------------------------------------------------
print("\n# --- Vector Indexing and Filtering ---")
v_np = np.array([10, 3, 543, 56, 23, 56])
print("Original array:", v_np)

# Indexing (Python 0-based vs R 1-based)
a_indexed = v_np[[2, 1, 0]]  # R: c(3,2,1)
print("Elements at indices [2,1,0]:", a_indexed)

# Remove elements: using slicing or mask
b_removed = v_np[1:]  # Remove first element
print("Array with first element removed:", b_removed)

indices_to_keep = [i for i in range(len(v_np)) if i not in [0, 2]]  # Remove first and third
z_removed = v_np[indices_to_keep]
print("Array with elements 0 and 2 removed:", z_removed)

# Logical comparisons and filtering
print("z_removed > 45:", z_removed > 45)
indices_gt_30 = np.where(v_np > 30)[0]
print("Indices where v > 30:", indices_gt_30)
elements_gt_25 = v_np[v_np > 25]
print("Elements > 25:", elements_gt_25)
elements_filtered = v_np[(v_np > 50) & (v_np < 100)]
print("Elements 50 < v < 100:", elements_filtered)

# -----------------------------------------------------------------------------
# SECTION 9: Vectorized Math Operations
# -----------------------------------------------------------------------------
print("\n# --- Vectorized Math Operations ---")
v1_np = np.array([2, 3, 6, 7, 9, 10, 1, 4, -8, 2, 3, 11])
v1_np[6] = 100
print("Modified v1:", v1_np)

v1_sorted_desc = np.sort(v1_np)[::-1]
print("v1 sorted descending:", v1_sorted_desc)

print("Sum:", np.sum(v1_np))
print("Mean:", np.mean(v1_np))
print("Standard deviation:", np.std(v1_np, ddof=1))
print("Min:", np.min(v1_np))
print("Max:", np.max(v1_np))
print("Length:", len(v1_np))

# -----------------------------------------------------------------------------
# SECTION 10: String Vector Operations
# -----------------------------------------------------------------------------
print("\n# --- String Vector Operations ---")
p_list = ["jan","feb","mar","apr","may","june","july","aug","sept","oct","nov","dec"]
print("Original list:", p_list)
print("Element at index 3:", p_list[3])
print("Sorted list:", sorted(p_list))
print("Slice [0:2]:", p_list[0:2])

# -----------------------------------------------------------------------------
# SECTION 11: Element-wise Vector Operations
# -----------------------------------------------------------------------------
print("\n# --- Element-wise Vector Operations ---")
v1_op = np.array([2, 3, 6, 7, 9, 10])
v2_op = np.array([1, 4, 8, 2, 3, 11])
print("v1 + v2:", v1_op + v2_op)
print("v1 - v2:", v1_op - v2_op)
print("v1 / v2:", v1_op / v2_op)
print("v1 % v2:", v1_op % v2_op)
print("sqrt(v1):", np.sqrt(v1_op))
print("v1 ** 3:", v1_op ** 3)

# -----------------------------------------------------------------------------
# SECTION 12: Lists of Lists / Dictionaries
# -----------------------------------------------------------------------------
print("\n# --- Lists of Lists ---")
t_py = [list(range(1, 5)), list(range(5, 9))]
print("List t:", t_py)

s_py = [list(range(1, 4)), True, False]
print("List s:", s_py)
print("Length of s:", len(s_py))

print("\n# --- Named Lists (Dictionaries) ---")
ls1_py = {'a': list(range(1, 6)), 'b': ["red", "blue"], 'd': [True, False, True]}
print("Dictionary ls1:", ls1_py)
print("ls1['a']:", ls1_py['a'])

# -----------------------------------------------------------------------------
# SECTION 13: Python Data Types and Conversion
# -----------------------------------------------------------------------------
print("\n# --- Python Data Types ---")
d_py = 4
i_py = 4
c_py = 'x'
com_py = 3 + 4j
bo_py = True & False

print(f"Type of {d_py}:", type(d_py))
print(f"Type of {i_py}:", type(i_py))
print(f"Type of '{c_py}':", type(c_py))
print(f"Type of {com_py}:", type(com_py))
print(f"Type of {bo_py}:", type(bo_py))

print("\n# --- Type Conversion ---")
x_py = str(com_py)
print("String conversion of complex:", x_py)
print("Type after conversion:", type(x_py))
