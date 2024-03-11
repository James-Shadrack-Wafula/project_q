
# ================ V1 =====================
# import sympy as sp

# # Define the symbols used in the equation
# x, y, z = sp.symbols('x y z')

# # Input the mathematical expression as a string
# equation_str = input("Enter a mathematical expression: ")

# # Parse the expression into a SymPy expression
# equation = sp.sympify(equation_str)

# try:
#     # Try to solve the equation
#     solution = sp.solve(equation, x)

#     if len(solution) == 0:
#         print("No solution found.")
#     else:
#         print("Solutions:")
#         for sol in solution:
#             print(sol)

# except sp.SympifyError:
#     print("Invalid input. Please enter a valid mathematical expression.")
# except Exception as e:
#     print("An error occurred:", e)



# ====================== v2 =====================================
import sympy as sp

# Define the symbols used in the equation
x, y, z = sp.symbols('x y z')

# Input the mathematical expression as a string
equation_str = input("Enter a mathematical expression: ")

# Parse the expression into a SymPy expression
equation = sp.sympify(equation_str)

try:
    # Try to solve the equation
    solution = sp.solve(equation, x, explain=True)

    if len(solution) == 0:
        print("No solution found.")
    else:
        print("Solutions:")
        for sol, explanation in solution:
            print("Solution:", sol)
            print("Explanation:")
            for step in explanation:
                print(step)

except sp.SympifyError:
    print("Invalid input. Please enter a valid mathematical expression.")
except Exception as e:
    print("An error occurred:", e)




# ============================ v3 ========================
'''

# Example of using the code to solve an equation

# Input the mathematical expression as a string
equation_str = "2*x + 3 - 7"

# Parse the expression into a SymPy expression
equation = sp.sympify(equation_str)

try:
    # Try to solve the equation symbolically
    symbolic_solution = sp.solve(equation, x)

    if symbolic_solution:
        print("Symbolic Solutions:")
        for sol in symbolic_solution:
            print(f"x = {sol}")
    else:
        print("No symbolic solution found.")

    # Try to solve the equation numerically if no symbolic solution
    if not symbolic_solution:
        numeric_solution = sp.solve(equation, x, numerical=True)

        if numeric_solution:
            print("Numeric Solutions:")
            for sol in numeric_solution:
                print(f"x ≈ {sol}")
        else:
            print("No numeric solution found.")

except sp.SympifyError:
    print("Invalid input. Please enter a valid mathematical expression.")
except Exception as e:
    print("An error occurred:", e)


'''