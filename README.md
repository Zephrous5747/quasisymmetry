# Approximate symmetry finder (small systems)

This is a summary of the module that I am making by stitching together Praveen’s code with Linjun’s and mine.

#### make\_pyscf\_hamiltonian.py

Can be used to produce a bunch of example Hamiltonians from a hard-coded list. Probably not a good practice but whatever.

#### show\_symmetries.py

Given an electronic structure Hamiltonian, calculates the quartets and shows them as a plot

#### find\_pauli\_symmetries.py

Finds symmetries via beam search (or something else that you want)  
input:

1. Hamiltonian  
2. the reference cost function (“fci”, “hf”, “cisd”)  
3. cost function  
4. keyword –senquart that constrains the symmetries to seniorities and quartets.

Output:

1. List of Pauli symmetries. If they are all Zs, also spits out a parity matrix.

#### optimize\_symmetries.py

Input arguments and keywords:

1. Hamiltonian. Checkfile or FCIDUMP or of.QubitOperator (?).  
2. Parity matrix of the symmetries  
3. Reference state: \--reference, “fci”, “hf”, “cisd” (defaults to fci)  
4. Cost function: variance to ref, NC to ref (defaults to NC)  
5. x0: optional initial guess for orbital rotations

Returns:

1. Optimized orbitals, if the symmetries were supplied with parity  
2. List of Pauli operators, if we used beam search.

#### metrics.py

Inputs:

1. Hamiltonian  
2. Parity matrix  
3. rotation (optional)

Outputs:

1. Decoupled energy  
2. K: number of sector eigenstates needed to reach chemical accuracy  
3. Which sectors do these eigenstates come from

