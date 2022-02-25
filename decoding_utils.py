from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

'''
    from N. David Mermin, “Lecture Notes on Quantum Computation,” Chapter 5, Cornell University, Physics 481-681, CS 483; Spring, 2006
'''
def encode_5_qubit_mermin(input_state):
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)

    initial_state = qc.initialize(input_state, 0)

    qc.z(0)
    qc.h(0)
    qc.z(0)

    qc.cx(0,1)
    qc.h(0)
    qc.h(1)

    qc.cx(0,2)
    qc.cx(1,2)
    qc.h(2)

    qc.cx(0,3)
    qc.cx(2,3)
    qc.h(0)
    qc.h(3)

    qc.cx(0,4)
    qc.cx(1,4)
    qc.cx(2,4)
    qc.h(0)
    qc.h(1)

    return qc


'''
    Network for encoding 5 qubit code, from fig 4.2 Gottesmann PhD thesis. 
'''
def encode_5_qubit(input_state):
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)

    initial_state = qc.initialize(input_state, 0)

    qc.h(range(1,5))
    qc.z([1,4])

    qc.cz(4,3)
    qc.cz(4,1)
    qc.cy(4,0)

    qc.cz(3,2)
    qc.cz(3,1)
    qc.cx(3,0)

    qc.cz(2,4)
    qc.cz(2,3)
    qc.cx(2,0)

    qc.cz(1,4)
    qc.cz(1,2)
    qc.cy(1,0)

    return qc

# swap tests to measure matrix elements
def hadamard_test(num_qubits, qc, target_circuit):
    ancilla = QuantumRegister(1)
    readout_bit = ClassicalRegister(1)
    controlled_target_circuit = target_circuit.control(1)
    qc.add_register(ancilla, readout_bit)
    qc.h(num_qubits) # this is ancilla
    qc.compose(controlled_target_circuit, qubits=range(num_qubits,-1,-1), inplace=True) # last qubit controls first amount
    qc.h(num_qubits)
    qc.measure(num_qubits,0)

    return qc


