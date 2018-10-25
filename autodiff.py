import numpy as np
import initializers as init

class Node(object):
    """Node in a computation graph. """
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.trainable = False
        self.initial_value = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Multiply by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            raise NotImplementedError

        return new_node



    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__


def Variable(name = '', shape = None, initializer = init.zeros_initializer()):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    variable_node = placeholder_op()
    variable_node.name = name
    variable_node.initial_value = initializer(shape)
    variable_node.trainable = True
    return variable_node

def placeholder(name):
    """User defined variables in an expression.
        e.g. x = placeholder(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node

class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.j

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s - %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of two input nodes, return result of element-wise subtraction """
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        raise NotImplementedError
        #return [output_grad, -output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        if isinstance(output_grad.op, OnesLikeOp):
            return [node.inputs[1], node.inputs[0]]
        else:
            return [output_grad*node.inputs[1], output_grad*node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of input node, return result of element-wise multiplication."""
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        return [node.const_attr * output_grad]

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given values of input nodes, return result of matrix multiplication."""
        if node.matmul_attr_trans_A:
            return np.dot(np.transpose(input_vals[0]), input_vals[1])
        elif node.matmul_attr_trans_B:
            return np.dot(input_vals[0], np.transpose(input_vals[1]))
        else:
            return np.dot(input_vals[0], input_vals[1])

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [matmul_op(output_grad, node.inputs[1], trans_B=True),
            matmul_op(node.inputs[0], output_grad, trans_A=True) ]


class TransposeOp(Op):
    """Op to take matrix tranpose of a node """
    def __call__(self, node_A):
        """

        Parameters
        ----------
        node_A:

        Returns
        -------

        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "TransposeOp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        return np.transpose(input_vals[0])

    def gradient(self, node, output_grad):
        pass


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Returns ones_like of the same shape as input."""
        if isinstance(input_vals[0], np.ndarray):
            return np.ones(input_vals[0].shape)
        else:
            return 1

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

class ExpXOp(Op):
    """ Op that represents e to the power of x """
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ExpXOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Given value of input node, return result of raising e to the power of input node."""
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        """Given gradient of exponent node, return gradient contribution to input."""
        return [output_grad * node]




class InverseOp(Op):
    """ Op that represents inverse of x. i.e.  1/x """
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "InverseOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """Given value of input node, return result of inverse of input node."""
        return 1/input_vals[0]

    def gradient(self, node, output_grad):
        """Given gradient of inverse node, return gradient contribution to input. derivative of 1/x is -1/(square(x)) """
        return [output_grad * node * node * -1]


class SizeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "SizeOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        return np.size(input_vals[0])

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]



class ReduceMeanOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceMeanOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        return np.mean(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * inverse(size(node.inputs[0]))]


class LogOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "LogOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Given value of input node, return result of log of input node."""
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        """Given gradient of log node, return gradient contribution to input. derivative of log x is 1/x """
        return [output_grad * inverse(node.inputs[0])]


class AssignOp(Op):
    def __call__(self, node_A, node_B):
        """ node_A is lhs, node_B is rhs, rhs will be assigned to lhs """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "AssignOp(%s = %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        """ to retain the proper dimension of lhs. in case of bias assignment if its not done then it was making the
        dimension of bias as (n, m) instead of (n, 1) """
        # target shape
        row, col = nval_map[node.inputs[0]].shape
        nval_map[node.inputs[0]] = nval_map[node.inputs[1]][0:row, 0:col]
        return nval_map[node.inputs[0]]

    def gradient(self, node, output_grad):
        pass


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "SoftmaxOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        a = input_vals[0]
        e_x = np.exp(a - np.max(a, axis=0), dtype=np.float32)
        sftmax = e_x / np.sum(e_x, axis=0, dtype=np.float32)
        return sftmax

    def gradient(self, node, output_grad):
        pass


class Softmax_With_Cross_EntropyOp(Op):
    def __call__(self, node_A, y):
        """ node_A will be matrix multiply node and node_b will be training labels """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, y]
        new_node.name = "Softmax_With_Cross_Entropy(logits=%s, labels=%s )" % (node_A.name, y.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        matmul_value, y = input_vals
        e_x = np.exp(matmul_value - np.max(matmul_value, axis=0), dtype=np.float32)
        sftmax = e_x / np.sum(e_x, axis=0, dtype=np.float32)
        clipped = np.clip(sftmax, 1e-10, 0.9999999)
        cross_ent = -np.sum(y * np.log(clipped) + (1 - y) * np.log(1 - clipped), axis=0, keepdims=True)
        return cross_ent

    def gradient(self, node, output_grad):
        matmul_node, y = node.inputs
        return [output_grad * (softmax(matmul_node) - y), oneslike_op(output_grad)]


class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "SigmoidOp(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        x = input_vals[0]
        return 1 / (1 + np.exp(-x))

    def gradient(self, node, output_grad):
        input_node = node.inputs[0]
        return [output_grad * ( sigmoid(input_node) * (-1 * (-1 + sigmoid(input_node))) )]


class Sigmoid_With_Cross_EntropyOp(Op):
    def __call__(self, node_A, y):
        """ node_A will be matrix multiply node and node_b will be training labels """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, y]
        new_node.name = "Sigmoid_With_Cross_Entropy(logits=%s, labels=%s )" % (node_A.name, y.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        node_val, y = input_vals
        cross_ent = -node_val * y + np.log(1 + np.exp(node_val))   # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        return cross_ent

    def gradient(self, node, output_grad):
        matmul_node, y = node.inputs
        return [output_grad * (sigmoid(matmul_node) - y), oneslike_op(output_grad)]


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReluOp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        node_val = input_vals[0]
        return np.maximum(0, node_val)

    def gradient(self, node, output_grad):
        input_node = node.inputs[0]
        return [output_grad * relu_derivative(input_node)]

class Relu_DerivativeOp(Op):
    def __call__(self, node_A):
        """ derivative of relu operation. It sets those elements to 1 which are greater than or equal to k
        and sets those elements to 0 which are less than k """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu_DerivativeOp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, nval_map = None):
        node_val = input_vals[0]
        return np.int64(node_val > 0)

    def gradient(self, node, output_grad):
        raise NotImplementedError




def matmul(x, y):
    matmul_node = matmul_op(x, y)
    return matmul_node

def expx(x):
    """ Creates the e to the power of x node """
    expx_node = expx_op(x)
    return expx_node

def inverse(x):
    """ Creates the inverse of x node """
    inverse_node = inverse_op(x)
    return inverse_node

def size(x):
    size_node = size_op(x)
    return size_node

def reduce_mean(x):
    rm_node = reduce_mean_op(x)
    return rm_node

def softmax(x):
    sftmax_node = softmax_op(x)
    return sftmax_node

def softmax_with_cross_entropy(logits, labels):
    softmax_with_ce_node = softmax_with_ce_op(logits, labels)
    return softmax_with_ce_node

def assign(x, y):
    assign_node = assign_op(x, y)
    return assign_node

def sigmoid(x):
    sigmoid_node = sigmoid_op(x)
    return sigmoid_node

def sigmoid_with_cross_entropy(logits, labels):
    sigmoid_with_ce_node = sigmoid_ce_op(logits, labels)
    return sigmoid_with_ce_node

def relu(x):
    relu_node = relu_op(x)
    return relu_node

def relu_derivative(x):
    relu_derivative_node = relu_derivative_op(x)
    return relu_derivative_node



# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
transpose_op = TransposeOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
expx_op = ExpXOp()
inverse_op = InverseOp()
reduce_mean_op = ReduceMeanOp()
size_op = SizeOp()
log_op = LogOp()
assign_op = AssignOp()
softmax_op = SoftmaxOp()
softmax_with_ce_op = Softmax_With_Cross_EntropyOp()
sigmoid_op = SigmoidOp()
sigmoid_ce_op = Sigmoid_With_Cross_EntropyOp()
relu_op = ReluOp()
relu_derivative_op = Relu_DerivativeOp()




class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """

        self.eval_node_list = eval_node_list
        self.node_to_val_map = {}
        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if node.trainable:
                self.node_to_val_map[node] = node.initial_value


    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        self.node_to_val_map.update( dict(feed_dict) )
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(self.eval_node_list)

        for node in topo_order:
            if not isinstance(node.op, PlaceholderOp):
                if isinstance(node.op, OnesLikeOp):
                    nd_var = node.inputs[0]
                    val = self.node_to_val_map[nd_var]
                    self.node_to_val_map[node] = node.op.compute(node, [val])
                else:
                    self.node_to_val_map[node] = node.op.compute(node, [self.node_to_val_map[inp] for inp in node.inputs], nval_map = self.node_to_val_map)

        # Collect node values.
        node_val_results = []
        assign_op_values = [self.node_to_val_map[node] for node in self.eval_node_list
                                                           if isinstance(node.op, AssignOp)]
        node_val_results.append(assign_op_values)
        other_op_values = [self.node_to_val_map[node] for node in self.eval_node_list
                                                          if not isinstance(node.op, AssignOp)]
        node_val_results.extend(other_op_values)

        return node_val_results

    def compute_value(self, node_list, feed_dict):
        self.eval_node_list = node_list
        return self.run(feed_dict)

def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.


    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]

    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    reverse_topo_order = list(reverse_topo_order)
    for node in reverse_topo_order[1:]:
        node_to_output_grads_list[node] = []

    for node in reverse_topo_order:
        node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
        grad_inputs = node.op.gradient(node, node_to_output_grad[node])

        if grad_inputs is not None:
            assert len(grad_inputs) == len(node.inputs)
            for inp, grad in zip(node.inputs, grad_inputs):
                if isinstance(grad.op, OnesLikeOp):
                    node_to_output_grads_list[inp] += [oneslike_op(inp)]
                else:
                    node_to_output_grads_list[inp] += [grad]

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

##############################
####### Helper Methods #######
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
